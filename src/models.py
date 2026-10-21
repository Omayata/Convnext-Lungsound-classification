"""โมเดลที่เสนอ และโมเดลของงานคู่เทียบ 6 งาน (RD_v2 ส่วนที่ 3.1 และ 3.2)

ทุกโมเดลรับ dict ของ feature (แต่ละตัวเป็น [B, C, H=freq, W=time]) แล้วคืน logits [B, 4]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext_audioset import AudioSetConvNeXt, load_audioset_weights, AUDIOSET_N_MELS
from .utils import N_CLASSES


# ============================================================================
# โมเดลที่เสนอ: ConvNeXt-Tiny (AudioSet) + late fusion ของ MFCC/ΔMFCC ผ่าน SE gating
# ============================================================================
class AuxCNN(nn.Module):
    """ทางเสริม: CNN เล็ก 3 block สำหรับ MFCC + ΔMFCC (2 channel)"""

    def __init__(self, in_ch=2, chs=(32, 64, 128), out_dim=256):
        super().__init__()
        layers, c = [], in_ch
        for k in chs:
            layers += [nn.Conv2d(c, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.GELU(),
                       nn.Conv2d(k, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.GELU(),
                       nn.MaxPool2d(2)]
            c = k
        self.body = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(2 * c, out_dim), nn.GELU())
        self.out_dim = out_dim

    def forward(self, x):
        x = self.body(x)
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.proj(x)


class SEGate(nn.Module):
    """SE-style gating บนเวกเตอร์ที่ต่อกันแล้ว ให้โมเดลเรียนว่าควรเชื่อแต่ละทางแค่ไหน"""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim // reduction), nn.ReLU(inplace=True),
                                nn.Linear(dim // reduction, dim), nn.Sigmoid())
        self.last_gate = None

    def forward(self, z):
        g = self.fc(z)
        self.last_gate = g.detach()
        return z * g


class ProposedNet(nn.Module):
    """
    backbone : "audioset" (ตัวหลัก) หรือ "imagenet" (ablation แยกผลของการ pretrain บนเสียง)
    aux      : "mfcc_delta" (Mel+MFCC+Δ), "mfcc" (Mel+MFCC) หรือ None (Mel อย่างเดียว)
    fusion   : "se" (ค่าเริ่มต้น) หรือ "concat"
    """

    def __init__(self, backbone="audioset", aux="mfcc_delta", fusion="se", drop_path=0.1,
                 dropout=0.3, audioset_ckpt=None, imagenet_pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "audioset":
            self.backbone = AudioSetConvNeXt(drop_path_rate=drop_path)
            if audioset_ckpt:
                load_audioset_weights(self.backbone, audioset_ckpt)
            else:
                print("  [warn] ไม่ได้ระบุ --audioset_ckpt จะเริ่มจากน้ำหนักสุ่ม (ใช้ได้แค่ทดสอบ pipeline)")
            d = self.backbone.out_dim
        elif backbone == "imagenet":
            import timm
            self.backbone = timm.create_model("convnext_tiny.fb_in1k", pretrained=imagenet_pretrained,
                                              in_chans=1, num_classes=0, drop_path_rate=drop_path)
            d = self.backbone.num_features
        else:
            raise ValueError(backbone)
        self.aux = None
        if aux:
            self.aux = AuxCNN(in_ch=2 if aux == "mfcc_delta" else 1)
            d += self.aux.out_dim
        self.gate = SEGate(d) if fusion == "se" else nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, N_CLASSES))

    def forward(self, mel, aux=None):
        # mel: [B, 1, n_mels, T] -> [B, 1, T, 224] ตามรูปแบบที่ checkpoint AudioSet ถูกฝึกมา
        x = mel.transpose(2, 3)
        if x.shape[-1] != AUDIOSET_N_MELS:
            x = F.interpolate(x, size=(x.shape[2], AUDIOSET_N_MELS), mode="bilinear", align_corners=False)
        z = self.backbone(x)
        if self.aux is not None:
            z = torch.cat([z, self.aux(aux)], dim=1)
        return self.head(self.gate(z))

    def branch_weights(self):
        """ค่าเฉลี่ยของ gate แยกตามทาง (ใช้วิเคราะห์ว่าโมเดลพึ่ง Mel หรือ MFCC มากกว่า)"""
        g = getattr(self.gate, "last_gate", None)
        if g is None or self.aux is None:
            return None
        d_aux = self.aux.out_dim
        return {"gate_mel": g[:, :-d_aux].mean().item(), "gate_aux": g[:, -d_aux:].mean().item()}


# ============================================================================
# คู่เทียบที่ 1: Multi-feature lightweight CNN (Wanasinghe et al., 2024, IEEE Access)
# Mel / MFCC / Chroma ซ้อนเป็น 3 channel (128×264), conv 3×3 same + ReLU, maxpool 2×2,
# BN, dropout, GAP, dense, Adam lr 1e-4, class weight
# ============================================================================
class WanasingheCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        def blk(i, o, drop):
            return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(o),
                    nn.MaxPool2d(2)] + ([nn.Dropout(drop)] if drop else [])

        self.features = nn.Sequential(*blk(in_ch, 32, 0), *blk(32, 64, 0.2), *blk(64, 128, 0),
                                      *blk(128, 256, 0.3))
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                  nn.Linear(256, N_CLASSES))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


# ============================================================================
# คู่เทียบที่ 2: CNN-BiGRU + SMOTE (Dongre et al., 2025, ICCUBEA)
# MFCC -> conv + maxpool -> BiGRU -> dense + dropout  (SMOTE ทำใน train.py เฉพาะชุด train)
# ============================================================================
class DongreCNNBiGRU(nn.Module):
    def __init__(self, n_mfcc=40, hidden=128):
        super().__init__()
        chs, layers, c = (32, 64, 128), [], 1
        for k in chs:
            layers += [nn.Conv2d(c, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = k
        self.cnn = nn.Sequential(*layers)
        f = n_mfcc // 8
        self.gru = nn.GRU(c * f, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                  nn.Linear(128, N_CLASSES))

    def forward(self, x):                      # x: [B, 1, n_mfcc, T]
        x = self.cnn(x)                        # [B, C, F', T']
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        y, _ = self.gru(x)
        y = torch.cat([y.mean(dim=1), y.amax(dim=1)], dim=1)
        return self.head(y)


# ============================================================================
# คู่เทียบที่ 3, 4, 6: transfer learning จาก ImageNet (timm)
# ============================================================================
class ImageNetCNN(nn.Module):
    """VGG16 (Shakya), MobileNetV2 (Puneet), DenseNet201 (Engin)"""

    TIMM = {"vgg16": "vgg16.tv_in1k", "mobilenetv2": "mobilenetv2_100.ra_in1k",
            "densenet201": "densenet201.tv_in1k"}

    def __init__(self, arch, pretrained=True, train_last_block_only=False, hidden=0, dropout=0.5):
        super().__init__()
        import timm
        self.net = timm.create_model(self.TIMM[arch], pretrained=pretrained, num_classes=0)
        self.arch = arch
        if arch == "vgg16":
            self.net.pre_logits = nn.Identity()   # ไม่ใช้ fc6/fc7 เดิม (ตัด ~120M parameter ที่ไม่ได้ใช้)
        d = self.net.num_features          # vgg16: ใช้ conv block (512) + head ใหม่ แบบ Keras include_top=False
        if train_last_block_only and arch == "vgg16":
            # Shakya: fine-tune เฉพาะ conv block สุดท้าย (block5 = features.24 เป็นต้นไป)
            for name, p in self.net.named_parameters():
                if name.startswith("features."):
                    p.requires_grad = int(name.split(".")[1]) >= 24
        head = [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)] if hidden else []
        self.head = nn.Sequential(*head, nn.Dropout(dropout), nn.Linear(hidden or d, N_CLASSES))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)        # spectrogram ช่องเดียว -> 3 ช่องแบบภาพ
        if self.arch == "vgg16":
            z = self.net.forward_features(x).mean(dim=(2, 3))
        else:
            z = self.net(x)
        return self.head(z)


# ============================================================================
# คู่เทียบที่ 5: Inception-based lightweight CNN (Park et al., 2025, Appl. Sci.)
# Mel / Chroma / MFCC ซ้อน (128×216), Inception block แบบ asymmetric kernel (1×3/3×1, 1×5/5×1),
# 3 stage, ตัวจำแนกเป็น conv 3×3 + adaptive max pool + FC, lr 1e-4, batch 32
# ============================================================================
class InceptionBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        c = o // 4

        def cbr(a, b, k, p):
            return [nn.Conv2d(a, b, k, padding=p, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]

        self.b1 = nn.Sequential(*cbr(i, c, 1, 0))
        self.b3 = nn.Sequential(*cbr(i, c, 1, 0), *cbr(c, c, (1, 3), (0, 1)), *cbr(c, c, (3, 1), (1, 0)))
        self.b5 = nn.Sequential(*cbr(i, c, 1, 0), *cbr(c, c, (1, 5), (0, 2)), *cbr(c, c, (5, 1), (2, 0)))
        self.bp = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), *cbr(i, c, 1, 0))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class ParkInceptionCNN(nn.Module):
    def __init__(self, in_ch=3, chs=(32, 64, 128)):
        super().__init__()
        layers = [nn.Conv2d(in_ch, chs[0], 3, padding=1, bias=False), nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True)]
        c = chs[0]
        for k in chs:
            layers += [InceptionBlock(c, k), InceptionBlock(k, k),
                       nn.Conv2d(k, k, 1, bias=False), nn.BatchNorm2d(k), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = k
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d(1)]
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(c, N_CLASSES)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))
