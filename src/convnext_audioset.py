"""ConvNeXt-Tiny ที่ฝึกบน AudioSet (Pellegrini et al., Interspeech 2023)

เขียนโครงสร้างใหม่ให้ชื่อ parameter ตรงกับ checkpoint ของ topel/audioset-convnext-inf
(third_party/1_models/audioset-convnext-inf/.../convnext.py) แต่ไม่ต้องพึ่ง torchlibrosa
เพราะเราคำนวณ log-Mel เองใน features.py

checkpoint ที่รองรับ:
    - convnext_tiny_471mAP.pth   (Zenodo 8020843)
    - model.safetensors          (Hugging Face: topel/ConvNeXt-Tiny-AT)
ให้วางไว้ที่ checkpoints/ แล้วชี้ด้วย --audioset_ckpt
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AUDIOSET_N_MELS = 224          # checkpoint ฝึกด้วย mel 224 bins (bn0 = BatchNorm2d(224))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps, self.data_format, self.shape = eps, data_format, (dim,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        y = (self.gamma * y).permute(0, 3, 1, 2)
        return x + self.drop_path(y)


class AudioSetConvNeXt(nn.Module):
    """รับ log-Mel [B, 1, T, 224] คืน embedding 768 มิติ (global mean+max pooling แบบต้นฉบับ)"""

    def __init__(self, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.1):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(AUDIOSET_N_MELS)
        stem = nn.Sequential(nn.Conv2d(1, dims[0], kernel_size=4, stride=4, padding=(4, 0)),
                             LayerNorm(dims[0], data_format="channels_first"))
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        dp = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages, cur = nn.ModuleList(), 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], dp[cur + j]) for j in range(depths[i])]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.out_dim = dims[-1]

    def forward(self, x):                       # x: [B, 1, T, F=224]
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
        x = x.mean(dim=3)                       # เฉลี่ยแกนความถี่
        x = x.max(dim=2).values + x.mean(dim=2) # max + mean ตามเวลา
        return self.norm(x)


def load_audioset_weights(model: AudioSetConvNeXt, ckpt_path):
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ checkpoint AudioSet ที่ {path}\n"
            "ดาวน์โหลด convnext_tiny_471mAP.pth จาก https://zenodo.org/record/8020843 "
            "หรือ model.safetensors จาก https://huggingface.co/topel/ConvNeXt-Tiny-AT แล้ววางใน checkpoints/")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # ตัดส่วนที่ไม่ใช้: STFT/logmel ของ torchlibrosa และ head 527 คลาสของ AudioSet
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("spectrogram_extractor", "logmel_extractor", "head_audioset", "spec_augmenter"))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint ขาด parameter: {missing[:8]} ... (รวม {len(missing)})")
    if unexpected:
        print(f"  [info] ข้าม key ที่ไม่ใช้ {len(unexpected)} ตัว เช่น {unexpected[:3]}")
    print(f"  โหลดน้ำหนัก AudioSet จาก {path.name} แล้ว")
    return model
