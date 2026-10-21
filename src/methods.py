"""ทะเบียนวิธีที่ใช้ทดลอง ตรงกับตาราง Comparison Methods ใน RD_v2 ส่วนที่ 3.2

แต่ละวิธีกำหนด:
    inputs : feature ที่ต้องใช้ {ชื่อ argument ของ forward: spec ของ features.py}
    build  : ฟังก์ชันสร้างโมเดล
    optim  : optimizer และ hyperparameter ตามงานต้นฉบับ (ค่าที่ต้นฉบับไม่ระบุ ใช้ค่าเริ่มต้นที่ใช้กันทั่วไป)
    smote  : ทำ SMOTE เฉพาะชุด train หรือไม่
    role   : proposed / ablation / competitor
ทุกวิธีใช้ split, preprocessing (resample, band-pass, ความยาวคลิป), STFT, epoch สูงสุด,
early stopping, augmentation และ loss ถ่วงน้ำหนักคลาสชุดเดียวกัน (กำหนดใน train.py)
"""
from . import models as M

MEL = {"name": "mel", "scale": "log"}
MFCC_DELTA = {"name": "mfcc_delta", "n_mfcc": 40}


def _stack3(n_mfcc, size):
    """Mel / MFCC / Chroma แต่ละตัว resize เป็นขนาดเดียวกันแล้วซ้อนเป็น 3 channel (ทำใน train.py)"""
    return {"stack": [{"name": "mel", "scale": "log", "resize": size},
                      {"name": "mfcc", "n_mfcc": n_mfcc, "resize": size},
                      {"name": "chroma", "resize": size}]}


def proposed_inputs(aux, mel_scale="log"):
    d = {"mel": {"name": "mel", "scale": mel_scale}}
    if aux == "mfcc_delta":
        d["aux"] = MFCC_DELTA
    elif aux == "mfcc":
        d["aux"] = {"name": "mfcc", "n_mfcc": 40}
    return d


def _proposed(backbone, aux):
    return lambda a: M.ProposedNet(backbone=backbone, aux=aux, fusion=a.fusion, drop_path=a.drop_path,
                                   audioset_ckpt=a.audioset_ckpt if backbone == "audioset" else None,
                                   imagenet_pretrained=not a.no_pretrained)


# batch 16: ConvNeXt-Tiny กับ input 801×224 ใช้ VRAM ราว 4–5 GB ต่อ 16 ตัวอย่างเมื่อเปิด AMP
ADAMW = {"name": "adamw", "lr": 1e-4, "weight_decay": 0.05, "batch_size": 16, "scheduler": "cosine"}

METHODS = {
    # ---------------- โมเดลที่เสนอ + ablation ----------------
    "proposed": dict(role="proposed", desc="ConvNeXt-Tiny (AudioSet) + Mel + MFCC + ΔMFCC (SE late fusion)",
                     inputs=lambda a: proposed_inputs("mfcc_delta", a.mel_scale),
                     build=_proposed("audioset", "mfcc_delta"), optim=ADAMW),
    "abl_mel_only": dict(role="ablation", desc="ConvNeXt-Tiny (AudioSet) + Mel อย่างเดียว [RQ2]",
                         inputs=lambda a: proposed_inputs(None, a.mel_scale),
                         build=_proposed("audioset", None), optim=ADAMW),
    "abl_mel_mfcc": dict(role="ablation", desc="ConvNeXt-Tiny (AudioSet) + Mel + MFCC (ไม่มี Δ) [RQ2]",
                         inputs=lambda a: proposed_inputs("mfcc", a.mel_scale),
                         build=_proposed("audioset", "mfcc"), optim=ADAMW),
    "abl_imagenet": dict(role="ablation", desc="ConvNeXt-Tiny (ImageNet) + Mel + MFCC + ΔMFCC [แยกผล pretrain]",
                         inputs=lambda a: proposed_inputs("mfcc_delta", a.mel_scale),
                         build=_proposed("imagenet", "mfcc_delta"), optim=ADAMW),
    # ablation ทีละเทคนิค: เหมือน proposed ทุกอย่าง ยกเว้นเทคนิคที่ระบุ (overrides ทับค่าใน train.py)
    **{k: dict(role="ablation", desc=f"proposed แต่{d}", inputs=lambda a: proposed_inputs("mfcc_delta", a.mel_scale),
               build=_proposed("audioset", "mfcc_delta"), optim=ADAMW, overrides=o)
       for k, d, o in [
           ("abl_no_pretrain", "ไม่ใช้น้ำหนัก pretrain (ConvNeXt เริ่มจากสุ่ม)", {"audioset_ckpt": ""}),
           ("abl_concat", "ไม่ใช้ SE gating (ต่อ embedding ตรงๆ)", {"fusion": "concat"}),
           ("abl_no_bandpass", "ไม่ทำ band-pass 120–1800 Hz", {"no_bandpass": True}),
           ("abl_pcen", "ใช้ PCEN แทน log-Mel", {"mel_scale": "pcen"}),
           ("abl_no_specaug", "ไม่ใช้ SpecAugment", {"no_specaug": True}),
           ("abl_no_mixup", "ไม่ใช้ mixup", {"mixup_alpha": 0.0}),
           ("abl_no_classweight", "ไม่ถ่วงน้ำหนักคลาสใน loss", {"no_class_weight": True}),
           ("abl_no_droppath", "ไม่ใช้ drop path", {"drop_path": 0.0}),
       ]},

    # ---------------- คู่เทียบ 6 งาน (ทำซ้ำบน protocol เดียวกัน) ----------------
    "wanasinghe_cnn": dict(role="competitor", desc="[1] Multi-feature CNN: Mel/MFCC/Chroma 3ch (Wanasinghe 2024)",
                           inputs=lambda a: {"x": _stack3(128, [128, 264])},
                           build=lambda a: M.WanasingheCNN(3),
                           optim={"name": "adam", "lr": 1e-4, "batch_size": 32}),
    "dongre_cnn_bigru": dict(role="competitor", desc="[2] CNN-BiGRU + SMOTE บน MFCC (Dongre 2025)",
                             inputs=lambda a: {"x": {"name": "mfcc", "n_mfcc": 40}},
                             build=lambda a: M.DongreCNNBiGRU(40),
                             optim={"name": "adam", "lr": 1e-3, "batch_size": 32, "scheduler": "plateau"},
                             smote=True),
    "shakya_vgg16": dict(role="competitor", desc="[3] VGG16 (ImageNet) + Mel 224×224 fine-tune block5 (Shakya 2026)",
                         inputs=lambda a: {"x": {"name": "mel", "scale": "log", "resize": [224, 224]}},
                         build=lambda a: M.ImageNetCNN("vgg16", not a.no_pretrained, train_last_block_only=True,
                                                       hidden=256, dropout=0.5),
                         optim={"name": "adam", "lr": 1e-6, "batch_size": 32}),
    "puneet_mobilenetv2": dict(role="competitor", desc="[4] MobileNetV2 (ImageNet) + STFT spectrogram (Puneet 2025)",
                               inputs=lambda a: {"x": {"name": "spec", "n_fft": 1024, "win_length": 128,
                                                       "hop_length": 64, "resize": [224, 224]}},
                               build=lambda a: M.ImageNetCNN("mobilenetv2", not a.no_pretrained, dropout=0.2),
                               optim={"name": "adam", "lr": 1e-4, "batch_size": 32}),
    "park_inception": dict(role="competitor", desc="[5] Inception lightweight CNN: Mel/Chroma/MFCC (Park 2025)",
                           inputs=lambda a: {"x": _stack3(40, [128, 216])},
                           build=lambda a: M.ParkInceptionCNN(3),
                           optim={"name": "adam", "lr": 1e-4, "batch_size": 32}),
    "engin_densenet201": dict(role="competitor", desc="[6] DenseNet201 (ImageNet) + Gammatonegram 256×256 (Engin 2026)",
                              inputs=lambda a: {"x": {"name": "gammatone", "n_ch": 64, "resize": [256, 256]}},
                              build=lambda a: M.ImageNetCNN("densenet201", not a.no_pretrained, dropout=0.0),
                              optim={"name": "adam", "lr": 1e-3, "batch_size": 16, "scheduler": "plateau"}),
}

ABLATIONS = [k for k, v in METHODS.items() if v["role"] == "ablation"]
COMPETITORS = [k for k, v in METHODS.items() if v["role"] == "competitor"]

# ชุดที่ใช้ใน Proof of Concept (RD 3.2: วิธีที่ 1–4 และ 7 เทียบกับโมเดลที่เสนอบน ICBHI)
POC_METHODS = ["proposed", "abl_mel_only", "wanasinghe_cnn", "dongre_cnn_bigru",
               "shakya_vgg16", "puneet_mobilenetv2"]
FULL_METHODS = list(METHODS)
