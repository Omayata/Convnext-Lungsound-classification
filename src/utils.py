"""เครื่องมือทั่วไป: seed, path, การอ่านไฟล์เสียง"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ให้พิมพ์ภาษาไทยได้แม้ console ของ Windows ไม่ใช่ UTF-8 หรือส่ง output ลงไฟล์
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]          # โฟลเดอร์ repo
DATASETS_DIR = ROOT / "datasets"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SPLITS_DIR = ROOT / "splits"
CKPT_DIR = ROOT / "checkpoints"

# ลำดับคลาสใช้ร่วมกันทุก dataset เพื่อให้ ICBHI score มีความหมายเดียวกัน
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
N_CLASSES = 4


def set_seed(seed: int, deterministic: bool = False):
    """คุมความสุ่มของน้ำหนักเริ่มต้น ลำดับข้อมูล และ augmentation"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_wav(path, target_sr: int) -> np.ndarray:
    """อ่าน wav เป็น mono float32 แล้ว resample เป็น target_sr

    ใช้ soundfile + scipy แทน torchaudio.load เพราะบน Windows ไม่ต้องตั้ง backend
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return x


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 5) -> np.ndarray:
    """Butterworth band-pass แบบ zero-phase (ตัดเสียงหัวใจและ noise นอกช่วงเสียงปอด)"""
    from scipy.signal import butter, sosfiltfilt

    high = min(high, sr / 2 * 0.99)
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def fit_length(x: np.ndarray, n: int, sr: int, mode: str = "repeat") -> np.ndarray:
    """ปรับความยาวให้เท่ากัน: ตัดส่วนเกิน หรือเติมโดยวนซ้ำ (แบบ Patch-Mix) / เติมศูนย์"""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) >= n:
        return x[:n].astype(np.float32)
    if mode == "zero":
        out = np.zeros(n, dtype=np.float32)
        start = (n - len(x)) // 2
        out[start:start + len(x)] = x
        return out
    reps = int(np.ceil(n / len(x)))
    out = np.tile(x, reps)[:n].astype(np.float32)
    fade = min(int(sr / 16), n // 4)                 # fade-out ท้ายคลิปกันรอยต่อกระโดด
    if fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
