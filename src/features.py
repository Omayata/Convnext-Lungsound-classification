"""แปลงเสียงเป็น feature แบบภาพ (ตาม RD_v2 ส่วนที่ 3)

ทุก feature คำนวณด้วย torch บน GPU เป็น batch แล้ว cache ไว้ใน cache/ เพื่อใช้ซ้ำทุก seed

feature ที่มี:
    mel        : log-Mel (หรือ PCEN) 50–2000 Hz          -> ทางหลักของโมเดลที่เสนอ
    mfcc_delta : MFCC + ΔMFCC ซ้อน 2 channel           -> ทางเสริมของโมเดลที่เสนอ
    mfcc       : MFCC อย่างเดียว (ใช้ใน ablation Mel+MFCC)
    chroma     : Chromagram (ใช้เฉพาะคู่เทียบ Wanasinghe / Park ตามต้นฉบับ)
    spec       : log STFT spectrogram (คู่เทียบ Puneet)
    gammatone  : Gammatonegram 64 ช่อง (คู่เทียบ Engin)
"""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from .utils import CACHE_DIR

# พารามิเตอร์ STFT ชุดเดียวกันทุกวิธี (RD ส่วนที่ 4: Feature Engineering)
STFT = dict(n_fft=2048, win_length=400, hop_length=160)   # window 25 ms, hop 10 ms ที่ 16 kHz
MEL = dict(n_mels=128, fmin=50.0, fmax=2000.0)


def _stft_power(x, n_fft, win_length, hop_length):
    win = torch.hann_window(win_length, device=x.device)
    s = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=win,
                   center=True, pad_mode="reflect", return_complex=True)
    return s.abs() ** 2                                     # [B, F, T]


def _power_to_db(p, amin=1e-10):
    return 10.0 * torch.log10(torch.clamp(p, min=amin))


def _delta(x, width=9):
    """Δ แบบ regression (เหมือน librosa/HTK) ตามแกนเวลา x: [B, C, T]"""
    n = width // 2
    k = torch.arange(-n, n + 1, dtype=x.dtype, device=x.device)
    k = (k / (k ** 2).sum()).flip(0).view(1, 1, -1)
    b, c, t = x.shape
    y = F.pad(x.reshape(b * c, 1, t), (n, n), mode="replicate")
    return F.conv1d(y, k).reshape(b, c, t)


def _pcen(e, sr, hop, time_constant=0.06, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """Per-Channel Energy Normalization (Wang et al., 2017) ใช้แทน log เมื่อเปิด --mel_scale pcen"""
    t_frames = time_constant * sr / hop
    s = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    m = torch.empty_like(e)
    m[..., 0] = e[..., 0]
    for t in range(1, e.shape[-1]):
        m[..., t] = (1 - s) * m[..., t - 1] + s * e[..., t]
    return (e / (eps + m) ** alpha + delta) ** r - delta ** r


def gammatone_weights(sr, n_fft, n_ch=64, fmin=50.0, fmax=None, order=4):
    """เมทริกซ์น้ำหนัก gammatone บนแกน FFT

    ใช้ magnitude response ของ gammatone filter อันดับ 4 แบบประมาณ
    |H(f)| = [1 + ((f - fc) / b)^2]^(-order/2),  b = 1.019 * ERB(fc)
    ความถี่กลางเว้นระยะแบบ ERB (Glasberg & Moore, 1990) จาก fmin ถึง Nyquist ตาม Engin et al.
    """
    fmax = fmax or sr / 2
    ear_q, min_bw = 9.26449, 24.7
    erb_rate = lambda f: ear_q * np.log(1 + f / (ear_q * min_bw))
    inv = lambda e: (np.exp(e / ear_q) - 1) * ear_q * min_bw
    cf = inv(np.linspace(erb_rate(fmin), erb_rate(fmax * 0.95), n_ch))
    b = 1.019 * (cf / ear_q + min_bw)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    w = (1 + ((freqs[None, :] - cf[:, None]) / b[:, None]) ** 2) ** (-order / 2)
    return w.astype(np.float32)


class FeatureBank:
    """คำนวณ feature ตาม spec ที่แต่ละวิธีต้องการ

    spec ตัวอย่าง: {"name": "mel", "scale": "log", "resize": [224, 224]}
    """

    def __init__(self, sr=16000, device="cpu"):
        import librosa

        self.sr, self.device = sr, torch.device(device)
        self._mel_fb = torch.tensor(librosa.filters.mel(sr=sr, n_fft=STFT["n_fft"], n_mels=MEL["n_mels"],
                                                        fmin=MEL["fmin"], fmax=MEL["fmax"]),
                                    device=self.device)                       # [n_mels, F]
        self._chroma_fb = torch.tensor(librosa.filters.chroma(sr=sr, n_fft=STFT["n_fft"], n_chroma=12),
                                       device=self.device)
        from scipy.fft import dct
        self._dct = torch.tensor(dct(np.eye(MEL["n_mels"]), type=2, norm="ortho", axis=0),
                                 dtype=torch.float32, device=self.device)     # [n_mels, n_mels]
        self._gt = {}

    # ---- feature พื้นฐาน -------------------------------------------------------
    def mel_power(self, x):
        return torch.matmul(self._mel_fb, _stft_power(x, **STFT))              # [B, M, T]

    def log_mel(self, x):
        return _power_to_db(self.mel_power(x))

    def mfcc(self, x, n_mfcc=40):
        return torch.einsum("km,bmt->bkt", self._dct[:n_mfcc], self.log_mel(x))

    def compute(self, x, spec):
        name = spec["name"]
        if name == "mel":
            if spec.get("scale", "log") == "pcen":
                out = _pcen(self.mel_power(x) * (2 ** 31), self.sr, STFT["hop_length"])
            else:
                out = self.log_mel(x)
            out = out[:, None]
        elif name == "mfcc":
            out = self.mfcc(x, spec.get("n_mfcc", 40))[:, None]
        elif name == "mfcc_delta":
            m = self.mfcc(x, spec.get("n_mfcc", 40))
            out = torch.stack([m, _delta(m)], dim=1)
        elif name == "chroma":
            c = torch.matmul(self._chroma_fb, _stft_power(x, **STFT))
            out = (c / (c.amax(dim=1, keepdim=True) + 1e-10))[:, None]          # normalize ต่อเฟรมแบบ librosa
        elif name == "spec":
            p = _stft_power(x, spec.get("n_fft", 1024), spec.get("win_length", 128), spec.get("hop_length", 64))
            out = _power_to_db(p)[:, None]
        elif name == "gammatone":
            n_fft = 512
            key = (n_fft, spec.get("n_ch", 64))
            if key not in self._gt:
                self._gt[key] = torch.tensor(gammatone_weights(self.sr, n_fft, key[1]), device=self.device)
            p = _stft_power(x, n_fft, 400, 160).sqrt()                        # 25 ms / 10 ms ตาม Engin
            out = (20 * torch.log10(torch.matmul(self._gt[key], p) + 1e-8))[:, None]
        else:
            raise ValueError(name)
        if spec.get("resize"):
            out = F.interpolate(out, size=tuple(spec["resize"]), mode="bilinear", align_corners=False)
        return out                                                               # [B, C, H(freq), W(time)]


def spec_key(spec, wave_tag, sr):
    s = json.dumps({"spec": spec, "wave": wave_tag, "sr": sr, "stft": STFT, "mel": MEL}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


@torch.no_grad()
def extract(waves, spec, wave_tag, sr=16000, device="cpu", batch_size=64, use_cache=True, verbose=True):
    """คำนวณ feature ของทุกตัวอย่างแล้วคืน np.float16 [N, C, H, W]"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"feat_{spec['name']}_{spec_key(spec, wave_tag, sr)}.npy"
    if use_cache and f.exists():
        return np.load(f, mmap_mode="r")
    bank = FeatureBank(sr, device)
    out = None
    for i in range(0, len(waves), batch_size):
        x = torch.from_numpy(np.asarray(waves[i:i + batch_size], dtype=np.float32)).to(bank.device)
        y = bank.compute(x, spec).cpu().numpy().astype(np.float16)
        if out is None:
            out = np.zeros((len(waves),) + y.shape[1:], dtype=np.float16)
        out[i:i + len(y)] = y
    if verbose:
        print(f"  feature {spec['name']}: {out.shape}")
    if use_cache:
        np.save(f, out)
    return out
