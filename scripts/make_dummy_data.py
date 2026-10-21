"""สร้างข้อมูลจำลองขนาดเล็กของทั้ง 3 dataset เพื่อทดสอบ pipeline ก่อนมีข้อมูลจริง

    python scripts/make_dummy_data.py
    python -m src.train --dataset icbhi --data_root datasets_dummy/ICBHI_2017 --method proposed \
        --epochs 2 --audioset_ckpt "" --no_pretrained

เสียงเป็น noise + เสียงจำลอง (wheeze = โทนต่อเนื่อง, crackle = pulse สั้นๆ) ไม่ใช่เสียงปอดจริง
"""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets_dummy"
rng = np.random.default_rng(0)


def synth(sec, sr, crackle, wheeze):
    t = np.arange(int(sec * sr)) / sr
    x = 0.05 * rng.standard_normal(len(t))
    x += 0.2 * np.sin(2 * np.pi * 0.3 * t) * rng.standard_normal(len(t)) * 0.3   # "ลมหายใจ"
    if wheeze:
        x += 0.15 * np.sin(2 * np.pi * rng.uniform(300, 900) * t)
    if crackle:
        for c in rng.integers(0, len(t) - 50, int(sec * 6)):
            x[c:c + 40] += 0.6 * np.hanning(40) * np.sin(np.linspace(0, 20, 40))
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def icbhi(n_files=60):
    d = OUT / "ICBHI_2017"
    d.mkdir(parents=True, exist_ok=True)
    names = [l.split()[0] for l in (ROOT / "splits" / "icbhi_official_split.txt").read_text().splitlines() if l.strip()]
    for name in rng.choice(names, n_files, replace=False):
        sr = int(rng.choice([4000, 10000, 44100]))
        rows, t, audio = [], 0.0, []
        while t < 18:
            dur = rng.uniform(1.5, 4.0)
            c, w = int(rng.random() < 0.35), int(rng.random() < 0.25)
            audio.append(synth(dur, sr, c, w))
            rows.append(f"{t:.3f}\t{t + dur:.3f}\t{c}\t{w}")
            t += dur
        sf.write(d / f"{name}.wav", np.concatenate(audio), sr)
        (d / f"{name}.txt").write_text("\n".join(rows) + "\n")


def kauh(n_pat=30):
    d = OUT / "KAUH" / "Audio Files"
    d.mkdir(parents=True, exist_ok=True)
    kinds = [("N", 0, 0), ("Crep", 1, 0), ("E W", 0, 1), ("I E W", 0, 1), ("Crep E W", 1, 1), ("Bronchial", 0, 0)]
    for p in range(1, n_pat + 1):
        k, c, w = kinds[rng.integers(len(kinds))]
        x = synth(rng.uniform(5, 20), 4000, c, w)
        for mode in "BDE":
            sf.write(d / f"{mode}P{p}_Asthma,{k},P L L,{rng.integers(20, 80)},M.wav", x, 4000)


def hf_lung(n=80):
    for split in ("train", "test"):
        d = OUT / "HF_Lung_V1" / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n if split == "train" else n // 4):
            c, w = int(rng.random() < 0.4), int(rng.random() < 0.3)
            stem = f"steth_201906{rng.integers(10, 28)}_{rng.integers(8, 18):02d}_{rng.integers(0, 60):02d}_{i:02d}"
            sf.write(d / f"{stem}.wav", synth(15, 4000, c, w), 4000)
            lines = ["I 00:00:00.500 00:00:02.000", "E 00:00:02.000 00:00:03.500"]
            if c:
                lines.append("D 00:00:04.000 00:00:05.000")
            if w:
                lines.append("Wheeze 00:00:06.000 00:00:08.000")
            (d / f"{stem}_label.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    icbhi(); kauh(); hf_lung()
    print("สร้างข้อมูลจำลองที่", OUT)
