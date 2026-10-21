"""เตรียม cache ของเสียงและตรวจข้อมูลก่อนรันจริง

    python -m src.prepare_data --dataset icbhi
    python -m src.prepare_data --dataset kauh --kauh_filter D
    python -m src.prepare_data --dataset hf_lung

พิมพ์: จำนวนตัวอย่าง, การกระจายคลาส, จำนวนผู้ป่วย/กลุ่ม, ขนาด train/val/test ของทุก fold
และสำหรับ ICBHI จะตรวจว่าได้ train 4,142 / test 2,756 รอบตาม official split
"""
import argparse

import numpy as np

from .datasets import load_dataset, get_split, DATASET_DEFAULTS
from .utils import CLASS_NAMES

ICBHI_EXPECTED = {"train": 4142, "test": 2756}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="icbhi", choices=list(DATASET_DEFAULTS))
    p.add_argument("--data_root", default=None)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--desired_length", type=float, default=None)
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--kauh_filter", default="D", choices=["B", "D", "E"])
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()

    band = None if a.no_bandpass else (120, 1800)
    meta, wav = load_dataset(a.dataset, a.sr, a.desired_length, band, a.data_root, a.kauh_filter, rebuild=a.rebuild)
    print(f"\n{a.dataset}: {len(meta)} ตัวอย่าง จาก {meta['file'].nunique()} ไฟล์, "
          f"{meta['group'].nunique()} กลุ่ม (ผู้ป่วย/การบันทึก), waveform {wav.shape}")
    counts = np.bincount(meta["label"], minlength=4)
    print("คลาส:", ", ".join(f"{c} {n}" for c, n in zip(CLASS_NAMES, counts)))
    skipped = meta.attrs.get("skipped") if hasattr(meta, "attrs") else None
    if skipped:
        print(f"ข้าม {len(skipped)} ไฟล์ (ดูตัวอย่าง 10 ไฟล์แรก):")
        for s in skipped[:10]:
            print("  ", s)

    if a.dataset == "icbhi":
        for sp, n in ICBHI_EXPECTED.items():
            got = int((meta["split"] == sp).sum())
            print(f"official {sp}: {got} รอบ {'OK' if got == n else f'ไม่ตรง (ควรได้ {n})'}")
        folds = ["official"]
    else:
        folds = list(range(5)) + (["official"] if a.dataset == "hf_lung" else [])
    for f in folds:
        tr, va, te = get_split(meta, a.dataset, f)
        y = meta["label"].to_numpy()
        print(f"fold {f}: train {len(tr)} | val {len(va)} | test {len(te)} | "
              f"test classes {np.bincount(y[te], minlength=4).tolist()}")


if __name__ == "__main__":
    main()
