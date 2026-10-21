"""โหลด dataset เสียงปอด 3 ชุด และแบ่งข้อมูลแบบแยกผู้ป่วย (ตาม RD_v2 ส่วนที่ 2)

ทุก dataset ถูกแปลงเป็นตาราง metadata หนึ่งแถวต่อหนึ่งตัวอย่าง:
    uid, dataset, file, start, end, label, patient, group, split, device

label ใช้ลำดับเดียวกันทุกชุด: 0 normal, 1 crackle (DAS), 2 wheeze (CAS), 3 both

- ICBHI 2017 : ระดับรอบหายใจ, official split 60/40 (train 4,142 / test 2,756 รอบ)
- KAUH       : ระดับไฟล์, ใช้ฟิลเตอร์โหมดเดียว, stratified group 5-fold ตามผู้ป่วย
- HF_Lung_V1 : ระดับคลิป 15 วินาที, แปลง label เหตุการณ์ CAS/DAS เป็น 4 คลาส,
               group 5-fold (จัดกลุ่มตามการบันทึก เพราะไม่มีรหัสผู้ป่วย)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .utils import DATASETS_DIR, SPLITS_DIR, CACHE_DIR, load_wav, bandpass, fit_length

DATASET_DEFAULTS = {
    # desired_length: ความยาวคลิปหลังปรับ (วินาที)
    "icbhi": {"root": DATASETS_DIR / "ICBHI_2017", "desired_length": 8.0},
    "kauh": {"root": DATASETS_DIR / "KAUH", "desired_length": 15.0},
    "hf_lung": {"root": DATASETS_DIR / "HF_Lung_V1", "desired_length": 15.0},
}


def _lab(crackle: bool, wheeze: bool) -> int:
    return int(crackle) + 2 * int(wheeze)          # 0 N, 1 C, 2 W, 3 both


# ----------------------------------------------------------------------------
# ICBHI 2017
# ----------------------------------------------------------------------------
def build_icbhi(root: Path) -> pd.DataFrame:
    split_file = SPLITS_DIR / "icbhi_official_split.txt"
    split = pd.read_csv(split_file, sep=r"\s+", names=["file", "split"], engine="python")
    split = dict(zip(split["file"], split["split"]))

    rows = []
    wavs = sorted(Path(root).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"ไม่พบไฟล์ .wav ใน {root} (ดู datasets/README.md)")
    for wav in wavs:
        stem = wav.stem
        if stem not in split:
            raise KeyError(f"{stem} ไม่อยู่ใน official split (ชื่อไฟล์อาจต้องแก้ด้วย script.sh)")
        ann = pd.read_csv(wav.with_suffix(".txt"), sep="\t", header=None,
                          names=["start", "end", "crackle", "wheeze"])
        tok = stem.split("_")                          # 101_1b1_Al_sc_Meditron
        for i, r in ann.iterrows():
            rows.append(dict(uid=f"{stem}#{i}", dataset="icbhi", file=str(wav),
                             start=float(r.start), end=float(r.end),
                             label=_lab(r.crackle == 1, r.wheeze == 1),
                             patient=tok[0], group=tok[0], split=split[stem], device=tok[4]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# KAUH (Fraiwan et al., 2021, Mendeley Data)
# ชื่อไฟล์ เช่น "BP1_Asthma,I E W,P L L,70,M.wav"
#   ตัวแรก B/D/E = โหมดฟิลเตอร์ Bell / Diaphragm / Extended, P1 = รหัสผู้ป่วย
#   ช่องที่ 2 = ชนิดเสียง (N, Crep, C, E W, I E W, ...)
# ----------------------------------------------------------------------------
KAUH_CRACKLE = re.compile(r"\b(c|crep|crepitation|crackles?|fine crackles?|coarse crackles?)\b", re.I)
KAUH_WHEEZE = re.compile(r"\b(w|wheez\w*)\b", re.I)
KAUH_NORMAL = re.compile(r"\b(n|normal)\b", re.I)


def parse_kauh_sound(field: str):
    """แปลงช่องชนิดเสียงเป็น label 4 คลาส คืน None ถ้าเป็นเสียงชนิดอื่น (เช่น bronchial)"""
    f = field.replace("&", " ").replace("+", " ").replace("/", " ")
    c, w = bool(KAUH_CRACKLE.search(f)), bool(KAUH_WHEEZE.search(f))
    if c or w:
        return _lab(c, w)
    if KAUH_NORMAL.search(f):
        return 0
    return None


def build_kauh(root: Path, filter_mode: str = "D") -> pd.DataFrame:
    wavs = sorted(Path(root).rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"ไม่พบไฟล์ .wav ใน {root} (ดู datasets/README.md)")
    rows, skipped = [], []
    for wav in wavs:
        m = re.match(r"^\s*([BDE])\s*P\s*(\d+)", wav.stem, re.I)
        parts = wav.stem.split(",")
        if not m or len(parts) < 2:
            skipped.append((wav.name, "อ่านรูปแบบชื่อไฟล์ไม่ได้"))
            continue
        mode, pid = m.group(1).upper(), m.group(2)
        if mode != filter_mode.upper():
            continue                                    # ใช้ฟิลเตอร์โหมดเดียว กันสำเนาซ้ำ 3 ชุด
        lab = parse_kauh_sound(parts[1])
        if lab is None:
            skipped.append((wav.name, f"ชนิดเสียง '{parts[1].strip()}' ไม่อยู่ใน N/C/W"))
            continue
        rows.append(dict(uid=wav.stem, dataset="kauh", file=str(wav), start=0.0, end=-1.0,
                         label=lab, patient=f"P{pid}", group=f"P{pid}", split="", device="Littmann3200"))
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    return df


# ----------------------------------------------------------------------------
# HF_Lung_V1 (Hsu et al., 2021)
# แต่ละคลิปมี <ชื่อ>.wav และ <ชื่อ>_label.txt บรรทัดละเหตุการณ์ เช่น "D 00:00:01.234 00:00:02.000"
#   D = DAS (crackle), Wheeze/Stridor/Rhonchi = CAS, I/E = ช่วงหายใจเข้า/ออก (ไม่ใช้)
# ----------------------------------------------------------------------------
HF_CAS = {"wheeze", "stridor", "rhonchi", "cas", "w", "s", "r"}
HF_DAS = {"d", "das", "crackle", "crackles"}


def hf_group(stem: str) -> str:
    """จัดกลุ่มคลิปที่น่าจะมาจากการบันทึกเดียวกัน (อุปกรณ์ + วันที่ + ชั่วโมง)

    HF_Lung_V1 ไม่มีรหัสผู้ป่วย ชื่อไฟล์เป็น เช่น steth_20190621_09_37_59 จึงจัดกลุ่มแบบหยาบ
    (กลุ่มหยาบกว่าจริงกันการรั่วได้มากกว่า) และต้องเขียนเป็นข้อจำกัดในรายงาน
    """
    m = re.match(r"^([A-Za-z]+)_(\d{8})_(\d{2})", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else stem


def build_hf_lung(root: Path) -> pd.DataFrame:
    wavs = sorted(Path(root).rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"ไม่พบไฟล์ .wav ใน {root} (ดู datasets/README.md)")
    rows = []
    for wav in wavs:
        lab_file = wav.with_name(wav.stem + "_label.txt")
        cas = das = False
        if lab_file.exists():
            for line in lab_file.read_text(errors="ignore").splitlines():
                tok = line.strip().split()
                if not tok:
                    continue
                t = tok[0].lower()
                cas |= t in HF_CAS
                das |= t in HF_DAS
        official = "test" if "test" in [p.lower() for p in wav.parts] else "train"
        rows.append(dict(uid=wav.stem, dataset="hf_lung", file=str(wav), start=0.0, end=-1.0,
                         label=_lab(das, cas), patient="", group=hf_group(wav.stem),
                         split=official, device=wav.stem.split("_")[0]))
    return pd.DataFrame(rows)


def build_metadata(name: str, root=None, kauh_filter: str = "D") -> pd.DataFrame:
    root = Path(root) if root else DATASET_DEFAULTS[name]["root"]
    if name == "icbhi":
        return build_icbhi(root)
    if name == "kauh":
        return build_kauh(root, kauh_filter)
    if name == "hf_lung":
        return build_hf_lung(root)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Waveform cache: อ่าน resample band-pass ตัดรอบหายใจ ปรับความยาว แล้วเก็บเป็น .npy
# ----------------------------------------------------------------------------
def cache_name(name, sr, length, bp, kauh_filter, root=None):
    bp_tag = f"bp{int(bp[0])}-{int(bp[1])}" if bp else "nobp"
    extra = f"_{kauh_filter}" if name == "kauh" else ""
    if root and Path(root).resolve() != Path(DATASET_DEFAULTS[name]["root"]).resolve():
        import hashlib
        extra += "_" + hashlib.md5(str(Path(root).resolve()).encode()).hexdigest()[:6]
    return f"{name}{extra}_sr{sr}_len{length:g}_{bp_tag}"


def load_dataset(name: str, sr: int = 16000, desired_length=None, band=(120, 1800),
                 root=None, kauh_filter: str = "D", pad_mode: str = "repeat", rebuild=False, verbose=True):
    """คืน (meta DataFrame, waveforms float16 array [N, L]) พร้อม cache ไว้ใช้รอบถัดไป"""
    desired_length = desired_length or DATASET_DEFAULTS[name]["desired_length"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = cache_name(name, sr, desired_length, band, kauh_filter, root)
    f_meta, f_wav = CACHE_DIR / f"{tag}.csv", CACHE_DIR / f"{tag}.npy"
    if f_meta.exists() and f_wav.exists() and not rebuild:
        meta = pd.read_csv(f_meta, dtype={"patient": str, "group": str, "split": str}, keep_default_na=False)
        return meta, np.load(f_wav, mmap_mode="r")

    meta = build_metadata(name, root, kauh_filter)
    n = int(round(desired_length * sr))
    wav = np.zeros((len(meta), n), dtype=np.float16)
    cur_file, data = None, None
    for i, r in enumerate(meta.itertuples()):
        if r.file != cur_file:
            data = load_wav(r.file, sr)
            if band:
                data = bandpass(data, sr, band[0], band[1])   # กรองทั้งไฟล์ก่อนตัด ลดผลขอบ
            peak = np.abs(data).max() + 1e-8
            data = data / peak
            cur_file = r.file
        seg = data if r.end < 0 else data[int(r.start * sr):int(r.end * sr)]
        wav[i] = fit_length(seg, n, sr, pad_mode).astype(np.float16)
        if verbose and i % 500 == 0:
            print(f"  [{name}] {i}/{len(meta)}")
    meta.to_csv(f_meta, index=False)
    np.save(f_wav, wav)
    return meta, wav


# ----------------------------------------------------------------------------
# การแบ่งข้อมูล (แยกผู้ป่วยเสมอ)  split คงที่ ไม่ขึ้นกับ seed ของการรัน
# ----------------------------------------------------------------------------
SPLIT_SEED = 0


def _group_val(meta_tr: pd.DataFrame, val_ratio: float):
    """แบ่งผู้ป่วยราว val_ratio ของชุด train เป็น validation (stratified ตาม label)"""
    k = max(2, int(round(1 / val_ratio)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SPLIT_SEED)
    tr, va = next(sgkf.split(meta_tr, meta_tr["label"], meta_tr["group"]))
    return meta_tr.index[tr].to_numpy(), meta_tr.index[va].to_numpy()


def kfold_assignments(meta: pd.DataFrame, name: str, n_splits: int = 5) -> np.ndarray:
    """กำหนด fold ครั้งเดียวแล้วบันทึกไว้ใน splits/ เพื่อให้ทุกวิธีใช้ fold ชุดเดียวกัน"""
    f = SPLITS_DIR / f"{name}_group{n_splits}fold.csv"
    if f.exists():
        saved = pd.read_csv(f, keep_default_na=False)
        mp = dict(zip(saved["uid"], saved["fold"]))
        if set(mp) == set(meta["uid"]):
            return meta["uid"].map(mp).to_numpy()
        print(f"  [warn] {f.name} ไม่ตรงกับข้อมูลปัจจุบัน สร้างใหม่")
    fold = np.zeros(len(meta), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED)
    for k, (_, te) in enumerate(sgkf.split(meta, meta["label"], meta["group"])):
        fold[te] = k
    SPLITS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"uid": meta["uid"], "group": meta["group"], "fold": fold}).to_csv(f, index=False)
    return fold


def get_split(meta: pd.DataFrame, name: str, fold="official", val_ratio: float = 0.2):
    """คืน index ของ train / val / test

    fold="official": ICBHI official split (และ HF_Lung ใช้ train/test ทางการได้)
    fold=0..4     : group 5-fold (KAUH, HF_Lung)
    """
    meta = meta.reset_index(drop=True)
    if str(fold) == "official":
        if not set(meta["split"]) >= {"train", "test"}:
            raise ValueError(f"{name} ไม่มี official split ให้ใช้ --fold 0..4")
        te = meta.index[meta["split"] == "test"].to_numpy()
        trv = meta[meta["split"] == "train"]
    else:
        folds = kfold_assignments(meta, name)
        te = np.where(folds == int(fold))[0]
        trv = meta[folds != int(fold)]
    tr, va = _group_val(trv, val_ratio)
    # ตรวจซ้ำว่าไม่มีกลุ่ม (ผู้ป่วย) รั่วข้ามชุด
    g = meta["group"].to_numpy()
    assert not (set(g[tr]) & set(g[va])), "พบผู้ป่วยซ้ำระหว่าง train และ validation"
    overlap = (set(g[tr]) | set(g[va])) & set(g[te])
    if overlap:
        if name == "icbhi" and str(fold) == "official" and overlap <= ICBHI_OFFICIAL_OVERLAP:
            # official split ของ ICBHI แบ่งระดับไฟล์ มีผู้ป่วย 2 คนอยู่ทั้งสองฝั่งมาแต่ต้น
            # คงไว้ตามต้นฉบับเพื่อให้เทียบกับงานอื่นได้ และเขียนเป็นข้อจำกัดในรายงาน
            pass
        elif name == "hf_lung" and str(fold) == "official":
            print(f"  [warn] train/test ทางการของ HF_Lung มีกลุ่มการบันทึกซ้ำกัน {len(overlap)} กลุ่ม "
                  "(ใช้ --fold 0..4 ตาม RD เพื่อแยกกลุ่ม)")
        else:
            raise AssertionError(f"พบผู้ป่วยซ้ำข้าม train/test: {sorted(overlap)[:10]}")
    return tr, va, te


# ผู้ป่วยที่อยู่ทั้ง train และ test ใน official_split.txt (ตรวจจากไฟล์ 2026-10-18)
ICBHI_OFFICIAL_OVERLAP = {"156", "218"}
