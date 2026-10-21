"""สร้างรายงาน demo (ไฟล์ HTML ไฟล์เดียว เปิดใน browser ได้เลย) จากผลของ --plan demo

    python -m src.run_experiments --plan demo
    python scripts/demo_report.py              # -> results/demo/demo_report.html

ภาพที่ได้ (แต่ละวิธีมีสีของตัวเอง ใช้สีเดียวกันทุกกราฟ):
    1. ICBHI score และ macro-F1: โมเดลที่เสนอ เทียบคู่เทียบ 6 งาน
    2. Ablation: ตัดเทคนิคออกทีละอย่าง แล้วคะแนนเปลี่ยนจากโมเดลที่เสนอเท่าไร
    3. F1 รายคลาส
    4. Confusion matrix ของโมเดลที่เสนอ เทียบกับ Mel อย่างเดียว
    5. ICBHI score บน validation ในแต่ละ epoch
"""
import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "results" / "demo"

# ชื่อในกราฟ (ภาษาอังกฤษ เพราะ matplotlib บน Windows มักไม่มีฟอนต์ไทย) และสีประจำวิธี
LABEL = {
    "proposed": "Proposed: ConvNeXt-AS + Mel+MFCC+Δ",
    "wanasinghe_cnn": "[1] Wanasinghe: CNN + Mel/MFCC/Chroma",
    "dongre_cnn_bigru": "[2] Dongre: CNN-BiGRU + SMOTE",
    "shakya_vgg16": "[3] Shakya: VGG16 + Mel",
    "puneet_mobilenetv2": "[4] Puneet: MobileNetV2 + STFT",
    "park_inception": "[5] Park: Inception CNN",
    "engin_densenet201": "[6] Engin: DenseNet201 + Gammatone",
}
COLOR = {
    "proposed": "#1f5fbf",
    "wanasinghe_cnn": "#e8743b",
    "dongre_cnn_bigru": "#19a979",
    "shakya_vgg16": "#ed4a7b",
    "puneet_mobilenetv2": "#945ecf",
    "park_inception": "#c9a227",
    "engin_densenet201": "#13a4b4",
    "abl_mel_only": "#7f7f7f",
}
# ablation: (ชื่อในกราฟ, คำอธิบายไทยในตาราง) ตัด/เปลี่ยนทีละเทคนิคจากโมเดลที่เสนอ
ABL = {
    "abl_mel_only": ("− MFCC & ΔMFCC (Mel only)", "ตัดทางเสริม MFCC + ΔMFCC เหลือ Mel อย่างเดียว"),
    "abl_mel_mfcc": ("− ΔMFCC (Mel + MFCC)", "ตัด ΔMFCC ออก"),
    "abl_imagenet": ("AudioSet → ImageNet weights", "ใช้น้ำหนัก ImageNet แทน AudioSet"),
    "abl_no_pretrain": ("− pretraining (random init)", "ไม่ใช้น้ำหนัก pretrain เลย"),
    "abl_concat": ("− SE gating (plain concat)", "รวม embedding ด้วยการต่อกันตรงๆ แทน SE gating"),
    "abl_no_bandpass": ("− band-pass 120–1800 Hz", "ไม่กรองความถี่"),
    "abl_pcen": ("log-Mel → PCEN", "ใช้ PCEN แทน log-Mel"),
    "abl_no_specaug": ("− SpecAugment", "ไม่ใช้ SpecAugment"),
    "abl_no_mixup": ("− mixup", "ไม่ใช้ mixup"),
    "abl_no_classweight": ("− class-weighted loss", "ไม่ถ่วงน้ำหนักคลาสใน loss"),
    "abl_no_droppath": ("− drop path", "ไม่ใช้ drop path"),
}
CLASSES = ["normal", "crackle", "wheeze", "both"]


def name(m):
    return LABEL.get(m) or (ABL[m][0] if m in ABL else m)


def color(m):
    return COLOR.get(m, "#999999")


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def bar_main(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.5 * len(df) + 1.6))
    fig.subplots_adjust(wspace=0.75)
    for ax, metric, title in zip(axes, ["icbhi_score", "macro_f1"], ["ICBHI score (%)", "Macro-F1 (%)"]):
        d = df.sort_values(metric)
        y = np.arange(len(d))
        ax.barh(y, d[metric], color=[color(m) for m in d["method"]])
        for yi, v in zip(y, d[metric]):
            ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=9)
        ax.set_yticks(y, [name(m) for m in d["method"]], fontsize=8.5)
        for t, m in zip(ax.get_yticklabels(), d["method"]):
            t.set_color(color(m))
            t.set_fontweight("bold" if m == "proposed" else "normal")
        ax.set_title(title)
        ax.set_xlim(0, max(100, d[metric].max() + 8))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def bar_ablation(df, ref):
    """คะแนนของ ablation ลบด้วยโมเดลที่เสนอ: ติดลบ (แดง) = ตัดเทคนิคนั้นแล้วแย่ลง แปลว่าเทคนิคนั้นช่วย"""
    d = df[df["method"].isin(ABL)]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(d) + 1.8))
    fig.subplots_adjust(wspace=0.6)
    for ax, metric, title in zip(axes, ["icbhi_score", "macro_f1"], ["Δ ICBHI score vs Proposed (pp)", "Δ Macro-F1 vs Proposed (pp)"]):
        delta = (d[metric] - ref[metric]).to_numpy()
        order = np.argsort(delta)[::-1]
        y = np.arange(len(d))
        vals = delta[order]
        ax.barh(y, vals, color=["#d64545" if v < 0 else "#3a9a5b" for v in vals])
        span = max(1.0, np.abs(vals).max())
        for yi, v in zip(y, vals):
            ax.text(v + (0.03 if v >= 0 else -0.03) * span, yi, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ax.set_yticks(y, [ABL[m][0] for m in d["method"].to_numpy()[order]], fontsize=8.5)
        ax.axvline(0, color="#333", lw=1)
        ax.set_xlim(-span * 1.3, span * 1.3)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def bar_per_class(df):
    d = df.set_index("method")
    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = np.arange(len(CLASSES))
    w = 0.8 / len(d)
    for i, (m, r) in enumerate(d.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [r[f"f1_{c}"] for c in CLASSES], w,
               label=name(m), color=color(m), edgecolor="white")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Per-class F1 on ICBHI test")
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


def confusions(run_root, methods):
    shown = [m for m in methods if (run_root / m / "foldofficial_seed0" / "confusion.csv").exists()]
    if not shown:
        return None
    fig, axes = plt.subplots(1, len(shown), figsize=(4.6 * len(shown), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, shown):
        cm = pd.read_csv(run_root / m / "foldofficial_seed0" / "confusion.csv", index_col=0).to_numpy()
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{100*norm[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASSES, fontsize=8)
        ax.set_yticks(range(4), CLASSES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name(m), fontsize=9, color=color(m))
    return fig_to_b64(fig)


def curves(run_root, df):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for _, r in df.iterrows():
        m = r["method"]
        f = run_root / m / "foldofficial_seed0" / "history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        ax.plot(h["epoch"], h["val_icbhi_score"], marker="o", lw=2.8 if m == "proposed" else 1.4,
                color=color(m), label=name(m), zorder=3 if m == "proposed" else 2)
    ax.set_xlabel("epoch")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("val ICBHI score (%)")
    ax.set_title("Validation ICBHI score per epoch")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEMO / "results.csv"))
    p.add_argument("--run_root", default=str(DEMO / "runs" / "icbhi"))
    p.add_argument("--out", default=str(DEMO / "demo_report.html"))
    p.add_argument("--synthetic", action="store_true", help="ผลมาจากข้อมูลสังเคราะห์ (make_dummy_data.py) ติดป้ายเตือนในรายงาน")
    a = p.parse_args()
    if not Path(a.csv).exists():
        sys.exit(f"ไม่พบ {a.csv} ให้รัน python -m src.run_experiments --plan demo ก่อน")
    df = pd.read_csv(a.csv, dtype={"fold": str})
    df = df[df["dataset"] == "icbhi"].dropna(subset=["icbhi_score"])
    df = df.sort_values("time").groupby("method").tail(1)            # ใช้รันล่าสุดของแต่ละวิธี
    run_root = Path(a.run_root)
    main_df = df[df["method"].isin(LABEL)]                             # โมเดลที่เสนอ + คู่เทียบ
    ref = df[df["method"] == "proposed"]
    ref = ref.iloc[0] if len(ref) else None

    cols = ["method", "icbhi_score", "sensitivity", "specificity", "macro_f1",
            "f1_crackle", "f1_wheeze", "f1_both", "params_M", "epochs_run", "train_time_min"]
    heads = ["วิธี", "ICBHI score", "Se", "Sp", "Macro-F1", "F1 crackle", "F1 wheeze", "F1 both",
             "พารามิเตอร์ (M)", "epoch", "เวลาฝึก (นาที)"]
    t1 = main_df[cols].sort_values("icbhi_score", ascending=False).copy()
    t1["method"] = [f'<span style="color:{color(m)};font-weight:bold">■</span> {name(m)}' for m in t1["method"]]
    t1.columns = heads

    abl = df[df["method"].isin(ABL)].copy()
    t2 = None
    if ref is not None and len(abl):
        abl["d_icbhi"] = abl["icbhi_score"] - ref["icbhi_score"]
        abl["d_f1"] = abl["macro_f1"] - ref["macro_f1"]
        abl = abl.sort_values("d_icbhi")
        t2 = pd.DataFrame({"สิ่งที่เปลี่ยนจากโมเดลที่เสนอ": [ABL[m][1] for m in abl["method"]],
                           "--method": abl["method"], "ICBHI score": abl["icbhi_score"], "Δ ICBHI": abl["d_icbhi"],
                           "Macro-F1": abl["macro_f1"], "Δ Macro-F1": abl["d_f1"],
                           "สรุป (demo)": ["เทคนิคนี้ช่วย" if v < 0 else "ยังไม่เห็นว่าช่วย" for v in abl["d_icbhi"]]})

    imgs = {"main": bar_main(main_df) if len(main_df) else None,
            "abl": bar_ablation(df, ref) if ref is not None else None,
            "cls": bar_per_class(main_df) if len(main_df) else None,
            "cm": confusions(run_root, ["proposed", "abl_mel_only"]), "curve": curves(run_root, main_df)}
    n_ep = int(df["epochs_run"].max())
    img = lambda k: f'<img src="data:image/png;base64,{imgs[k]}">' if imgs[k] else "<p>(ไม่มีข้อมูล)</p>"
    tbl = lambda t: t.round(2).to_html(index=False, escape=False) if t is not None else "<p>(ไม่มีข้อมูล)</p>"
    if a.synthetic:
        note = (f'<p class="note" style="background:#ffe1e1;border-color:#d64545"><b>ข้อมูลสังเคราะห์ ตัวเลขทุกตัวในรายงานนี้ไม่มีความหมาย:</b> '
                f'เสียงสร้างขึ้นด้วย scripts/make_dummy_data.py (ไม่ใช่เสียงปอดจริง) ใช้ {len(df)} วิธี ฝึก {n_ep} epoch บน CPU '
                f'น้ำหนัก pretrain เป็นแบบทดสอบ รายงานนี้แสดงแค่ว่า pipeline ทั้งเส้นทำงานได้ และหน้าตาของผลที่จะได้เมื่อรันกับ ICBHI จริง</p>')
    else:
        note = (f'<p class="note"><b>นี่คือ demo ไม่ใช่ผลสุดท้าย:</b> ใช้ข้อมูลจริงและ hyperparameter จริงของทุกวิธี ต่างแค่ฝึก {n_ep} epoch และ seed เดียว '
                f'จึงยังไม่ทดสอบสถิติ ผลจริงจะใช้สูงสุด 50 epoch + early stopping และ 5 seeds ตาม RD_v2</p>')
    html = f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Lung Sound Demo</title>
<style>
body{{font-family:"Leelawadee UI","Tahoma",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}} img{{max-width:100%}}
.note{{background:#fff4d6;border-left:4px solid #e0a000;padding:10px 14px}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;font-size:13px;width:100%}} th,td{{border:1px solid #ddd;padding:5px 7px;text-align:right}}
th{{background:#f2f4f8}} td:first-child{{text-align:left}}
</style></head><body>
<h1>{"[ข้อมูลสังเคราะห์] " if a.synthetic else ""}Demo: Mel–MFCC Fusion + AudioSet-Pretrained ConvNeXt บน ICBHI 2017</h1>
{note}
<p><b>Setting เดียวกันทุกวิธี:</b> ICBHI 4 คลาสระดับรอบหายใจ, official split (train 4,142 / test 2,756 รอบ),
validation แยกผู้ป่วย 20% ของ train, resample 16 kHz, band-pass 120–1800 Hz, augmentation เฉพาะ train,
loss ถ่วงน้ำหนักคลาส, เลือก epoch จาก ICBHI score บน validation แล้วเปิด test ครั้งเดียว</p>
<h2>1. โมเดลที่เสนอ เทียบคู่เทียบ 6 งาน (ผลบน test set)</h2>{img("main")}
<div class="wrap">{tbl(t1)}</div>
<h2>2. Ablation: ตัดเทคนิคออกทีละอย่าง</h2>
<p>แต่ละแถวคือโมเดลที่เสนอที่เปลี่ยนไปหนึ่งอย่าง ค่าที่แสดงคือคะแนนลบด้วยโมเดลที่เสนอ
<b style="color:#d64545">แดง (ติดลบ)</b> = ตัดออกแล้วแย่ลง แปลว่าเทคนิคนั้นช่วย,
<b style="color:#3a9a5b">เขียว (บวก)</b> = ตัดออกแล้วไม่แย่ลง ยังไม่เห็นประโยชน์จาก demo นี้</p>
{img("abl")}
<div class="wrap">{tbl(t2)}</div>
<h2>3. F1 รายคลาส (RQ2: feature ช่วย crackle / wheeze ต่างกันไหม)</h2>{img("cls")}
<h2>4. Confusion matrix: โมเดลที่เสนอ เทียบกับ Mel อย่างเดียว</h2>{img("cm")}
<h2>5. ICBHI score บน validation ระหว่างฝึก</h2>{img("curve")}
<p style="color:#777;font-size:12px">อ้างอิงบน official split เดียวกัน (ตัวเลขจากเปเปอร์ ไม่ได้รันเอง): BTS (Interspeech 2024) 63.54%</p>
</body></html>"""
    if a.synthetic:      # ข้อมูลสังเคราะห์มีแค่ไม่กี่สิบรอบ ไม่ใช่ official split จริง
        html = html.replace("<b>Setting เดียวกันทุกวิธี:</b>", "<b>Setting ที่จะใช้กับข้อมูลจริง (รอบนี้ใช้ข้อมูลสังเคราะห์ train/val/test อย่างละ 40 รอบ):</b>")
    Path(a.out).write_text(html, encoding="utf-8")
    print("สร้างรายงานแล้ว:", a.out)


if __name__ == "__main__":
    main()
