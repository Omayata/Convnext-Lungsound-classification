"""สรุปผลและทดสอบสถิติ (RD_v2 ส่วนที่ 6–7)

    1. ตารางสรุป mean ± SD และ 95% CI ต่อ dataset × method (หน่วยคือ seed;
       dataset แบบ k-fold เฉลี่ยทุก fold ใน seed เดียวกันก่อน)
    2. เทียบโมเดลที่เสนอกับทุกวิธีแบบจับคู่ตาม seed:
       Shapiro-Wilk ของผลต่าง -> paired t-test (หลัก) และ Wilcoxon (ใช้แทนเมื่อไม่ปกติ, ต้องมี >= 6 seeds)
       ปรับค่า p ด้วย Holm (one-vs-control) ทดสอบทางเดียว H1: proposed > วิธีอื่น
    3. ข้ามหลาย dataset: Friedman + Nemenyi critical difference (Demšar, 2006)

ใช้:
    python -m src.stats                       # อ่าน results/results.csv
    python -m src.stats --metric macro_f1
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats as st

from .utils import RESULTS_DIR

# ค่า q_alpha ของ Nemenyi ที่ alpha = 0.05 (Demšar 2006, Table 5) สำหรับ k = 2..10 วิธี
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def per_seed(df, metric):
    """ค่า metric ต่อ (dataset, method, seed) โดยเฉลี่ยข้าม fold"""
    return df.groupby(["dataset", "method", "seed"])[metric].mean().reset_index()


def summary(df, metric):
    rows = []
    for (ds, m), g in per_seed(df, metric).groupby(["dataset", "method"]):
        x = g[metric].to_numpy()
        n = len(x)
        sd = x.std(ddof=1) if n > 1 else np.nan
        half = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append({"dataset": ds, "method": m, "n_seeds": n, "mean": x.mean(), "sd": sd,
                     "ci95_low": x.mean() - half, "ci95_high": x.mean() + half})
    return pd.DataFrame(rows).sort_values(["dataset", "mean"], ascending=[True, False])


def compare_to_control(df, metric, control="proposed"):
    rows = []
    ps = per_seed(df, metric)
    for ds, g in ps.groupby("dataset"):
        wide = g.pivot(index="seed", columns="method", values=metric)
        if control not in wide:
            continue
        sub = []
        for m in wide.columns:
            if m == control:
                continue
            pair = wide[[control, m]].dropna()
            n = len(pair)
            if n < 2:
                continue
            d = (pair[control] - pair[m]).to_numpy()
            sw_p = st.shapiro(d).pvalue if n >= 3 and np.ptp(d) > 0 else np.nan
            t_p = st.ttest_rel(pair[control], pair[m], alternative="greater").pvalue if np.ptp(d) > 0 else np.nan
            w_p = st.wilcoxon(d, alternative="greater").pvalue if n >= 6 and np.any(d != 0) else np.nan
            sub.append({"dataset": ds, "control": control, "method": m, "n_pairs": n,
                        "mean_diff": d.mean(), "shapiro_p": sw_p, "ttest_p": t_p, "wilcoxon_p": w_p})
        if sub:
            s = pd.DataFrame(sub)
            s["ttest_p_holm"] = holm(s["ttest_p"].fillna(1.0))
            if s["wilcoxon_p"].notna().any():
                s["wilcoxon_p_holm"] = holm(s["wilcoxon_p"].fillna(1.0))
            s["significant_0.05"] = s["ttest_p_holm"] < 0.05
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def friedman_nemenyi(df, metric):
    """ใช้ค่าเฉลี่ยต่อ dataset ของแต่ละวิธี (ต้องมีอย่างน้อย 2 dataset และทุกวิธีครบทุก dataset)"""
    mean = per_seed(df, metric).groupby(["dataset", "method"])[metric].mean().unstack()
    mean = mean.dropna(axis=1)
    n, k = mean.shape
    if n < 2 or k < 3:
        return None
    ranks = mean.rank(axis=1, ascending=False)
    stat, p = st.friedmanchisquare(*[mean[c] for c in mean.columns])
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n))
    return {"n_datasets": n, "k_methods": k, "friedman_chi2": stat, "friedman_p": p, "nemenyi_cd": cd,
            "avg_rank": ranks.mean().sort_values().to_dict()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(RESULTS_DIR / "results.csv"))
    p.add_argument("--metric", default="icbhi_score")
    p.add_argument("--control", default="proposed")
    p.add_argument("--include_limit", action="store_true", help="รวมการรัน debug ที่ใช้ --limit")
    a = p.parse_args()

    df = pd.read_csv(a.csv, dtype={"fold": str})
    if "limit" in df and not a.include_limit:
        df = df[df["limit"].fillna(0) == 0]
    df = df.dropna(subset=[a.metric])
    if df.empty:
        print("ยังไม่มีผลที่ใช้ได้")
        return
    pd.set_option("display.width", 200)
    out = RESULTS_DIR

    s = summary(df, a.metric)
    s.to_csv(out / f"summary_{a.metric}.csv", index=False)
    print(f"\n=== สรุป {a.metric} (mean ± SD, 95% CI) ===")
    print(s.round(2).to_string(index=False))

    c = compare_to_control(df, a.metric, a.control)
    if not c.empty:
        c.to_csv(out / f"tests_{a.metric}.csv", index=False)
        print(f"\n=== {a.control} เทียบกับแต่ละวิธี (paired ตาม seed, ทางเดียว, Holm) ===")
        print(c.round(4).to_string(index=False))
        if (c["n_pairs"] < 6).any():
            print("หมายเหตุ: seed < 6 จึงไม่รายงาน Wilcoxon (ค่า p ต่ำสุดที่ n=5 ทางเดียวคือ 0.031, สองทาง 0.0625)")

    fr = friedman_nemenyi(df, a.metric)
    if fr:
        print("\n=== Friedman + Nemenyi ข้าม dataset ===")
        for k, v in fr.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
