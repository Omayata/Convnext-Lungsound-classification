"""รันหลายการทดลองต่อกัน และข้ามการรันที่มีผลใน results.csv แล้ว (สั่งต่อได้หลังเครื่องดับ)

แผนที่มี:
    poc   : ICBHI official split, 6 วิธีของ PoC (RD 3.2: วิธี 1–4 และ 7 + โมเดลที่เสนอ), seed 0–2
    main  : ICBHI official split, ทุกวิธี (10 วิธี), seed 0–4
    cross : KAUH และ HF_Lung_V1, ทุกวิธี, group 5-fold × seed 0–4  (RQ3)
    demo  : ตัวอย่างสั้นสำหรับนำเสนอ: ICBHI, ทุกวิธี (คู่เทียบ 6 + ablation 11), seed 0, 5 epoch
            ข้อมูลจริง พารามิเตอร์จริง ต่างแค่จำนวน epoch (ผลแยกไว้ใน results/demo/)
    tune  : ปรับ lr บน validation ให้ทุกวิธีด้วยงบเท่ากัน (lr × 0.1, 1, 10; seed 0; ไม่เปิด test)

ตัวอย่าง:
    python -m src.run_experiments --plan demo                    # demo ทุกวิธี
    python -m src.run_experiments --plan demo --methods proposed abl_mel_only   # เลือกบางวิธี
    python -m src.run_experiments --plan poc --audioset_ckpt checkpoints/convnext_tiny_471mAP.pth
    python -m src.run_experiments --plan main --methods proposed abl_mel_only --seeds 0 1 2 3 4
    python -m src.run_experiments --plan main --use_tuned          # ใช้ lr ที่ดีที่สุดจาก tune
"""
import argparse
import gc
import itertools
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

from .methods import METHODS, POC_METHODS, FULL_METHODS, ABLATIONS, COMPETITORS
from .train import parse_args as train_args, run
from .utils import RESULTS_DIR

PLANS = {
    "poc": dict(datasets=["icbhi"], folds={"icbhi": ["official"]}, methods=POC_METHODS, seeds=[0, 1, 2]),
    "main": dict(datasets=["icbhi"], folds={"icbhi": ["official"]}, methods=FULL_METHODS, seeds=[0, 1, 2, 3, 4]),
    "cross": dict(datasets=["kauh", "hf_lung"], folds={"kauh": list(range(5)), "hf_lung": list(range(5))},
                  methods=FULL_METHODS, seeds=[0, 1, 2, 3, 4]),
    "tune": dict(datasets=["icbhi"], folds={"icbhi": ["official"]}, methods=FULL_METHODS, seeds=[0]),
    # demo: ไม่ใช่ผลสำหรับรายงาน ฝึกแค่ 5 epoch เพื่อแสดงว่า pipeline ทำงานจริงบนข้อมูลจริง
    "demo": dict(datasets=["icbhi"], folds={"icbhi": ["official"]}, methods=["proposed"] + COMPETITORS + ABLATIONS,
                 seeds=[0],
                 train_args=["--epochs", "5", "--patience", "5", "--warmup_epochs", "1"]),
}
DEMO_DIR = RESULTS_DIR / "demo"
TUNE_MULT = [0.1, 1.0, 10.0]
TUNE_CSV = RESULTS_DIR / "tune.csv"


def done_runs(csv):
    if not Path(csv).exists():
        return set()
    df = pd.read_csv(csv, dtype={"fold": str})
    return set(zip(df.dataset, df.fold.astype(str), df.method, df.seed, df.lr.round(12)))


def tuned_lr():
    df = pd.read_csv(TUNE_CSV, dtype={"fold": str})
    best = df.sort_values("val_icbhi_score", ascending=False).groupby("method").head(1)
    return dict(zip(best.method, best.lr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="poc", choices=list(PLANS))
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--methods", nargs="*", choices=list(METHODS))
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--folds", nargs="*", help="ทับ fold ของทุก dataset")
    p.add_argument("--use_tuned", action="store_true", help="ใช้ lr ที่ดีที่สุดจาก results/tune.csv")
    p.add_argument("--dry_run", action="store_true", help="แสดงรายการรันโดยไม่รันจริง")
    args, extra = p.parse_known_args()        # argument อื่นส่งต่อให้ train.py เช่น --audioset_ckpt

    plan = PLANS[args.plan]
    datasets = args.datasets or plan["datasets"]
    methods = args.methods or plan["methods"]
    seeds = args.seeds if args.seeds is not None else plan["seeds"]
    lrs = tuned_lr() if args.use_tuned else {}
    csv = {"tune": TUNE_CSV, "demo": DEMO_DIR / "results.csv"}.get(args.plan, RESULTS_DIR / "results.csv")
    plan_args = plan.get("train_args", []) + (["--run_root", str(DEMO_DIR / "runs")] if args.plan == "demo" else [])

    jobs = []
    for ds in datasets:
        folds = args.folds or plan["folds"].get(ds, ["official"])
        for m, f, s in itertools.product(methods, folds, seeds):
            base_lr = METHODS[m]["optim"]["lr"]
            mults = TUNE_MULT if args.plan == "tune" else [None]
            for mult in mults:
                lr = base_lr * mult if mult else lrs.get(m, base_lr)
                jobs.append((ds, str(f), m, s, lr))

    done = done_runs(csv)
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3], round(j[4], 12)) not in done]
    print(f"แผน {args.plan}: ทั้งหมด {len(jobs)} รัน, ทำแล้ว {len(jobs) - len(todo)}, เหลือ {len(todo)}")
    for j in todo:
        print("  ", j)
    if args.dry_run:
        return

    failed = []
    for i, (ds, f, m, s, lr) in enumerate(todo, 1):
        print(f"\n######## [{i}/{len(todo)}] {ds} fold={f} {m} seed={s} lr={lr:g} ########")
        argv = ["--dataset", ds, "--fold", f, "--method", m, "--seed", str(s), "--lr", repr(lr),
                "--results_csv", str(csv)] + (["--no_test"] if args.plan == "tune" else []) + plan_args + extra
        try:
            run(train_args(argv))
        except Exception:
            traceback.print_exc()
            failed.append((ds, f, m, s))
        gc.collect()                          # คืน VRAM ก่อนรันถัดไป (ทุกรันอยู่ใน process เดียวกัน)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if failed:
        print("\nรันที่ล้มเหลว:", *failed, sep="\n  ")
        sys.exit(1)
    if args.plan == "demo":
        print("\nเสร็จแล้ว สร้างรายงานด้วย: python scripts/demo_report.py")


if __name__ == "__main__":
    main()
