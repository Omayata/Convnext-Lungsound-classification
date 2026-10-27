"""ฝึกและประเมิน 1 การรัน (dataset × fold × method × seed)

ตัวอย่าง:
    python -m src.train --dataset icbhi --method proposed --seed 0 \
        --audioset_ckpt checkpoints/convnext_tiny_471mAP.pth

ขั้นตอน (ตาม RD_v2):
    1. โหลดเสียง resample 16 kHz, band-pass 120–1800 Hz, ตัดตามรอบหายใจ, ปรับความยาว
    2. แบ่ง train / val / test แบบแยกผู้ป่วย (split คงที่ ไม่ขึ้นกับ seed)
    3. คำนวณ feature ของแต่ละวิธี, normalize ด้วยค่าจากชุด train เท่านั้น
    4. (เฉพาะ Dongre) SMOTE เฉพาะชุด train
    5. ฝึกด้วย loss ถ่วงน้ำหนักคลาส, SpecAugment + mixup เฉพาะชุด train,
       early stopping ตาม ICBHI score บน validation
    6. เปิด test set ครั้งเดียวด้วยโมเดลที่ดีที่สุดบน validation แล้วบันทึกผล
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import load_dataset, get_split, cache_name, DATASET_DEFAULTS
from .features import extract
from .methods import METHODS
from .metrics import all_metrics, confusion
from .models import ProposedNet
from .utils import set_seed, get_device, count_params, RESULTS_DIR, CLASS_NAMES, N_CLASSES, ROOT


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lung sound classification (RD_v2)")
    p.add_argument("--dataset", default="icbhi", choices=list(DATASET_DEFAULTS))
    p.add_argument("--data_root", default=None, help="โฟลเดอร์ข้อมูล (ค่าเริ่มต้น datasets/<ชื่อชุด>)")
    p.add_argument("--fold", default="official", help="official (ICBHI) หรือ 0..4 (group 5-fold)")
    p.add_argument("--method", default="proposed", choices=list(METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    # preprocessing (เหมือนกันทุกวิธี)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--desired_length", type=float, default=None, help="วินาที (ICBHI 8, อื่นๆ 15)")
    p.add_argument("--no_bandpass", action="store_true", help="ปิด band-pass 120–1800 Hz (ablation)")
    p.add_argument("--kauh_filter", default="D", choices=["B", "D", "E"], help="โหมดฟิลเตอร์ของ KAUH ที่ใช้")
    p.add_argument("--val_ratio", type=float, default=0.2)
    # โมเดลที่เสนอ
    ckpt = ROOT / "checkpoints" / "convnext_tiny_471mAP.pth"
    if not ckpt.exists() and (ROOT / "checkpoints" / "model.safetensors").exists():
        ckpt = ROOT / "checkpoints" / "model.safetensors"        # น้ำหนักชุดเดียวกันจาก Hugging Face
    p.add_argument("--audioset_ckpt", default=str(ckpt))
    p.add_argument("--mel_scale", default="log", choices=["log", "pcen"])
    p.add_argument("--fusion", default="se", choices=["se", "concat"])
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--no_pretrained", action="store_true", help="ไม่โหลดน้ำหนัก pretrain (ใช้ทดสอบเท่านั้น)")
    # การฝึก (เหมือนกันทุกวิธี)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=None, help="ทับค่า lr ของวิธีนั้น")
    p.add_argument("--batch_size", type=int, default=None, help="ทับค่า batch size ของวิธีนั้น")
    p.add_argument("--micro_batch", type=int, default=None,
                   help="แบ่ง batch เป็นชิ้นละ N ตัวอย่างแล้วสะสม gradient (effective batch เท่าเดิม) "
                        "ค่าเริ่มต้น: อัตโนมัติ 8 สำหรับโมเดล ConvNeXt เมื่อ GPU มี VRAM < 8 GB, 0 = ปิด")
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="0 = ปิด mixup")
    p.add_argument("--no_specaug", action="store_true")
    p.add_argument("--no_class_weight", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    # การบันทึก
    p.add_argument("--results_csv", default=str(RESULTS_DIR / "results.csv"))
    p.add_argument("--run_root", default=str(RESULTS_DIR / "runs"), help="โฟลเดอร์เก็บผลรายรัน")
    p.add_argument("--tag", default="", help="ต่อท้ายชื่อวิธีในผล เช่น _pcen")
    p.add_argument("--no_test", action="store_true", help="ไม่เปิด test set (ใช้ตอนปรับ hyperparameter)")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="ใช้ข้อมูลแค่ N ตัวอย่างต่อชุด (debug)")
    p.add_argument("--feature_batch", type=int, default=64)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# ข้อมูล
# ----------------------------------------------------------------------------
def build_features(args, waves, wave_tag, inputs, device):
    feats = {}
    for key, spec in inputs.items():
        if "stack" in spec:
            parts = [extract(waves, s, wave_tag, args.sr, device, args.feature_batch) for s in spec["stack"]]
            feats[key] = np.concatenate([np.asarray(p) for p in parts], axis=1)
        else:
            feats[key] = extract(waves, spec, wave_tag, args.sr, device, args.feature_batch)
    return feats


def channel_stats(arr, idx, chunk=512):
    """mean/std ต่อ channel จากชุด train เท่านั้น (กัน leakage ข้อ normalization)"""
    c = arr.shape[1]
    s, s2, n = np.zeros(c), np.zeros(c), 0
    for i in range(0, len(idx), chunk):
        x = np.asarray(arr[np.sort(idx[i:i + chunk])], dtype=np.float32)
        s += x.sum(axis=(0, 2, 3))
        s2 += (x.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def smote(x, y, k=5, rng=None):
    """SMOTE (Chawla et al., 2002) บน feature ที่ flatten แล้ว ทำเฉพาะชุด train จนทุกคลาสเท่าคลาสใหญ่สุด"""
    from sklearn.neighbors import NearestNeighbors

    rng = rng or np.random.default_rng(0)
    shape = x.shape[1:]
    counts = np.bincount(y, minlength=N_CLASSES)
    target = counts.max()
    new_x, new_y = [], []
    for c in range(N_CLASSES):
        need = target - counts[c]
        xc = x[y == c].reshape(counts[c], -1).astype(np.float32)
        if need <= 0 or counts[c] < 2:
            continue
        nn_ = NearestNeighbors(n_neighbors=min(k + 1, counts[c])).fit(xc)
        _, nb = nn_.kneighbors(xc)
        base = rng.integers(0, counts[c], need)
        pick = nb[base, rng.integers(1, nb.shape[1], need)]
        lam = rng.random((need, 1), dtype=np.float32)
        new_x.append((xc[base] + lam * (xc[pick] - xc[base])).reshape((need,) + shape).astype(np.float16))
        new_y.append(np.full(need, c))
    if not new_x:
        return x, y
    print(f"  SMOTE: เพิ่มตัวอย่างสังเคราะห์ {sum(len(a) for a in new_y)} ตัว (เฉพาะชุด train)")
    return np.concatenate([x] + new_x), np.concatenate([y] + new_y)


class Split:
    def __init__(self, feats, labels, uids):
        self.feats, self.labels, self.uids = feats, np.asarray(labels), list(uids)

    def __len__(self):
        return len(self.labels)

    def batches(self, bs, shuffle=False, rng=None):
        order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            if shuffle and len(b) < 2:
                continue
            yield {k: np.asarray(v[b]) for k, v in self.feats.items()}, self.labels[b]


def to_device(xb, stats, device):
    out = {}
    for k, v in xb.items():
        t = torch.from_numpy(v.astype(np.float32)).to(device, non_blocking=True)
        m, s = stats[k]
        out[k] = (t - m) / s
    return out


def spec_augment(x, rng, n_freq=2, n_time=2, f_ratio=0.15, t_ratio=0.1):
    """SpecAugment (Park et al., 2019) ปิดแถบความถี่/เวลาด้วยค่า 0 (= ค่าเฉลี่ยหลัง normalize)"""
    b, _, h, w = x.shape
    dev = x.device
    keep = torch.ones(b, 1, h, w, dtype=torch.bool, device=dev)
    fh = torch.arange(h, device=dev).view(1, h, 1)
    tw_ = torch.arange(w, device=dev).view(1, 1, w)
    for _ in range(n_freq):
        width = torch.from_numpy(rng.integers(0, max(1, int(h * f_ratio)) + 1, b)).to(dev).view(b, 1, 1)
        start = torch.from_numpy(rng.integers(0, h, b)).to(dev).view(b, 1, 1)
        keep &= ~((fh >= start) & (fh < start + width)).unsqueeze(1)
    for _ in range(n_time):
        width = torch.from_numpy(rng.integers(0, max(1, int(w * t_ratio)) + 1, b)).to(dev).view(b, 1, 1)
        start = torch.from_numpy(rng.integers(0, w, b)).to(dev).view(b, 1, 1)
        keep &= ~((tw_ >= start) & (tw_ < start + width)).unsqueeze(1)
    return x * keep


# ----------------------------------------------------------------------------
# การฝึก
# ----------------------------------------------------------------------------
def make_optimizer(model, cfg, args, steps_per_epoch):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr or cfg["lr"]
    if cfg["name"] == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.get("weight_decay", 0.05))
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=cfg.get("weight_decay", 0.0))
    sched, kind = None, cfg.get("scheduler")
    if kind == "cosine":
        total = args.epochs * steps_per_epoch
        warm = args.warmup_epochs * steps_per_epoch

        def f(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    elif kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    return opt, sched, kind, lr


# วัดบน RTX 3050 Ti 4 GB (AMP, input ICBHI 8 s): ConvNeXt batch 16 ใช้ ~4.4 GB, batch 8 ~2.4 GB
# คู่เทียบทุกตัวใช้ batch ต้นฉบับได้ไม่เกิน ~2.3 GB จึงแบ่งอัตโนมัติเฉพาะ ProposedNet (ไม่ไปเปลี่ยน BatchNorm ของคู่เทียบ)
LOW_VRAM_GB, LOW_VRAM_MICRO = 8, 8


def resolve_micro_batch(micro, bs, device, model):
    """ขนาด micro-batch ที่ใช้จริง (0 = ไม่แบ่ง) None = อัตโนมัติตาม VRAM ของ GPU"""
    if micro is None:
        micro = 0
        if device.type == "cuda" and isinstance(model, ProposedNet):
            vram = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
            micro = LOW_VRAM_MICRO if vram < LOW_VRAM_GB else 0
    return micro if 0 < micro < bs else 0


@torch.no_grad()
def predict(model, split, stats, device, bs, amp):
    model.eval()
    probs, gates = [], []
    t0 = time.time()
    for xb, _ in split.batches(bs):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(**to_device(xb, stats, device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        if hasattr(model, "branch_weights") and model.branch_weights():
            gates.append(model.branch_weights())
    ms = 1000 * (time.time() - t0) / max(1, len(split))
    gate = {k: float(np.mean([g[k] for g in gates])) for k in gates[0]} if gates else {}
    return np.concatenate(probs), ms, gate


def run(args):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    cfg = METHODS[args.method]
    for k, v in cfg.get("overrides", {}).items():      # ablation: ปิด/เปลี่ยนเทคนิคเดียว
        setattr(args, k, v)
    name = args.method + args.tag
    print(f"== {args.dataset} | fold {args.fold} | {name} | seed {args.seed} | {device} ==")
    print(f"   {cfg['desc']}")

    band = None if args.no_bandpass else (120, 1800)
    desired = args.desired_length or DATASET_DEFAULTS[args.dataset]["desired_length"]
    meta, waves = load_dataset(args.dataset, args.sr, desired, band, args.data_root, args.kauh_filter)
    wave_tag = cache_name(args.dataset, args.sr, desired, band, args.kauh_filter, args.data_root)
    tr, va, te = get_split(meta, args.dataset, args.fold, args.val_ratio)
    if args.limit:
        tr, va, te = (np.sort(rng.permutation(i)[:args.limit]) for i in (tr, va, te))
    y = meta["label"].to_numpy()
    print(f"   train {len(tr)} | val {len(va)} | test {len(te)} | "
          f"train class counts {np.bincount(y[tr], minlength=N_CLASSES).tolist()}")

    feats = build_features(args, waves, wave_tag, cfg["inputs"](args), device)
    stats = {}
    for k, arr in feats.items():
        m, s = channel_stats(arr, tr)
        stats[k] = (torch.tensor(m, device=device).view(1, -1, 1, 1), torch.tensor(s, device=device).view(1, -1, 1, 1))

    tr_feats = {k: np.asarray(v[np.sort(tr)]) for k, v in feats.items()}
    tr_sorted = np.sort(tr)
    tr_y = y[tr_sorted]
    if cfg.get("smote"):
        (key,) = tr_feats.keys()
        tr_feats[key], tr_y = smote(tr_feats[key], tr_y, rng=np.random.default_rng(args.seed))
    train = Split(tr_feats, tr_y, [f"syn{i}" for i in range(len(tr_y))])
    val = Split({k: v[np.sort(va)] for k, v in feats.items()}, y[np.sort(va)], meta["uid"].to_numpy()[np.sort(va)])
    test = Split({k: v[np.sort(te)] for k, v in feats.items()}, y[np.sort(te)], meta["uid"].to_numpy()[np.sort(te)])

    model = cfg["build"](args).to(device)
    n_params = count_params(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bs = args.batch_size or cfg["optim"]["batch_size"]
    steps = math.ceil(len(train) / bs)
    opt, sched, sched_kind, lr = make_optimizer(model, cfg["optim"], args, steps)
    mb = resolve_micro_batch(args.micro_batch, bs, device, model)
    eval_bs = (mb or bs) * 2
    print(f"   params {n_params/1e6:.2f}M (trainable {n_train/1e6:.2f}M) | lr {lr} | batch {bs}"
          + (f" (micro-batch {mb} สะสม gradient)" if mb else ""))

    counts = np.bincount(train.labels, minlength=N_CLASSES).astype(np.float64)
    w = None if args.no_class_weight else torch.tensor(counts.sum() / (N_CLASSES * np.maximum(counts, 1)),
                                                      dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, best_epoch, bad, hist = -1.0, None, -1, 0, []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in train.batches(bs, shuffle=True, rng=rng):
            x = to_device(xb, stats, device)
            yt = torch.from_numpy(yb).long().to(device)
            if not args.no_specaug:
                x = {k: spec_augment(v, rng) for k, v in x.items()}
            lam, perm = 1.0, None
            if args.mixup_alpha > 0:
                lam = float(rng.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(len(yt), device=device)
                x = {k: lam * v + (1 - lam) * v[perm] for k, v in x.items()}
            opt.zero_grad(set_to_none=True)
            # augmentation ทำกับทั้ง batch แล้วจึงแบ่ง micro-batch (ชิ้นขนาดใกล้กัน กัน BatchNorm เจอชิ้นละ 1 ตัว)
            n = len(yt)
            chunks = np.array_split(np.arange(n), math.ceil(n / mb)) if mb and mb < n else [np.arange(n)]
            yp = yt[perm] if perm is not None else None
            loss_sum = 0.0
            for c in chunks:
                c = torch.from_numpy(c).to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    out = model(**{k: v[c] for k, v in x.items()})
                    loss = criterion(out.float(), yt[c])
                    if yp is not None:
                        loss = lam * loss + (1 - lam) * criterion(out.float(), yp[c])
                frac = len(c) / n
                scaler.scale(loss * frac).backward()
                loss_sum += loss.item() * frac
            loss = torch.tensor(loss_sum)
            if args.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched_kind == "cosine":
                sched.step()
            losses.append(loss.item())

        pv, _, _ = predict(model, val, stats, device, eval_bs, amp)
        mv = all_metrics(val.labels, pv.argmax(1))
        if sched_kind == "plateau":
            sched.step(mv["icbhi_score"])
        hist.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "sec": time.time() - t0,
                     **{f"val_{k}": v for k, v in mv.items()}})
        flag = ""
        if mv["icbhi_score"] > best:
            best, best_epoch, bad, flag = mv["icbhi_score"], epoch + 1, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"   ep {epoch+1:3d} loss {np.mean(losses):.4f} | val score {mv['icbhi_score']:.2f} "
              f"Se {mv['sensitivity']:.1f} Sp {mv['specificity']:.1f} F1 {mv['macro_f1']:.2f} "
              f"| {time.time()-t0:.0f}s{flag}")
        if bad >= args.patience:
            print(f"   early stop (ไม่ดีขึ้น {args.patience} epoch)")
            break
    train_time = time.time() - t_start
    model.load_state_dict(best_state)

    row = {"time": datetime.now().isoformat(timespec="seconds"), "dataset": args.dataset, "fold": str(args.fold),
           "method": name, "role": cfg["role"], "seed": args.seed, "lr": lr, "batch_size": bs,
           "micro_batch": mb,
           "epochs_run": len(hist), "best_epoch": best_epoch, "params_M": round(n_params / 1e6, 3),
           "trainable_M": round(n_train / 1e6, 3), "train_time_min": round(train_time / 60, 2),
           "val_icbhi_score": round(best, 3), "mel_scale": args.mel_scale, "bandpass": not args.no_bandpass,
           "limit": args.limit}

    run_dir = Path(args.run_root) / args.dataset / name / f"fold{args.fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    if not args.no_test:
        pt, ms, gate = predict(model, test, stats, device, eval_bs, amp)
        pred = pt.argmax(1)
        mt = all_metrics(test.labels, pred)
        row.update({k: round(v, 3) for k, v in mt.items()})
        row["infer_ms_per_sample"] = round(ms, 3)
        row.update({k: round(v, 4) for k, v in gate.items()})
        pd.DataFrame(confusion(test.labels, pred), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(run_dir / "confusion.csv")
        pd.DataFrame({"uid": test.uids, "label": test.labels, "pred": pred,
                      **{f"p_{c}": pt[:, i] for i, c in enumerate(CLASS_NAMES)}}).to_csv(run_dir / "test_predictions.csv", index=False)
        print(f"   TEST: ICBHI score {mt['icbhi_score']:.2f} (Se {mt['sensitivity']:.2f}, Sp {mt['specificity']:.2f}) "
              f"macro-F1 {mt['macro_f1']:.2f}")
    if args.save_ckpt:
        torch.save(best_state, run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "result": row}, indent=2, default=str),
                                          encoding="utf-8")
    out_csv = Path(args.results_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out_csv.exists():
        df = pd.concat([pd.read_csv(out_csv, dtype={"fold": str}), df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return row


if __name__ == "__main__":
    run(parse_args())
