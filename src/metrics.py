"""Metric ตาม RD_v2 ส่วนที่ 5

ICBHI Score = (Se + Sp) / 2  ตามนิยามของ ICBHI challenge (เหมือน get_score ของ Patch-Mix)
    Sp = สัดส่วนรอบปกติที่ทำนายเป็น normal
    Se = สัดส่วนรอบผิดปกติ (crackle, wheeze, both) ที่ทำนาย "ถูกคลาส"
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

from .utils import CLASS_NAMES


def icbhi_scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    normal = y_true == 0
    sp = (y_pred[normal] == 0).mean() if normal.any() else np.nan
    ab = ~normal
    se = (y_pred[ab] == y_true[ab]).mean() if ab.any() else np.nan
    return 100 * sp, 100 * se, 100 * (sp + se) / 2


def all_metrics(y_true, y_pred):
    sp, se, score = icbhi_scores(y_true, y_pred)
    labels = list(range(len(CLASS_NAMES)))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "icbhi_score": score, "sensitivity": se, "specificity": sp,
        "macro_f1": 100 * f1.mean(),
        "balanced_acc": 100 * balanced_accuracy_score(y_true, y_pred),
        "accuracy": 100 * float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
    }
    for name, v in zip(CLASS_NAMES, f1):
        out[f"f1_{name}"] = 100 * v
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
