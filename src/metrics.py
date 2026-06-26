"""metrics.py — metric helpers. AUROC for Hateful Memes, F1/Acc for sarcasm."""
from __future__ import annotations

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support


def compute(task_metric: str, y_true, y_pred, y_score) -> dict:
    """Return all metrics; `primary` echoes the dataset's headline metric (matrix yaml).
    y_score = P(positive) for AUROC; y_pred = hard {0,1} for F1/Acc."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    try:
        out["auroc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["auroc"] = None  # single-class slice
    out["primary"] = out.get(task_metric)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
