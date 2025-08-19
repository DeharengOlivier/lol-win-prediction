"""Evaluation metrics: accuracy, ROC-AUC, F1, confusion matrix.

We report several metrics rather than accuracy alone. On this balanced target
accuracy is meaningful, but ROC-AUC (threshold-free ranking quality) and F1
(precision/recall balance) give a fuller and more honest picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .model import RoleModel


@dataclass
class RoleMetrics:
    """Held-out test metrics for one role."""

    role: str
    n_test: int
    accuracy: float
    roc_auc: float
    f1: float
    confusion: np.ndarray  # 2x2 [[TN, FP], [FN, TP]]


def evaluate_role(role_model: RoleModel, threshold: float = 0.5) -> RoleMetrics:
    """Score a trained role model on its held-out test set.

    The test set was set aside in :func:`lolwin.model.train_role` and never seen
    during tuning or fitting, so these numbers are an unbiased estimate.
    """
    X_test, y_test = role_model.X_test, role_model.y_test
    proba = role_model.model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return RoleMetrics(
        role=role_model.role,
        n_test=len(y_test),
        accuracy=float(accuracy_score(y_test, pred)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        f1=float(f1_score(y_test, pred)),
        confusion=confusion_matrix(y_test, pred),
    )


def evaluate_all(
    models: dict[str, RoleModel], threshold: float = 0.5
) -> pd.DataFrame:
    """Evaluate every role and return a tidy results table.

    Columns: role, n_test, accuracy, roc_auc, f1. A final ``Overall`` row gives
    the sample-size-weighted average of each metric.
    """
    rows: list[dict] = []
    for role, rm in models.items():
        m = evaluate_role(rm, threshold)
        rows.append(
            {
                "role": m.role,
                "n_test": m.n_test,
                "accuracy": m.accuracy,
                "roc_auc": m.roc_auc,
                "f1": m.f1,
            }
        )
    table = pd.DataFrame(rows)

    # Weighted overall row (weight by test size so larger roles count more).
    weights = table["n_test"]
    overall = {
        "role": "Overall",
        "n_test": int(weights.sum()),
        "accuracy": float(np.average(table["accuracy"], weights=weights)),
        "roc_auc": float(np.average(table["roc_auc"], weights=weights)),
        "f1": float(np.average(table["f1"], weights=weights)),
    }
    return pd.concat([table, pd.DataFrame([overall])], ignore_index=True)


def format_results(table: pd.DataFrame) -> str:
    """Render a results table as a Markdown-friendly string for the README/logs."""
    lines = ["| Role | n_test | Accuracy | ROC-AUC | F1 |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['role']} | {int(r['n_test'])} | {r['accuracy']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['f1']:.4f} |"
        )
    return "\n".join(lines)
