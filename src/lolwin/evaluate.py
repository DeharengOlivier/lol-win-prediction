"""Evaluation metrics: accuracy, ROC-AUC, F1, confusion matrix.

We report several metrics rather than accuracy alone. On this balanced target
accuracy is meaningful, but ROC-AUC (threshold-free ranking quality) and F1
(precision/recall balance) give a fuller and more honest picture.

The held-out frame is passed in explicitly rather than carried by the model, so
scoring a model on rows it was fitted on takes a deliberate act.
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

DEFAULT_THRESHOLD = 0.5


@dataclass
class RoleMetrics:
    """Held-out test metrics for one role."""

    role: str
    n_test: int
    accuracy: float
    roc_auc: float
    f1: float
    confusion: np.ndarray  # 2x2 [[TN, FP], [FN, TP]]


def evaluate_role(
    role_model: RoleModel,
    test: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    target: str = "win",
) -> RoleMetrics:
    """Score a trained role model on the held-out rows of its own role.

    Complexity: O(rows x trees) for the prediction, O(rows) for the metrics.

    Args:
        role_model: The fitted model.
        test: The held-out frame, normally ``split.test``. Rows of other roles
            are filtered out here.
        threshold: Probability above which a row is predicted as a win.
        target: Target column.

    Returns:
        The metrics for this role.

    Raises:
        ValueError: The held-out frame carries no row for this role, or only
            one class, which leaves ROC-AUC undefined.
    """
    rows = test[test["role"] == role_model.role]
    if rows.empty:
        raise ValueError(f"No held-out row for role {role_model.role!r}.")
    y_test = rows[target]
    if y_test.nunique() < 2:
        raise ValueError(
            f"Held-out rows for role {role_model.role!r} contain a single class, "
            "so ROC-AUC is undefined. Use a larger test set."
        )

    proba = role_model.model.predict_proba(rows[role_model.features])[:, 1]
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
    models: dict[str, RoleModel],
    test: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Evaluate every role on the held-out frame and return a tidy results table.

    Columns: role, n_test, accuracy, roc_auc, f1. The final ``Overall`` row is
    the test-size-weighted mean of the per-role metrics. For ROC-AUC that is a
    mean of five per-role AUCs, not the AUC of the pooled predictions: the roles
    are separate ranking problems and pooling them would score a comparison the
    model is never asked to make. All five roles hold out the same games, so the
    weights are equal in practice.
    """
    rows: list[dict[str, object]] = []
    for role_model in models.values():
        metrics = evaluate_role(role_model, test, threshold)
        rows.append(
            {
                "role": metrics.role,
                "n_test": metrics.n_test,
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc,
                "f1": metrics.f1,
            }
        )
    table = pd.DataFrame(rows)

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
    for _, row in table.iterrows():
        lines.append(
            f"| {row['role']} | {int(row['n_test'])} | {row['accuracy']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['f1']:.4f} |"
        )
    return "\n".join(lines)
