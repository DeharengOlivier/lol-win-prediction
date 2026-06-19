"""Expanded evaluation tests: metric correctness on a tiny known set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolwin.evaluate import RoleMetrics, evaluate_all, evaluate_role, format_results
from lolwin.features import build_feature_matrix
from lolwin.model import RoleModel, train_all_roles, train_role


class _PerfectModel:
    """A stub model whose probabilities perfectly match the labels."""

    def __init__(self, y):
        self._y = np.asarray(y, dtype=float)

    def predict_proba(self, X):
        p1 = self._y  # exact label as the positive-class probability
        return np.column_stack([1 - p1, p1])


def _role_model_from(y_test) -> RoleModel:
    y = pd.Series(y_test, name="win")
    X = pd.DataFrame({"f": np.arange(len(y), dtype=float)})
    return RoleModel(
        role="Test",
        model=_PerfectModel(y),
        features=["f"],
        best_params={},
        cv_auc=float("nan"),
        X_test=X,
        y_test=y,
    )


def test_perfect_predictions_give_perfect_metrics():
    rm = _role_model_from([0, 1, 0, 1, 1, 0])
    m = evaluate_role(rm)
    assert m.accuracy == 1.0
    assert m.roc_auc == 1.0
    assert m.f1 == 1.0
    # Confusion matrix: all on the diagonal. TN=3, TP=3, no errors.
    assert m.confusion.tolist() == [[3, 0], [0, 3]]
    assert m.n_test == 6


def test_metrics_in_unit_range_on_synthetic(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Mid", feats, tune=False)
    m = evaluate_role(rm)
    for v in (m.accuracy, m.roc_auc, m.f1):
        assert 0.0 <= v <= 1.0


def test_threshold_affects_predictions():
    # All probabilities are 0.4. At threshold 0.5 nothing is positive; at 0.3 all are.
    y = pd.Series([0, 1, 0, 1], name="win")
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})

    class _Const:
        def predict_proba(self, X):
            p = np.full(len(X), 0.4)
            return np.column_stack([1 - p, p])

    rm = RoleModel("R", _Const(), ["f"], {}, float("nan"), X, y)
    high = evaluate_role(rm, threshold=0.5)  # predict all 0
    low = evaluate_role(rm, threshold=0.3)  # predict all 1
    # With all-0 predictions, the two actual positives are false negatives.
    assert high.confusion[1, 1] == 0  # no true positives
    assert low.confusion[1, 1] == 2  # both positives caught


def test_evaluate_all_overall_row_is_weighted_average(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    models = train_all_roles(df, feats, tune=False, verbose=False)
    table = evaluate_all(models)

    per_role = table[table["role"] != "Overall"]
    overall = table[table["role"] == "Overall"].iloc[0]

    expected_acc = np.average(per_role["accuracy"], weights=per_role["n_test"])
    assert overall["accuracy"] == pytest.approx(expected_acc)
    assert overall["n_test"] == per_role["n_test"].sum()


def test_evaluate_all_has_one_row_per_role_plus_overall(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    models = train_all_roles(df, feats, tune=False, verbose=False)
    table = evaluate_all(models)
    assert len(table) == len(models) + 1
    assert table.iloc[-1]["role"] == "Overall"


def test_format_results_renders_all_rows(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    models = train_all_roles(df, feats, tune=False, verbose=False)
    md = format_results(evaluate_all(models))
    lines = md.splitlines()
    # header + separator + 5 roles + overall = 8 lines
    assert len(lines) == 2 + len(models) + 1
    assert "Overall" in md
