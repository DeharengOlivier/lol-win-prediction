"""Expanded evaluation tests: metric correctness on a tiny known set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolwin.evaluate import evaluate_all, evaluate_role, format_results
from lolwin.features import add_derived_features
from lolwin.model import RoleModel, split_by_game, train_all_roles, train_role


def _prepared(players):
    split = split_by_game(add_derived_features(players))
    return split, split.select_features()


class _PerfectModel:
    """A stub model whose probabilities perfectly match the labels."""

    def __init__(self, y):
        self._y = np.asarray(y, dtype=float)

    def predict_proba(self, X):
        p1 = self._y  # exact label as the positive-class probability
        return np.column_stack([1 - p1, p1])


def _stub(model, labels) -> tuple[RoleModel, pd.DataFrame]:
    """A role model plus the held-out frame it is meant to be scored on."""
    test = pd.DataFrame(
        {
            "role": ["Test"] * len(labels),
            "win": list(labels),
            "f": np.arange(len(labels), dtype=float),
        }
    )
    role_model = RoleModel(
        role="Test",
        model=model,
        features=["f"],
        best_params={},
        cv_auc=float("nan"),
    )
    return role_model, test


def test_perfect_predictions_give_perfect_metrics():
    labels = [0, 1, 0, 1, 1, 0]
    role_model, test = _stub(_PerfectModel(labels), labels)
    metrics = evaluate_role(role_model, test)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.f1 == 1.0
    # Confusion matrix: all on the diagonal. TN=3, TP=3, no errors.
    assert metrics.confusion.tolist() == [[3, 0], [0, 3]]
    assert metrics.n_test == 6


def test_metrics_in_unit_range_on_synthetic(synthetic_players):
    split, feats = _prepared(synthetic_players)
    metrics = evaluate_role(train_role(split, "Mid", feats, tune=False), split.test)
    for value in (metrics.accuracy, metrics.roc_auc, metrics.f1):
        assert 0.0 <= value <= 1.0


def test_threshold_affects_predictions():
    # All probabilities are 0.4. At threshold 0.5 nothing is positive; at 0.3 all are.
    class _Const:
        def predict_proba(self, X):
            p = np.full(len(X), 0.4)
            return np.column_stack([1 - p, p])

    role_model, test = _stub(_Const(), [0, 1, 0, 1])
    high = evaluate_role(role_model, test, threshold=0.5)  # predict all 0
    low = evaluate_role(role_model, test, threshold=0.3)  # predict all 1
    assert high.confusion[1, 1] == 0  # no true positives
    assert low.confusion[1, 1] == 2  # both positives caught


def test_a_role_absent_from_the_held_out_frame_is_refused():
    role_model, test = _stub(_PerfectModel([0, 1]), [0, 1])
    with pytest.raises(ValueError, match="No held-out row"):
        evaluate_role(role_model, test.iloc[0:0])


def test_a_single_class_held_out_frame_is_refused():
    """ROC-AUC is undefined on one class; saying so beats an sklearn traceback."""
    role_model, test = _stub(_PerfectModel([1, 1, 1]), [1, 1, 1])
    with pytest.raises(ValueError, match="single class"):
        evaluate_role(role_model, test)


def test_evaluate_role_ignores_rows_of_other_roles():
    labels = [0, 1, 0, 1]
    role_model, test = _stub(_PerfectModel(labels), labels)
    other = test.copy()
    other["role"] = "Other"
    other["win"] = 1 - other["win"]
    metrics = evaluate_role(role_model, pd.concat([test, other], ignore_index=True))
    assert metrics.n_test == 4
    assert metrics.accuracy == 1.0


def test_evaluate_all_overall_row_is_weighted_average(synthetic_players):
    split, feats = _prepared(synthetic_players)
    models = train_all_roles(split, feats, tune=False)
    table = evaluate_all(models, split.test)

    per_role = table[table["role"] != "Overall"]
    overall = table[table["role"] == "Overall"].iloc[0]

    expected_acc = np.average(per_role["accuracy"], weights=per_role["n_test"])
    assert overall["accuracy"] == pytest.approx(expected_acc)
    assert overall["n_test"] == per_role["n_test"].sum()


def test_evaluate_all_has_one_row_per_role_plus_overall(synthetic_players):
    split, feats = _prepared(synthetic_players)
    models = train_all_roles(split, feats, tune=False)
    table = evaluate_all(models, split.test)
    assert len(table) == len(models) + 1
    assert table.iloc[-1]["role"] == "Overall"


def test_format_results_renders_all_rows(synthetic_players):
    split, feats = _prepared(synthetic_players)
    models = train_all_roles(split, feats, tune=False)
    markdown = format_results(evaluate_all(models, split.test))
    lines = markdown.splitlines()
    # header + separator + 5 roles + overall = 8 lines
    assert len(lines) == 2 + len(models) + 1
    assert "Overall" in markdown
