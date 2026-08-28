"""Tests for training, the train/test split discipline, and evaluation."""

from __future__ import annotations

import numpy as np

from lolwin.evaluate import evaluate_all, evaluate_role, format_results
from lolwin.features import add_derived_features
from lolwin.model import split_by_game, train_all_roles, train_role


def _prepared(players):
    """The pipeline's own order: derive, split, then select on training rows."""
    split = split_by_game(add_derived_features(players))
    return split, split.select_features()


def test_the_split_holds_out_whole_games(synthetic_players):
    split, _ = _prepared(synthetic_players)
    assert set(split.train["game_id"]).isdisjoint(set(split.test["game_id"]))


def test_train_role_records_the_features_it_was_given(synthetic_players):
    split, feats = _prepared(synthetic_players)
    role_model = train_role(split, "Top", feats, tune=False)
    assert role_model.features == feats


def test_evaluate_role_metric_ranges(synthetic_players):
    split, feats = _prepared(synthetic_players)
    role_model = train_role(split, "Bot", feats, tune=False)
    metrics = evaluate_role(role_model, split.test)
    for value in (metrics.accuracy, metrics.roc_auc, metrics.f1):
        assert 0.0 <= value <= 1.0
    # On the synthetic data the signal is strong; the model must beat random.
    assert metrics.roc_auc > 0.6
    assert metrics.confusion.shape == (2, 2)


def test_train_all_roles_and_table(synthetic_players):
    split, feats = _prepared(synthetic_players)
    models = train_all_roles(split, feats, tune=False)
    assert set(models) == {"Top", "Jungle", "Mid", "Bot", "Support"}
    table = evaluate_all(models, split.test)
    # 5 roles + 1 overall row.
    assert len(table) == 6
    assert table.iloc[-1]["role"] == "Overall"
    assert {"accuracy", "roc_auc", "f1"}.issubset(table.columns)


def test_reproducible_same_seed(synthetic_players):
    split, feats = _prepared(synthetic_players)
    first = evaluate_role(train_role(split, "Jungle", feats, tune=False), split.test)
    second = evaluate_role(train_role(split, "Jungle", feats, tune=False), split.test)
    assert np.isclose(first.roc_auc, second.roc_auc)


def test_format_results_markdown(synthetic_players):
    split, feats = _prepared(synthetic_players)
    models = train_all_roles(split, feats, tune=False)
    markdown = format_results(evaluate_all(models, split.test))
    assert markdown.startswith("| Role |")
    assert "Overall" in markdown
