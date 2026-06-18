"""Tests for training, the train/test split discipline, and evaluation."""

from __future__ import annotations

import numpy as np

from lolwin.evaluate import evaluate_all, evaluate_role, format_results
from lolwin.features import build_feature_matrix
from lolwin.model import train_all_roles, train_role


def test_train_role_holds_out_test_by_game(synthetic_players):
    """Train and test sets must not share any game_id (grouped split)."""
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Mid", feats, tune=False)
    df_role = df[df["role"] == "Mid"].reset_index(drop=True)
    test_games = set(df_role.loc[rm.X_test.index, "game_id"])
    train_games = set(df_role["game_id"]) - test_games
    assert test_games.isdisjoint(train_games)


def test_train_role_features_match(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Top", feats, tune=False)
    assert list(rm.X_test.columns) == feats


def test_evaluate_role_metric_ranges(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Bot", feats, tune=False)
    m = evaluate_role(rm)
    for v in (m.accuracy, m.roc_auc, m.f1):
        assert 0.0 <= v <= 1.0
    # On the synthetic data the signal is strong; model must beat random.
    assert m.roc_auc > 0.6
    assert m.confusion.shape == (2, 2)


def test_train_all_roles_and_table(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    models = train_all_roles(df, feats, tune=False, verbose=False)
    assert set(models) == {"Top", "Jungle", "Mid", "Bot", "Support"}
    table = evaluate_all(models)
    # 5 roles + 1 overall row.
    assert len(table) == 6
    assert table.iloc[-1]["role"] == "Overall"
    assert {"accuracy", "roc_auc", "f1"}.issubset(table.columns)


def test_reproducible_same_seed(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    a = evaluate_role(train_role(df, "Jungle", feats, tune=False, random_state=7))
    b = evaluate_role(train_role(df, "Jungle", feats, tune=False, random_state=7))
    assert np.isclose(a.roc_auc, b.roc_auc)


def test_format_results_markdown(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    models = train_all_roles(df, feats, tune=False, verbose=False)
    md = format_results(evaluate_all(models))
    assert md.startswith("| Role |")
    assert "Overall" in md
