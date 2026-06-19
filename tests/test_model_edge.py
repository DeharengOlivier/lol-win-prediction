"""Expanded model + split-discipline tests, all on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lolwin.features import build_feature_matrix
from lolwin.model import (
    DEFAULT_PARAM_GRID,
    _grid_combinations,
    train_role,
    tune_role,
)


def test_grid_combinations_cardinality():
    grid = {"a": [1, 2], "b": [10, 20, 30], "c": [0]}
    combos = _grid_combinations(grid)
    assert len(combos) == 2 * 3 * 1
    # Every combo carries exactly the grid's keys.
    assert all(set(c) == {"a", "b", "c"} for c in combos)


def test_default_grid_expands_to_expected_size():
    combos = _grid_combinations(DEFAULT_PARAM_GRID)
    expected = 1
    for v in DEFAULT_PARAM_GRID.values():
        expected *= len(v)
    assert len(combos) == expected


def test_grouped_split_no_game_straddles_boundary(synthetic_players):
    """Reinforced version of the split test across multiple roles and seeds."""
    df, feats = build_feature_matrix(synthetic_players)
    for role in ("Top", "Support"):
        for seed in (1, 13):
            rm = train_role(df, role, feats, tune=False, random_state=seed)
            df_role = df[df["role"] == role].reset_index(drop=True)
            test_games = set(df_role.loc[rm.X_test.index, "game_id"])
            train_idx = df_role.index.difference(rm.X_test.index)
            train_games = set(df_role.loc[train_idx, "game_id"])
            assert test_games.isdisjoint(train_games), (
                f"game_id straddled split for role={role} seed={seed}"
            )


def test_test_set_fraction_is_respected(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Mid", feats, tune=False, test_size=0.3)
    df_role = df[df["role"] == "Mid"]
    n_test = len(rm.X_test)
    # Grouped split is per-game; allow a tolerance band around 30%.
    frac = n_test / len(df_role)
    assert 0.2 <= frac <= 0.4


def test_train_role_x_test_columns_are_exactly_features(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Jungle", feats, tune=False)
    assert list(rm.X_test.columns) == feats
    # No leaky/meta columns survive into the model's input space.
    assert "game_id" not in rm.X_test.columns
    assert "tower_kills" not in rm.X_test.columns


def test_tune_role_returns_params_from_grid(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    df_role = df[df["role"] == "Bot"].reset_index(drop=True)
    small_grid = {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "max_depth": [3],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
    }
    best_params, auc = tune_role(
        df_role[feats],
        df_role["win"],
        df_role["game_id"],
        param_grid=small_grid,
        n_splits=3,
    )
    assert set(best_params) == set(small_grid)
    assert best_params["n_estimators"] == 50
    assert 0.0 <= auc <= 1.0
    # Strong synthetic signal: CV AUC should beat random.
    assert auc > 0.6


def test_tune_role_picks_best_of_two_configs(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    df_role = df[df["role"] == "Top"].reset_index(drop=True)
    grid = {
        "n_estimators": [10, 200],
        "learning_rate": [0.1],
        "max_depth": [4],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
    }
    best_params, _ = tune_role(
        df_role[feats], df_role["win"], df_role["game_id"], param_grid=grid, n_splits=3
    )
    # best_params must be one of the two candidate configs.
    assert best_params["n_estimators"] in (10, 200)


def test_no_tune_uses_default_config_and_fits(synthetic_players):
    df, feats = build_feature_matrix(synthetic_players)
    rm = train_role(df, "Support", feats, tune=False)
    assert rm.best_params["n_estimators"] == 300
    assert np.isnan(rm.cv_auc)  # no CV ran
    # Model is fitted and can predict.
    proba = rm.model.predict_proba(rm.X_test)
    assert proba.shape == (len(rm.X_test), 2)
