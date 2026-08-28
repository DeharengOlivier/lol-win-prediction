"""Expanded model + split-discipline tests, all on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from lolwin.features import add_derived_features
from lolwin.model import (
    DEFAULT_PARAM_GRID,
    UNTUNED_PARAMS,
    _grid_combinations,
    split_by_game,
    train_role,
    tune_role,
)


def _prepared(players, **split_kwargs):
    split = split_by_game(add_derived_features(players), **split_kwargs)
    return split, split.select_features()


def test_grid_combinations_cardinality():
    grid = {"a": [1, 2], "b": [10, 20, 30], "c": [0]}
    combos = _grid_combinations(grid)
    assert len(combos) == 2 * 3 * 1
    # Every combo carries exactly the grid's keys.
    assert all(set(c) == {"a", "b", "c"} for c in combos)


def test_default_grid_expands_to_expected_size():
    combos = _grid_combinations(DEFAULT_PARAM_GRID)
    expected = 1
    for values in DEFAULT_PARAM_GRID.values():
        expected *= len(values)
    assert len(combos) == expected


@pytest.mark.parametrize("seed", [1, 13])
def test_no_game_straddles_the_split(synthetic_players, seed):
    split, _ = _prepared(synthetic_players, random_state=seed)
    assert set(split.train["game_id"]).isdisjoint(set(split.test["game_id"]))


def test_test_set_fraction_is_respected(synthetic_players):
    split, _ = _prepared(synthetic_players, test_size=0.3)
    total = len(split.train) + len(split.test)
    # The split is per game, so allow a tolerance band around 30%.
    assert 0.2 <= len(split.test) / total <= 0.4


def test_a_role_with_no_training_row_is_refused(synthetic_players):
    split, feats = _prepared(synthetic_players)
    with pytest.raises(ValueError, match="Unranked"):
        train_role(split, "Unranked", feats, tune=False)


def test_tune_role_returns_params_from_grid(synthetic_players):
    split, feats = _prepared(synthetic_players)
    train = split.train[split.train["role"] == "Bot"]
    small_grid = {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "max_depth": [3],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
    }
    best_params, auc = tune_role(
        train[feats],
        train["win"],
        train["game_id"],
        param_grid=small_grid,
        n_splits=3,
    )
    assert set(best_params) == set(small_grid)
    assert best_params["n_estimators"] == 50
    assert 0.0 <= auc <= 1.0
    # Strong synthetic signal: CV AUC should beat random.
    assert auc > 0.6


def test_tune_role_picks_one_of_the_offered_configs(synthetic_players):
    split, feats = _prepared(synthetic_players)
    train = split.train[split.train["role"] == "Top"]
    grid = {
        "n_estimators": [10, 200],
        "learning_rate": [0.1],
        "max_depth": [4],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
    }
    best_params, _ = tune_role(
        train[feats], train["win"], train["game_id"], param_grid=grid, n_splits=3
    )
    assert best_params["n_estimators"] in (10, 200)


def test_no_tune_uses_the_documented_default_and_fits(synthetic_players):
    split, feats = _prepared(synthetic_players)
    role_model = train_role(split, "Support", feats, tune=False)
    assert role_model.best_params == UNTUNED_PARAMS
    assert np.isnan(role_model.cv_auc)  # no CV ran
    held_out = split.test[split.test["role"] == "Support"]
    proba = role_model.model.predict_proba(held_out[feats])
    assert proba.shape == (len(held_out), 2)


def test_the_untuned_defaults_are_not_shared_between_models(synthetic_players):
    """best_params must be a copy, so mutating one model cannot affect the next."""
    split, feats = _prepared(synthetic_players)
    first = train_role(split, "Top", feats, tune=False)
    first.best_params["max_depth"] = 99
    second = train_role(split, "Mid", feats, tune=False)
    assert second.best_params["max_depth"] == UNTUNED_PARAMS["max_depth"]
    assert UNTUNED_PARAMS["max_depth"] != 99
