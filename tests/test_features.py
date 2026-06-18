"""Tests for feature engineering and selection, including the leakage guard."""

from __future__ import annotations

import numpy as np

from lolwin.features import (
    DERIVED_FEATURES,
    LEAKY_COLUMNS,
    META_COLUMNS,
    add_derived_features,
    build_feature_matrix,
    mutual_information,
    select_features,
)


def test_add_derived_features_creates_columns(synthetic_players):
    df = add_derived_features(synthetic_players)
    for col in DERIVED_FEATURES:
        assert col in df.columns
    # No NaN / inf even though some players have 0 kills or 0 deaths.
    assert np.isfinite(df[DERIVED_FEATURES].to_numpy()).all()


def test_derived_features_handle_zero_division(synthetic_players):
    df = synthetic_players.copy()
    df.loc[df.index[0], "player_deaths"] = 0
    df.loc[df.index[0], "player_kills"] = 0
    out = add_derived_features(df)
    assert np.isfinite(out.loc[out.index[0], DERIVED_FEATURES].to_numpy()).all()


def test_mutual_information_is_sorted(synthetic_players):
    df = add_derived_features(synthetic_players)
    mi = mutual_information(df, ["player_kills", "player_deaths", "gold_earned"])
    assert list(mi.columns) == ["feature", "mi_score"]
    assert mi["mi_score"].is_monotonic_decreasing


def test_select_features_returns_subset(synthetic_players):
    df = add_derived_features(synthetic_players)
    feats = select_features(df, threshold=0.0)
    assert len(feats) >= 1


def test_build_feature_matrix_excludes_leaky_and_meta(synthetic_players):
    """The core leakage guarantee: no team-objective or id column ever selected."""
    _, feats = build_feature_matrix(synthetic_players)
    forbidden = set(LEAKY_COLUMNS) | set(META_COLUMNS)
    assert forbidden.isdisjoint(feats), f"Leaky/meta column leaked into {feats}"


def test_leaky_columns_would_dominate_if_used(synthetic_players):
    """Sanity: the leaky columns are far more predictive than legit ones.

    This documents *why* they are excluded: in the synthetic data they perfectly
    separate the classes, exactly like in the real data.
    """
    df = add_derived_features(synthetic_players)
    mi_leaky = mutual_information(df, ["tower_kills"])["mi_score"].iloc[0]
    mi_legit = mutual_information(df, ["player_kills"])["mi_score"].iloc[0]
    assert mi_leaky > mi_legit
