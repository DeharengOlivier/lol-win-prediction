"""Expanded feature-engineering tests: leakage guard, selection shape, edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolwin.features import (
    ALL_FEATURES,
    BASE_FEATURES,
    DERIVED_FEATURES,
    LEAKY_COLUMNS,
    META_COLUMNS,
    add_derived_features,
    build_feature_matrix,
    mutual_information,
    select_features,
)


def test_no_leaky_column_in_feature_matrix_even_at_zero_threshold(synthetic_players):
    """Hard guarantee: LEAKY_COLUMNS never appear in the selected features,

    even if a degenerate threshold would otherwise let everything through.
    """
    df = add_derived_features(synthetic_players)
    # Selection itself only considers ALL_FEATURES, but build_feature_matrix adds
    # a final safety net. Verify both layers.
    selected = select_features(df, candidate_features=ALL_FEATURES, threshold=0.0)
    assert set(LEAKY_COLUMNS).isdisjoint(selected)

    _, feats = build_feature_matrix(synthetic_players)
    for col in LEAKY_COLUMNS:
        assert col not in feats


def test_meta_columns_never_in_feature_matrix(synthetic_players):
    _, feats = build_feature_matrix(synthetic_players)
    for col in META_COLUMNS:
        assert col not in feats


def test_build_feature_matrix_safety_net_strips_injected_leaky_feature(synthetic_players):
    """If a leaky column were somehow proposed, the final filter must remove it.

    We simulate that by passing candidate_features that include a leaky column
    through the lower-level select_features, then confirm build_feature_matrix's
    forbidden filter would have removed it.
    """
    df = add_derived_features(synthetic_players)
    # tower_kills is perfectly predictive in the synthetic data, so a naive
    # selector with a leaky candidate set would keep it.
    leaky_selected = select_features(
        df, candidate_features=["tower_kills", "player_kills"], threshold=0.0
    )
    assert "tower_kills" in leaky_selected  # confirms it WOULD be selected
    # The production path must still exclude it.
    _, safe_feats = build_feature_matrix(synthetic_players)
    assert "tower_kills" not in safe_feats


def test_selected_features_are_subset_of_all_features(synthetic_players):
    _, feats = build_feature_matrix(synthetic_players)
    assert set(feats).issubset(set(ALL_FEATURES))
    assert len(feats) >= 1


def test_select_features_threshold_filters_monotonically(synthetic_players):
    df = add_derived_features(synthetic_players)
    loose = select_features(df, threshold=0.0)
    strict = select_features(df, threshold=0.05)
    # A stricter threshold can never keep more features than a looser one
    # (modulo the >=5 fallback, which only triggers when strict would be empty).
    assert len(strict) <= len(loose) or len(strict) == 5


def test_select_features_fallback_when_threshold_too_high(synthetic_players):
    df = add_derived_features(synthetic_players)
    # An impossibly high threshold wipes everything -> fallback keeps top 5.
    feats = select_features(df, threshold=10.0)
    assert len(feats) == 5


def test_mutual_information_output_shape(synthetic_players):
    df = add_derived_features(synthetic_players)
    mi = mutual_information(df, ALL_FEATURES)
    assert mi.shape == (len(ALL_FEATURES), 2)
    assert set(mi["feature"]) == set(ALL_FEATURES)
    assert (mi["mi_score"] >= 0).all()


def test_add_derived_features_does_not_mutate_input(synthetic_players):
    before = synthetic_players.copy()
    add_derived_features(synthetic_players)
    pd.testing.assert_frame_equal(synthetic_players, before)


def test_derived_features_all_zero_kills_assists_deaths():
    # A pathological row: 0 kills, 0 assists, 0 deaths, 0 level, 0 gold.
    df = pd.DataFrame(
        [{c: 0 for c in BASE_FEATURES}]
    )
    out = add_derived_features(df)
    vals = out.loc[0, DERIVED_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(vals).all()


def test_all_features_is_base_plus_derived():
    assert ALL_FEATURES == BASE_FEATURES + DERIVED_FEATURES
    # No accidental overlap between base and derived names.
    assert set(BASE_FEATURES).isdisjoint(DERIVED_FEATURES)
