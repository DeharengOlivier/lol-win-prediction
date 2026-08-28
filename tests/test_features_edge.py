"""Expanded feature-engineering tests: leakage guard, selection shape, edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lolwin.features import (
    ALL_FEATURES,
    BASE_FEATURES,
    DERIVED_FEATURES,
    FALLBACK_FEATURE_COUNT,
    LEAKY_COLUMNS,
    META_COLUMNS,
    add_derived_features,
    mutual_information,
    select_features,
)


def test_no_leaky_column_is_selected_even_at_a_zero_threshold(synthetic_players):
    """Hard guarantee: LEAKY_COLUMNS never appear in the selected features,

    even if a degenerate threshold would otherwise let everything through.
    """
    df = add_derived_features(synthetic_players)
    selected = select_features(df, candidate_features=ALL_FEATURES, threshold=0.0)
    assert set(LEAKY_COLUMNS).isdisjoint(selected)


def test_meta_columns_are_never_selected(synthetic_players):
    df = add_derived_features(synthetic_players)
    feats = select_features(df)
    for col in META_COLUMNS:
        assert col not in feats


def test_a_leaky_column_offered_as_a_candidate_is_still_refused(synthetic_players):
    """The exclusion holds against the caller, not only against the default set.

    tower_kills separates the classes perfectly in the synthetic data, so a
    selector that only ranked by mutual information would put it first. It is
    filtered inside select_features, which is the single function that decides
    the model's inputs.
    """
    df = add_derived_features(synthetic_players)
    ranked = mutual_information(df, ["tower_kills", "player_kills"])
    assert ranked.iloc[0]["feature"] == "tower_kills"  # it WOULD rank first

    selected = select_features(
        df, candidate_features=["tower_kills", "player_kills"], threshold=0.0
    )
    assert selected == ["player_kills"]


def test_selected_features_are_a_subset_of_all_features(synthetic_players):
    df = add_derived_features(synthetic_players)
    feats = select_features(df)
    assert set(feats).issubset(set(ALL_FEATURES))
    assert len(feats) >= 1


def test_select_features_threshold_filters_monotonically(synthetic_players):
    df = add_derived_features(synthetic_players)
    loose = select_features(df, threshold=0.0)
    strict = select_features(df, threshold=0.05)
    # A stricter threshold can never keep more features than a looser one
    # (modulo the fallback, which only triggers when strict would be empty).
    assert len(strict) <= len(loose) or len(strict) == FALLBACK_FEATURE_COUNT


def test_select_features_fallback_when_threshold_too_high(synthetic_players):
    df = add_derived_features(synthetic_players)
    # An impossibly high threshold wipes everything -> fallback keeps the best few.
    feats = select_features(df, threshold=10.0)
    assert len(feats) == FALLBACK_FEATURE_COUNT


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
    df = pd.DataFrame([{c: 0 for c in BASE_FEATURES}])
    out = add_derived_features(df)
    vals = out.loc[0, DERIVED_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(vals).all()


def test_all_features_is_base_plus_derived():
    assert ALL_FEATURES == BASE_FEATURES + DERIVED_FEATURES
    # No accidental overlap between base and derived names.
    assert set(BASE_FEATURES).isdisjoint(DERIVED_FEATURES)
