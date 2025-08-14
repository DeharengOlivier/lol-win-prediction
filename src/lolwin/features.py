"""Feature engineering and mutual-information feature selection.

The central decision here is *which columns are allowed as model inputs*. The
project predicts a match outcome from one player's own end-of-game stats, so we
only ever use player-performance columns and features derived from them. We
explicitly forbid team-level objective columns, which are leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Raw player-performance columns. These describe what a single player did and
# are the legitimate, intended inputs of the model.
BASE_FEATURES: list[str] = [
    "player_kills",
    "player_deaths",
    "player_assists",
    "total_minions_killed",
    "gold_earned",
    "level",
    "total_damage_dealt_to_champions",
    "total_damage_taken",
    "wards_placed",
    "largest_killing_spree",
    "largest_multi_kill",
]

# DATA LEAKAGE GUARD.
# These columns are shared by all five players of a team and essentially encode
# the result (measured univariate ROC-AUC against ``win`` in parentheses):
#   tower_kills (0.99), inhibitor_kills (0.97), team_kills (0.91),
#   baron_kills (0.86), dragon_kills (0.84), herald_kills (0.65).
# A team that took the enemy towers/inhibitors has, by the rules of the game,
# almost certainly won. Feeding them in would inflate metrics to a meaningless
# ~0.99 and defeat the purpose ("what should a *player* focus on"). They are
# never added to the feature matrix; this list documents the exclusion.
LEAKY_COLUMNS: list[str] = [
    "team_kills",
    "tower_kills",
    "inhibitor_kills",
    "dragon_kills",
    "herald_kills",
    "baron_kills",
]

# Identifier / metadata columns that must never enter X.
META_COLUMNS: list[str] = ["win", "role", "player_name", "player_id", "game_id"]


def add_derived_features(players: pd.DataFrame) -> pd.DataFrame:
    """Add ratio / log features derived only from player-performance columns.

    Each derived feature has a gameplay rationale:
    - ``kda_ratio``      : (kills + assists) / deaths, the canonical efficiency
      metric. Deaths are floored at 1 to avoid division by zero.
    - ``assists_share``  : how much of a player's kill participation comes from
      assists vs solo kills (separates carries from enablers).
    - ``gold_per_level`` : gold efficiency relative to experience gained.
    - ``gold_per_kill``  : how much gold the player accrued per kill (farming vs
      fighting income).
    - ``spree_per_kill`` : how concentrated the kills were (snowballing).
    - ``log_gold``       : gold is right-skewed; the log stabilises it.

    Returns a new DataFrame with the extra columns appended.
    """
    df = players.copy()
    safe_deaths = df["player_deaths"].replace(0, 1)
    kill_participation = (df["player_kills"] + df["player_assists"]).replace(0, 1)

    df["kda_ratio"] = (df["player_kills"] + df["player_assists"]) / safe_deaths
    df["assists_share"] = df["player_assists"] / kill_participation
    df["gold_per_level"] = df["gold_earned"] / df["level"].replace(0, 1)
    df["gold_per_kill"] = df["gold_earned"] / df["player_kills"].replace(0, 1)
    df["spree_per_kill"] = df["largest_killing_spree"] / df["player_kills"].replace(0, 1)
    df["log_gold"] = np.log1p(df["gold_earned"])
    return df


DERIVED_FEATURES: list[str] = [
    "kda_ratio",
    "assists_share",
    "gold_per_level",
    "gold_per_kill",
    "spree_per_kill",
    "log_gold",
]

ALL_FEATURES: list[str] = BASE_FEATURES + DERIVED_FEATURES


def mutual_information(
    df: pd.DataFrame,
    features: list[str],
    target: str = "win",
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by mutual information with the target.

    Mutual information captures non-linear dependence, which is appropriate here
    because the relationship between, say, deaths and winning is far from
    linear. We standardise first (MI is scale-invariant in theory, but
    standardising keeps the estimator's k-NN distances well-behaved).

    Returns a DataFrame sorted by descending MI score, columns
    ``["feature", "mi_score"]``.
    """
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    scores = mutual_info_classif(
        X_scaled, y, discrete_features=False, random_state=random_state
    )
    return (
        pd.DataFrame({"feature": features, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )


def select_features(
    df: pd.DataFrame,
    candidate_features: list[str] = ALL_FEATURES,
    target: str = "win",
    threshold: float = 0.01,
    random_state: int = 42,
) -> list[str]:
    """Select features whose mutual information with the target exceeds a floor.

    Args:
        df: DataFrame already containing the candidate columns (call
            :func:`add_derived_features` first).
        candidate_features: Columns to consider.
        target: Target column name.
        threshold: Minimum MI score to keep a feature. The default 0.01 drops
            near-useless features (e.g. wards_placed, total_minions_killed) that
            only add noise and training time.
        random_state: Seed for the MI estimator.

    Returns:
        The list of retained feature names, ordered by descending MI.
    """
    ranked = mutual_information(df, candidate_features, target, random_state)
    selected = ranked.loc[ranked["mi_score"] >= threshold, "feature"].tolist()
    # Guard against a too-aggressive threshold wiping out all features.
    if not selected:
        selected = ranked["feature"].head(5).tolist()
    return selected


def build_feature_matrix(
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Convenience: add derived features and return (df, selected_feature_list).

    This is the one call the pipeline needs to go from a cleaned player table to
    a model-ready DataFrame plus the chosen feature list. It guarantees that no
    leaky or identifier column can slip into the feature list.
    """
    df = add_derived_features(players)
    selected = select_features(df)
    # Final safety net: never return a leaky/meta column even if names change.
    forbidden = set(LEAKY_COLUMNS) | set(META_COLUMNS)
    selected = [f for f in selected if f not in forbidden]
    return df, selected
