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

    All six are elementwise, so this is O(rows) time and one extra copy of the
    frame in memory.

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

    Complexity: O(n log n) per candidate feature, from the k-nearest-neighbour
    entropy estimator; O(n x candidates) memory for the standardised copy.

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


# Columns that must never be returned as model inputs, whatever a caller asks
# for: the team-objective leaks and the identifiers.
FORBIDDEN_FEATURES: frozenset[str] = frozenset(LEAKY_COLUMNS) | frozenset(META_COLUMNS)

# Kept when the threshold is so high that nothing clears it, so a mistuned
# threshold degrades to "the best few features" instead of to an empty model.
FALLBACK_FEATURE_COUNT = 5


def select_features(
    df: pd.DataFrame,
    candidate_features: list[str] | None = None,
    target: str = "win",
    threshold: float = 0.01,
    random_state: int = 42,
) -> list[str]:
    """Select features whose mutual information with the target exceeds a floor.

    **This reads labels, so it is a fitted decision.** Hand it training rows
    only. The pipeline never calls it directly; it goes through
    :meth:`lolwin.model.GameSplit.select_features`, which has no access to the
    held-out rows.

    Whatever is asked for, no column in :data:`FORBIDDEN_FEATURES` is ever
    returned: the exclusion is enforced here, in the one function that decides
    the model's inputs, rather than in a wrapper that a caller could bypass.

    Complexity: O(n log n x candidates), dominated by the k-NN estimator behind
    ``mutual_info_classif``. Measured at 12.3 s for 299,590 rows x 17 candidates.

    Args:
        df: DataFrame already containing the candidate columns (call
            :func:`add_derived_features` first).
        candidate_features: Columns to consider; :data:`ALL_FEATURES` when
            omitted.
        target: Target column name.
        threshold: Minimum MI score to keep a feature. The default 0.01 drops
            near-useless features (e.g. wards_placed, total_minions_killed) that
            only add noise and training time.
        random_state: Seed for the MI estimator.

    Returns:
        The retained feature names, ordered by descending MI.
    """
    candidates = ALL_FEATURES if candidate_features is None else candidate_features
    ranked = mutual_information(df, candidates, target, random_state)
    selected = ranked.loc[ranked["mi_score"] >= threshold, "feature"].tolist()
    # Guard against a too-aggressive threshold wiping out all features.
    if not selected:
        selected = ranked["feature"].head(FALLBACK_FEATURE_COUNT).tolist()
    return [feature for feature in selected if feature not in FORBIDDEN_FEATURES]
