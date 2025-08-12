"""lolwin: predict League of Legends match win/loss from a player's in-game stats.

The package is split into small, single-responsibility modules:

- ``data``      : load the raw CSVs, merge metadata, basic cleaning.
- ``features``  : feature engineering + mutual-information feature selection.
- ``model``     : per-role XGBoost training with leakage-free CV and tuning.
- ``evaluate``  : accuracy, ROC-AUC, F1, confusion matrix.
- ``cli``       : command-line entry point that runs the full pipeline.

A design note that drives the whole package: the goal is to predict the
outcome from a *single player's own performance*, so we deliberately never
feed team-level objective columns (tower_kills, inhibitor_kills, baron_kills,
...) into the model. Those columns are shared by all five players of a team
and are almost a direct encoding of the result (tower_kills alone reaches
~0.99 ROC-AUC), which would be data leakage. See ``features.LEAKY_COLUMNS``.
"""

from __future__ import annotations

ROLES: tuple[str, ...] = ("Top", "Jungle", "Mid", "Bot", "Support")
RANDOM_STATE: int = 42

__all__ = ["ROLES", "RANDOM_STATE"]
