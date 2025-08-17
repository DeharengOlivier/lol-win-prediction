"""Per-role XGBoost training with leakage-free hyperparameter tuning.

This module fixes the methodological leakage of the original notebook, where
hyperparameters were chosen by looking at the *test set* and early stopping used
the test set as its eval set. Here:

- Each role gets its own train / test split, grouped by ``game_id`` so the two
  opposing players of the same game never straddle the split.
- Hyperparameters are chosen by stratified, grouped K-fold cross-validation on
  the *training* data only, scored by ROC-AUC. The test set is touched exactly
  once, at the very end, for reporting.
- All randomness is seeded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from xgboost import XGBClassifier

from . import RANDOM_STATE, ROLES

# Modest grid. Kept small on purpose: with ~75k rows per role the gains from a
# huge grid are marginal and the runtime cost is real. These ranges cover the
# usual sweet spot for tabular XGBoost (depth, learning rate, trees, and a touch
# of regularisation via subsample / colsample).
DEFAULT_PARAM_GRID: dict[str, list] = {
    "n_estimators": [300, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


@dataclass
class RoleModel:
    """A trained model for one role plus everything needed to reuse it."""

    role: str
    model: XGBClassifier
    features: list[str]
    best_params: dict
    cv_auc: float
    # Test indices kept so :mod:`evaluate` can score on the exact held-out rows.
    X_test: pd.DataFrame = field(repr=False)
    y_test: pd.Series = field(repr=False)


def _grid_combinations(param_grid: dict[str, list]) -> list[dict]:
    """Expand a param grid dict into a list of concrete param dicts."""
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _make_classifier(params: dict, random_state: int) -> XGBClassifier:
    """Build an XGBClassifier with consistent, fixed base settings."""
    return XGBClassifier(
        **params,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def tune_role(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    param_grid: dict[str, list] = DEFAULT_PARAM_GRID,
    n_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[dict, float]:
    """Grid-search hyperparameters using grouped, stratified CV on TRAIN only.

    The scoring metric is ROC-AUC (not accuracy), because it is threshold-free
    and a better summary of ranking quality on a balanced binary target.

    Args:
        X, y: Training features and target for a single role.
        groups: ``game_id`` per row, so CV folds never split a game.
        param_grid: Hyperparameter search space.
        n_splits: Number of CV folds.
        random_state: Seed.

    Returns:
        ``(best_params, best_cv_auc)``.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params: dict = {}
    best_auc = -np.inf
    for params in _grid_combinations(param_grid):
        model = _make_classifier(params, random_state)
        # cross_val_score handles fit/predict per fold; groups prevents leakage.
        scores = cross_val_score(
            model, X, y, groups=groups, cv=cv, scoring="roc_auc", n_jobs=1
        )
        mean_auc = float(scores.mean())
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
    return best_params, best_auc


def train_role(
    df: pd.DataFrame,
    role: str,
    features: list[str],
    param_grid: dict[str, list] = DEFAULT_PARAM_GRID,
    target: str = "win",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    tune: bool = True,
) -> RoleModel:
    """Train one role's model end to end: split, tune on train, fit, hold out test.

    Args:
        df: Feature DataFrame (must contain ``features``, ``role``, ``game_id``).
        role: Which role to train ("Top", "Jungle", ...).
        features: Feature column names to use.
        param_grid: Search space (ignored if ``tune`` is False).
        target: Target column.
        test_size: Fraction of *games* held out for the final test set.
        random_state: Seed.
        tune: If False, skip the grid search and use a sensible default config
            (useful for quick smoke tests).

    Returns:
        A :class:`RoleModel` carrying the fitted model and its held-out test set.
    """
    df_role = df[df["role"] == role].reset_index(drop=True)
    X = df_role[features]
    y = df_role[target]
    groups = df_role["game_id"]

    # Grouped split: opposing players from the same game stay on the same side
    # of the train/test boundary. Measured impact is small (<0.002 AUC) but it
    # removes any doubt about same-game leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    if tune:
        best_params, cv_auc = tune_role(
            X_train, y_train, groups_train, param_grid, random_state=random_state
        )
    else:
        best_params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        cv_auc = float("nan")

    model = _make_classifier(best_params, random_state)
    model.fit(X_train, y_train)

    return RoleModel(
        role=role,
        model=model,
        features=features,
        best_params=best_params,
        cv_auc=cv_auc,
        X_test=X_test,
        y_test=y_test,
    )


def train_all_roles(
    df: pd.DataFrame,
    features: list[str],
    roles: tuple[str, ...] = ROLES,
    param_grid: dict[str, list] = DEFAULT_PARAM_GRID,
    random_state: int = RANDOM_STATE,
    tune: bool = True,
    verbose: bool = True,
) -> dict[str, RoleModel]:
    """Train one :class:`RoleModel` per role and return them keyed by role."""
    models: dict[str, RoleModel] = {}
    for role in roles:
        if verbose:
            print(f"Training role: {role}")
        models[role] = train_role(
            df, role, features, param_grid, random_state=random_state, tune=tune
        )
        if verbose:
            rm = models[role]
            print(f"  best params: {rm.best_params}")
            print(f"  cv ROC-AUC : {rm.cv_auc:.4f}")
    return models
