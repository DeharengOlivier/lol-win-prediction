"""The train/test protocol and per-role XGBoost training.

Everything fitted lives on the training side of one boundary, and the boundary
is cut once, along whole games:

- :func:`split_by_game` partitions the rows by ``game_id`` before anything is
  fitted, so the two opposing players of a match never straddle the split and
  every role is held out on the same games.
- :meth:`GameSplit.select_features` is the only selection entry point the
  pipeline uses. It can physically only see ``split.train``, which is what makes
  "the selector never reads a test label" a property of the design rather than
  a habit. (The pipeline used to select on the whole frame.)
- Hyperparameters are chosen by stratified, grouped K-fold cross-validation on
  the training rows only, scored by ROC-AUC.
- A :class:`RoleModel` deliberately does not carry its test set. Evaluation is
  handed the held-out frame explicitly, so a caller cannot score a model on data
  the model brought along with it.
- All randomness is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

from . import RANDOM_STATE, ROLES
from .features import select_features

# Share of the *games* held out for the final, single evaluation.
DEFAULT_TEST_SIZE = 0.2

# Modest grid. Kept small on purpose: with ~60k training rows per role the gains
# from a huge grid are marginal and the runtime cost is real. These ranges cover
# the usual sweet spot for tabular XGBoost (depth, learning rate, trees, and a
# touch of regularisation via subsample / colsample).
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [300, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# Used when tuning is switched off, for smoke tests and fast iteration.
UNTUNED_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@dataclass(frozen=True)
class GameSplit:
    """A train/test partition of the player rows, cut along whole games.

    Attributes:
        train: Rows every fitted decision is allowed to read.
        test: Rows touched exactly once, at reporting time.
    """

    train: pd.DataFrame
    test: pd.DataFrame

    def select_features(self, **kwargs: Any) -> list[str]:
        """Choose the model's features, reading the training rows only.

        This is the pipeline's selection entry point precisely because it has no
        way to reach :attr:`test`.

        Args:
            **kwargs: Forwarded to :func:`lolwin.features.select_features`
                (``candidate_features``, ``threshold``, ``random_state``).

        Returns:
            The retained feature names, ordered by descending mutual information.
        """
        return select_features(self.train, **kwargs)


def split_by_game(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> GameSplit:
    """Partition rows into train and test by holding out whole games.

    One split for all five roles, so the per-role numbers and the overall row
    describe the same held-out matches. Splitting per role, as this package used
    to, produced the same partition here only because every role saw the same
    game list under the same seed: correct by accident, not by construction.

    Complexity: O(rows), plus O(g log g) to order the g distinct games.

    Args:
        df: Player rows carrying a ``game_id`` column.
        test_size: Share of the *games* to hold out, within (0, 1).
        random_state: Seed.

    Returns:
        The partition.

    Raises:
        ValueError: ``test_size`` is outside (0, 1), which would leave one side
            of the split empty.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be within (0, 1), got {test_size}.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["game_id"]))
    return GameSplit(
        train=df.iloc[train_idx].reset_index(drop=True),
        test=df.iloc[test_idx].reset_index(drop=True),
    )


@dataclass
class RoleModel:
    """A trained model for one role plus what is needed to reuse and report it.

    Attributes:
        role: Which role this model was fitted for.
        model: The fitted classifier.
        features: The feature columns it expects, in order.
        best_params: The hyperparameters it was fitted with.
        cv_auc: Mean cross-validated ROC-AUC on the training rows, or NaN when
            tuning was skipped.
    """

    role: str
    model: XGBClassifier
    features: list[str]
    best_params: dict[str, Any]
    cv_auc: float


def _grid_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a param grid dict into a list of concrete param dicts.

    Complexity: O(prod(len(v) for v in param_grid.values())), the grid size.
    """
    keys, values = zip(*param_grid.items(), strict=True)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def _make_classifier(params: dict[str, Any], random_state: int) -> XGBClassifier:
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
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], float]:
    """Grid-search hyperparameters using grouped, stratified CV on training rows.

    The scoring metric is ROC-AUC rather than accuracy, because it is
    threshold-free and a better summary of ranking quality on a balanced target.

    Complexity: ``grid size x n_splits`` model fits.

    Args:
        X, y: Training features and target for a single role.
        groups: ``game_id`` per row, so CV folds never split a game.
        param_grid: Search space; :data:`DEFAULT_PARAM_GRID` when omitted.
        n_splits: Number of CV folds.
        random_state: Seed.

    Returns:
        ``(best_params, best_cv_auc)``.
    """
    grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params: dict[str, Any] = {}
    best_auc = -np.inf
    for params in _grid_combinations(grid):
        model = _make_classifier(params, random_state)
        # cross_val_score handles fit/predict per fold; groups prevents leakage.
        scores = cross_val_score(
            model, X, y, groups=groups, cv=cv, scoring="roc_auc", n_jobs=1
        )
        mean_auc = float(scores.mean())
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
    return best_params, float(best_auc)


def train_role(
    split: GameSplit,
    role: str,
    features: list[str],
    param_grid: dict[str, list[Any]] | None = None,
    target: str = "win",
    random_state: int = RANDOM_STATE,
    tune: bool = True,
) -> RoleModel:
    """Tune and fit one role's model on the training side of ``split``.

    Complexity: one fit, plus ``grid size x n_splits`` fits when ``tune`` is set.

    Args:
        split: The partition produced by :func:`split_by_game`. Only
            :attr:`GameSplit.train` is read.
        role: Which role to train ("Top", "Jungle", ...).
        features: Feature column names to use.
        param_grid: Search space; ignored when ``tune`` is False.
        target: Target column.
        random_state: Seed.
        tune: If False, skip the grid search and use :data:`UNTUNED_PARAMS`.

    Returns:
        The fitted :class:`RoleModel`.

    Raises:
        ValueError: No training row carries this role.
    """
    train = split.train[split.train["role"] == role]
    if train.empty:
        raise ValueError(f"No training row for role {role!r}.")

    X_train = train[features]
    y_train = train[target]

    if tune:
        best_params, cv_auc = tune_role(
            X_train,
            y_train,
            train["game_id"],
            param_grid,
            random_state=random_state,
        )
    else:
        best_params, cv_auc = dict(UNTUNED_PARAMS), float("nan")

    model = _make_classifier(best_params, random_state)
    model.fit(X_train, y_train)

    return RoleModel(
        role=role,
        model=model,
        features=list(features),
        best_params=best_params,
        cv_auc=cv_auc,
    )


def train_all_roles(
    split: GameSplit,
    features: list[str],
    roles: tuple[str, ...] = ROLES,
    param_grid: dict[str, list[Any]] | None = None,
    random_state: int = RANDOM_STATE,
    tune: bool = True,
) -> dict[str, RoleModel]:
    """Train one :class:`RoleModel` per role and return them keyed by role."""
    models: dict[str, RoleModel] = {}
    for role in roles:
        models[role] = train_role(
            split,
            role,
            features,
            param_grid,
            random_state=random_state,
            tune=tune,
        )
    return models
