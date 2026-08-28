"""Command-line entry point that runs the whole pipeline end to end.

Usage (from the repository root, with the package installed)::

    lolwin                     # full run, tuned, all data
    lolwin --sample 0.2        # fast iteration on 20% of games
    lolwin --no-tune           # skip grid search (smoke test)

It reports the mutual-information ranking, the chosen features, the per-role
best parameters and the held-out results table, then optionally writes the table
to CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import RANDOM_STATE, ROLES
from .data import DEFAULT_DATA_DIR, load_players
from .evaluate import evaluate_all, format_results
from .features import LEAKY_COLUMNS, add_derived_features, mutual_information
from .model import (
    DEFAULT_TEST_SIZE,
    GameSplit,
    RoleModel,
    split_by_game,
    train_all_roles,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_INPUT = 2


@dataclass(frozen=True)
class PipelineResult:
    """Everything one run produced, so a caller can reuse it without re-running.

    Attributes:
        results: The per-role metrics table, plus the ``Overall`` row.
        models: The fitted models, keyed by role.
        features: The feature columns chosen on the training rows.
        split: The train/test partition every number above was produced under.
    """

    results: pd.DataFrame
    models: dict[str, RoleModel]
    features: list[str]
    split: GameSplit


def run_pipeline(
    data_dir: str | Path | None = None,
    sample_frac: float | None = None,
    tune: bool = True,
    output_csv: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PipelineResult:
    """Run load -> derive -> split -> select -> train -> evaluate.

    The order matters and is the point of this function: the split happens
    before feature selection, so no fitted decision ever reads a held-out label.

    Args:
        data_dir: Folder with the CSV files; the package default when omitted.
        sample_frac: Fraction of games to keep, for fast iteration.
        tune: Whether to grid-search hyperparameters.
        output_csv: Where to write the results table, if anywhere.
        test_size: Share of games held out.
        random_state: Seed.

    Returns:
        The :class:`PipelineResult` for this run.
    """
    print("Loading data ...")
    players = load_players(
        DEFAULT_DATA_DIR if data_dir is None else data_dir,
        sample_frac=sample_frac,
        random_state=random_state,
    )
    print(f"  rows: {len(players):,} | games: {players['game_id'].nunique():,}")
    print(f"  win balance: {players['win'].mean():.4f} (0.5 == perfectly balanced)")
    print(f"  Excluded leaky team-objective columns: {LEAKY_COLUMNS}")

    df = add_derived_features(players)
    split = split_by_game(df, test_size=test_size, random_state=random_state)
    print(
        f"\nHeld out {split.test['game_id'].nunique():,} games "
        f"({len(split.test):,} rows) before anything was fitted."
    )

    print("\nSelecting features on the training rows only ...")
    features = split.select_features(random_state=random_state)
    print(mutual_information(split.train, features).to_string(index=False))
    print(f"\nSelected features ({len(features)}): {features}")

    print("\nTraining per-role XGBoost (grouped CV, ROC-AUC tuning) ...")
    models = train_all_roles(
        split, features, roles=ROLES, random_state=random_state, tune=tune
    )
    for role, role_model in models.items():
        print(f"  {role:<8} best params: {role_model.best_params}")
        print(f"  {'':<8} cv ROC-AUC : {role_model.cv_auc:.4f}")

    print("\nHeld-out test results:")
    table = evaluate_all(models, split.test)
    print(format_results(table))

    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        print(f"\nResults written to {destination}")

    return PipelineResult(results=table, models=models, features=features, split=split)


def _build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="lolwin", description="LoL per-role win-prediction pipeline."
    )
    parser.add_argument("--data-dir", default=None, help="Folder with the CSV files.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of games to use for fast iteration (e.g. 0.2). Omit for all.",
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip grid search (quick smoke test)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Share of games held out (default {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--output-csv", default=None, help="Optional path to save the results table."
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, and turn failures into exit codes.

    Returns:
        0 on success, 2 when the input is unusable, 1 on any other failure.
    """
    args = _build_parser().parse_args(argv)
    run_pipeline(
        data_dir=args.data_dir,
        sample_frac=args.sample,
        tune=not args.no_tune,
        output_csv=args.output_csv,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
