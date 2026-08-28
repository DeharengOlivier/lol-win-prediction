"""Command-line entry point that runs the whole pipeline end to end.

Usage (from the repository root, with the package installed)::

    lolwin                     # full run, tuned, all data
    lolwin --sample 0.2        # fast iteration on 20% of games
    lolwin --no-tune           # skip grid search (smoke test)
    lolwin -v                  # progress on stderr

The results table is the program's product and goes to stdout. Progress goes to
the package logger, which is silent until ``-v`` asks for it. Failures become
exit codes and a one-line message, never a traceback.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import RANDOM_STATE, ROLES
from .data import DEFAULT_DATA_DIR, load_players
from .errors import DataFormatError
from .evaluate import evaluate_all, format_results
from .features import LEAKY_COLUMNS, add_derived_features
from .model import (
    DEFAULT_TEST_SIZE,
    GameSplit,
    RoleModel,
    split_by_game,
    train_all_roles,
)
from .observability import configure_logging, get_logger

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_INPUT = 2

# -v raises the level one notch, -vv two.
_VERBOSITY_LEVELS = ("WARNING", "INFO", "DEBUG")

logger = get_logger(__name__)


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

    The order is the point of this function: the split happens before feature
    selection, so no fitted decision ever reads a held-out label.

    Args:
        data_dir: Folder with the CSV files; the package default when omitted.
        sample_frac: Fraction of games to keep, for fast iteration.
        tune: Whether to grid-search hyperparameters.
        output_csv: Where to write the results table, if anywhere.
        test_size: Share of games held out.
        random_state: Seed.

    Returns:
        The :class:`PipelineResult` for this run.

    Raises:
        FileNotFoundError: The dataset is not where it was expected.
        DataFormatError: The dataset does not match the required schema.
        ValueError: An argument is outside its accepted range.
    """
    players = load_players(
        DEFAULT_DATA_DIR if data_dir is None else data_dir,
        sample_frac=sample_frac,
        random_state=random_state,
    )
    logger.info(
        "loaded rows=%d games=%d win_rate=%.4f excluded_leaky=%d",
        len(players),
        players["game_id"].nunique(),
        players["win"].mean(),
        len(LEAKY_COLUMNS),
    )

    split = split_by_game(
        add_derived_features(players), test_size=test_size, random_state=random_state
    )
    logger.info(
        "split held_out_games=%d held_out_rows=%d train_rows=%d",
        split.test["game_id"].nunique(),
        len(split.test),
        len(split.train),
    )

    features = split.select_features(random_state=random_state)
    logger.info("selected_features=%d names=%s", len(features), ",".join(features))

    models = train_all_roles(
        split, features, roles=ROLES, random_state=random_state, tune=tune
    )
    for role, role_model in models.items():
        logger.info(
            "trained role=%s cv_auc=%.4f params=%s",
            role,
            role_model.cv_auc,
            role_model.best_params,
        )

    table = evaluate_all(models, split.test)

    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        logger.info("wrote results path=%s", destination)

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Progress on stderr: -v for INFO, -vv for DEBUG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, print the table, and map failures to codes.

    Returns:
        0 on success, 2 when the input or an argument is unusable, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    configure_logging(_VERBOSITY_LEVELS[min(args.verbose, len(_VERBOSITY_LEVELS) - 1)])

    try:
        result = run_pipeline(
            data_dir=args.data_dir,
            sample_frac=args.sample,
            tune=not args.no_tune,
            output_csv=args.output_csv,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (FileNotFoundError, DataFormatError, ValueError) as error:
        print(f"lolwin: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    # The CLI is the last line of defence: nothing below it turns a crash into
    # a message, so this catch is deliberately broad.
    except Exception as error:
        print(f"lolwin: unexpected failure: {error}", file=sys.stderr)
        return EXIT_FAILED

    print(format_results(result.results))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
