"""Command-line entry point that runs the whole pipeline end to end.

Usage (from the repo root, with the venv active)::

    python -m lolwin.cli                 # full run, tuned, all data
    python -m lolwin.cli --sample 0.2    # fast iteration on 20% of games
    python -m lolwin.cli --no-tune       # skip grid search (smoke test)

It prints the mutual-information ranking, the chosen features, per-role best
params, and the held-out results table, then optionally writes the table to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import RANDOM_STATE, ROLES
from .data import load_players
from .evaluate import evaluate_all, format_results
from .features import LEAKY_COLUMNS, build_feature_matrix, mutual_information
from .model import train_all_roles


def run_pipeline(
    data_dir: str | Path | None = None,
    sample_frac: float | None = None,
    tune: bool = True,
    output_csv: str | Path | None = None,
    random_state: int = RANDOM_STATE,
):
    """Run load -> features -> per-role train -> evaluate and return the table.

    Returns a tuple ``(results_table, models, selected_features)`` so callers
    (e.g. the notebook) can reuse the trained models.
    """
    print("Loading data ...")
    load_kwargs = {"sample_frac": sample_frac, "random_state": random_state}
    players = (
        load_players(data_dir, **load_kwargs)
        if data_dir is not None
        else load_players(**load_kwargs)
    )
    print(f"  rows: {len(players):,} | games: {players['game_id'].nunique():,}")
    print(f"  win balance: {players['win'].mean():.4f} (0.5 == perfectly balanced)")
    print(f"  Excluded leaky team-objective columns: {LEAKY_COLUMNS}")

    print("\nBuilding features + mutual-information selection ...")
    df, features = build_feature_matrix(players)
    mi = mutual_information(df, features)
    print(mi.to_string(index=False))
    print(f"\nSelected features ({len(features)}): {features}")

    print("\nTraining per-role XGBoost (grouped CV, ROC-AUC tuning) ...")
    models = train_all_roles(
        df, features, roles=ROLES, random_state=random_state, tune=tune
    )

    print("\nHeld-out test results:")
    table = evaluate_all(models)
    print(format_results(table))

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"\nResults written to {output_csv}")

    return table, models, features


def main() -> None:
    """Parse CLI args and run the pipeline."""
    parser = argparse.ArgumentParser(description="LoL per-role win-prediction pipeline.")
    parser.add_argument("--data-dir", default=None, help="Folder with the CSV files.")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of games to use for fast iteration (e.g. 0.2). Omit for full data.",
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip grid search (quick smoke test)."
    )
    parser.add_argument("--output-csv", default=None, help="Optional path to save results.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        sample_frac=args.sample,
        tune=not args.no_tune,
        output_csv=args.output_csv,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
