"""Measure how the pipeline scales with the number of games.

Rule 9 of the coding standard: a performance claim without a number is not a
claim. This script produces the numbers the README publishes.

Each size runs in its own subprocess, because ``ru_maxrss`` is a high-water
mark over the whole process lifetime: measuring several sizes in one process
would report the largest one three times.

The data is generated, not the real dataset, so anyone can reproduce the table
without downloading 42 MB. Tuning is off: the grid search is a constant
multiplier (grid size x folds fits) on top of the single-fit cost measured here,
and the full tuned run on the real data is timed separately in the README.

Usage::

    python benchmarks/bench_pipeline.py                 # the whole table
    python benchmarks/bench_pipeline.py --single 10000  # one size, in-process
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lolwin import ROLES
from lolwin.data import load_players
from lolwin.evaluate import evaluate_all
from lolwin.features import add_derived_features
from lolwin.model import split_by_game, train_all_roles

SIZES = (2_000, 10_000, 40_000)
PLAYERS_PER_GAME = 10


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    Linux reports kilobytes, macOS reports bytes.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def synthetic_games(n_games: int, seed: int = 0) -> pd.DataFrame:
    """Build ``n_games`` complete matches with a learnable win signal.

    Vectorised on purpose: a row-by-row builder would dominate the measurement
    it is supposed to set up.
    """
    rng = np.random.default_rng(seed)
    n_rows = n_games * PLAYERS_PER_GAME
    game_id = np.repeat(np.arange(n_games), PLAYERS_PER_GAME)
    team = np.tile(np.repeat([0, 1], 5), n_games)
    won = (team == (game_id % 2)).astype(int)

    def noisy(win_mean: float, loss_mean: float, spread: float) -> np.ndarray:
        centre = np.where(won == 1, win_mean, loss_mean)
        return np.maximum(0, rng.normal(centre, spread, n_rows)).astype(int)

    return pd.DataFrame(
        {
            "game_id": game_id,
            "player_id": np.arange(n_rows),
            "role": np.tile(list(ROLES), 2 * n_games),
            "win": won.astype(bool),
            "player_kills": noisy(8, 4, 3),
            "player_deaths": noisy(4, 7, 3),
            "player_assists": noisy(10, 6, 3),
            "total_minions_killed": noisy(180, 180, 40),
            "gold_earned": noisy(13000, 10000, 2000) + 1,
            "level": noisy(15, 13, 2) + 1,
            "total_damage_dealt_to_champions": noisy(20000, 20000, 5000),
            "total_damage_taken": noisy(25000, 25000, 6000),
            "wards_placed": noisy(12, 12, 4),
            "largest_killing_spree": noisy(4, 1, 2),
            "largest_multi_kill": noisy(2, 1, 1),
        }
    )


def measure(n_games: int) -> dict[str, float]:
    """Time one full untuned pipeline over ``n_games`` generated matches."""
    baseline = peak_rss_mb()
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        synthetic_games(n_games).to_csv(data_dir / "game_players_stats.csv", index=False)
        csv_mb = (data_dir / "game_players_stats.csv").stat().st_size / (1024 * 1024)

        start = time.perf_counter()
        players = load_players(data_dir)
        load_s = time.perf_counter() - start

        start = time.perf_counter()
        split = split_by_game(add_derived_features(players))
        split_s = time.perf_counter() - start

        start = time.perf_counter()
        features = split.select_features()
        select_s = time.perf_counter() - start

        start = time.perf_counter()
        models = train_all_roles(split, features, tune=False)
        train_s = time.perf_counter() - start

        start = time.perf_counter()
        evaluate_all(models, split.test)
        evaluate_s = time.perf_counter() - start

    return {
        "games": n_games,
        "rows": n_games * PLAYERS_PER_GAME,
        "csv_mb": csv_mb,
        "load_s": load_s,
        "split_s": split_s,
        "select_s": select_s,
        "train_s": train_s,
        "evaluate_s": evaluate_s,
        "peak_mb": peak_rss_mb() - baseline,
    }


HEADER = (
    "| Games | Rows | CSV | Load+validate | Split+derive | Select | "
    "Train (5 roles) | Evaluate | Peak RSS |"
)
SEPARATOR = "|---|---|---|---|---|---|---|---|---|"


def as_row(m: dict[str, float]) -> str:
    """Render one measurement as a Markdown table row."""
    return (
        f"| {int(m['games']):,} | {int(m['rows']):,} | {m['csv_mb']:.0f} MB | "
        f"{m['load_s']:.2f} s | {m['split_s']:.2f} s | {m['select_s']:.2f} s | "
        f"{m['train_s']:.1f} s | {m['evaluate_s']:.2f} s | {m['peak_mb']:.0f} MB |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single",
        type=int,
        default=None,
        help="Measure one size in this process (used by the parent run).",
    )
    args = parser.parse_args()

    if args.single is not None:
        print(as_row(measure(args.single)))
        return 0

    print(HEADER)
    print(SEPARATOR)
    for size in SIZES:
        # One process per size: ru_maxrss is a lifetime high-water mark.
        result = subprocess.run(
            [sys.executable, __file__, "--single", str(size)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
