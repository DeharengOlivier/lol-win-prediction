"""The benchmark is committed, so it is kept working like anything else.

A benchmark that silently stops matching the pipeline it measures publishes
numbers for code that no longer exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from bench_pipeline import PLAYERS_PER_GAME, as_row, measure, synthetic_games
from lolwin.data import validate_players


def test_generated_games_satisfy_the_loader_contract():
    frame = synthetic_games(20)
    validate_players(frame, source="benchmark")
    assert len(frame) == 20 * PLAYERS_PER_GAME


def test_every_generated_game_has_both_outcomes_and_all_roles():
    frame = synthetic_games(20)
    per_game = frame.groupby("game_id")
    assert (per_game["win"].nunique() == 2).all()
    assert (per_game["role"].nunique() == 5).all()


def test_the_generator_is_reproducible():
    from pandas.testing import assert_frame_equal

    assert_frame_equal(synthetic_games(10, seed=3), synthetic_games(10, seed=3))


def test_a_measurement_runs_end_to_end_and_renders():
    metrics = measure(60)
    assert metrics["rows"] == 600
    for key in ("load_s", "split_s", "select_s", "train_s", "evaluate_s"):
        assert metrics[key] >= 0.0
    assert as_row(metrics).startswith("| 60 |")


@pytest.mark.parametrize("n_games", [1, 2])
def test_the_generator_holds_at_its_smallest(n_games):
    frame = synthetic_games(n_games)
    validate_players(frame, source="benchmark")
