"""End-to-end CLI / pipeline tests on a tiny synthetic CSV (no big dataset)."""

from __future__ import annotations

import pandas as pd
import pytest

from lolwin.cli import run_pipeline
from lolwin.data import load_players


def test_run_pipeline_on_synthetic_dir(synthetic_data_dir):
    # tune=False keeps it fast; the pipeline still exercises load -> features ->
    # per-role train -> evaluate on the synthetic CSV.
    table, models, features = run_pipeline(
        data_dir=synthetic_data_dir, tune=False
    )
    assert set(models) == {"Top", "Jungle", "Mid", "Bot", "Support"}
    assert len(table) == 6  # 5 roles + Overall
    assert features, "pipeline must select at least one feature"
    # No leaky column slipped through the full pipeline.
    assert "tower_kills" not in features


def test_run_pipeline_writes_csv(synthetic_data_dir, tmp_path):
    out = tmp_path / "nested" / "results.csv"
    table, _, _ = run_pipeline(
        data_dir=synthetic_data_dir, tune=False, output_csv=out
    )
    assert out.exists()
    written = pd.read_csv(out)
    # Round-trips: same number of rows as the in-memory table.
    assert len(written) == len(table)
    assert "roc_auc" in written.columns


def test_run_pipeline_with_sample_frac(synthetic_data_dir):
    table, _, _ = run_pipeline(
        data_dir=synthetic_data_dir, sample_frac=0.5, tune=False
    )
    assert len(table) == 6


def test_load_players_roundtrips_synthetic_csv(synthetic_data_dir):
    df = load_players(synthetic_data_dir)
    # The legitimate player columns survive loading.
    assert "player_kills" in df.columns
    assert "gold_earned" in df.columns
    # win cleaned to int.
    assert df["win"].dtype.kind in "iu"


def test_load_players_drops_incomplete_game(tmp_path):
    """A game with fewer than 10 players must be dropped by cleaning."""
    rows = []
    # One complete 10-player game.
    for p in range(10):
        rows.append(_player_row(game_id=1, player_id=p))
    # One broken 7-player game.
    for p in range(7):
        rows.append(_player_row(game_id=2, player_id=100 + p))
    pd.DataFrame(rows).to_csv(tmp_path / "game_players_stats.csv", index=False)

    df = load_players(tmp_path)
    assert set(df["game_id"].unique()) == {1}
    assert len(df) == 10


def _player_row(game_id: int, player_id: int) -> dict:
    return {
        "game_id": game_id,
        "player_id": player_id,
        "role": "Mid",
        "win": player_id % 2 == 0,
        "player_kills": 5,
        "player_deaths": 4,
        "player_assists": 6,
        "total_minions_killed": 180,
        "gold_earned": 12000,
        "level": 14,
        "total_damage_dealt_to_champions": 20000,
        "total_damage_taken": 25000,
        "wards_placed": 12,
        "largest_killing_spree": 2,
        "largest_multi_kill": 1,
        "team_kills": 20,
        "tower_kills": 5,
        "inhibitor_kills": 1,
        "dragon_kills": 2,
        "herald_kills": 1,
        "baron_kills": 0,
    }


def test_load_players_drops_exact_duplicate_rows(tmp_path):
    rows = [_player_row(game_id=1, player_id=p) for p in range(10)]
    # Duplicate the whole game once; dedup should bring it back to 10 rows.
    pd.DataFrame(rows + rows).to_csv(tmp_path / "game_players_stats.csv", index=False)
    df = load_players(tmp_path)
    assert len(df) == 10
