"""Tests for the data loading and cleaning layer."""

from __future__ import annotations

import pandas as pd
import pytest

from lolwin.data import load_players


def test_load_players_basic(synthetic_data_dir):
    df = load_players(synthetic_data_dir)
    assert len(df) > 0
    # win must be cleaned to an integer 0/1 target.
    assert set(df["win"].unique()) <= {0, 1}
    assert df["win"].dtype.kind in "iu"


def test_load_players_keeps_only_full_games(synthetic_data_dir):
    df = load_players(synthetic_data_dir)
    sizes = df.groupby("game_id")["player_id"].size()
    assert (sizes == 10).all()


def test_load_players_sampling_is_by_game(synthetic_data_dir):
    df = load_players(synthetic_data_dir, sample_frac=0.25, random_state=1)
    # Sampling keeps whole games => row count is a multiple of 10.
    assert len(df) % 10 == 0
    sizes = df.groupby("game_id")["player_id"].size()
    assert (sizes == 10).all()


def test_load_players_with_metadata(synthetic_data_dir):
    df = load_players(synthetic_data_dir, with_metadata=True)
    assert "league_name" in df.columns


def test_load_players_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_players(tmp_path)


def test_sample_frac_validation(synthetic_data_dir):
    with pytest.raises(ValueError):
        load_players(synthetic_data_dir, sample_frac=1.5)
