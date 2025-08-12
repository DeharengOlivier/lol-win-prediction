"""Data loading, merging and basic cleaning.

This module is the single place that knows where the raw CSV files live and
what they contain. Everything downstream consumes the tidy DataFrame returned
by :func:`load_players`.

The dataset (not committed to the repo) is made of three CSVs:

- ``game_players_stats.csv`` : one row per (game, player). This is the table we
  model on. ~374k rows, 10 players per game, perfectly balanced ``win`` label.
- ``game_metadata.csv``      : one row per game (date, tournament, league...).
  Useful for context / time-based splitting, merged in optionally.
- ``game_events.csv``        : raw kill/objective events. Not used by the
  baseline model (kept out to avoid post-game outcome leakage), documented for
  completeness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default location: a git-ignored ``data/`` folder at the repo root.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PLAYERS_FILE = "game_players_stats.csv"
METADATA_FILE = "game_metadata.csv"
EVENTS_FILE = "game_events.csv"


def load_players(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    with_metadata: bool = False,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the player-stats table, clean it and optionally merge metadata.

    Args:
        data_dir: Folder containing the raw CSV files.
        with_metadata: If True, left-join the per-game metadata (date, league,
            tournament) on ``game_id``. The model does not use these columns,
            but they are handy for EDA and for a time-based split.
        sample_frac: If set (e.g. 0.2), keep a stratified-by-game random subset
            of the games for fast iteration. Sampling is done at the *game*
            level so we never split a game across the sample boundary. Final
            reported metrics must use ``sample_frac=None``.
        random_state: Seed for reproducible sampling.

    Returns:
        A cleaned DataFrame, one row per (game, player), with ``win`` cast to a
        0/1 integer.

    Raises:
        FileNotFoundError: If the expected CSV is missing (clear message telling
            the user where to put the data).
    """
    data_dir = Path(data_dir)
    players_path = data_dir / PLAYERS_FILE
    if not players_path.exists():
        raise FileNotFoundError(
            f"Could not find {players_path}. Download the dataset and place the "
            f"CSV files in '{data_dir}' (see README.md, section 'Dataset')."
        )

    players = pd.read_csv(players_path)
    players = _clean_players(players)

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")
        # Sample whole games so the 10 players of a game stay together.
        games = players["game_id"].drop_duplicates()
        keep = games.sample(frac=sample_frac, random_state=random_state)
        players = players[players["game_id"].isin(keep)].reset_index(drop=True)

    if with_metadata:
        metadata = load_metadata(data_dir)
        players = players.merge(metadata, on="game_id", how="left")

    return players


def load_metadata(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the per-game metadata table (date, tournament, league).

    The ``date`` column is parsed to datetime so it can be used for a
    chronological train/test split if desired.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}.")
    metadata = pd.read_csv(metadata_path)
    if "date" in metadata.columns:
        # errors="coerce": a few malformed dates should not crash the load.
        metadata["date"] = pd.to_datetime(metadata["date"], errors="coerce")
    return metadata


def _clean_players(players: pd.DataFrame) -> pd.DataFrame:
    """Apply basic, defensible cleaning to the player-stats table.

    Why each step:
    - Cast ``win`` (boolean in the raw file) to int so it is a clean 0/1 target.
    - Drop the handful of games that do not have the expected 10 players: they
      are likely remakes / corrupted rows and could distort per-role stats.
    - Drop exact duplicate rows if any (defensive; the raw file is usually clean).
    """
    players = players.copy()
    players["win"] = players["win"].astype(int)

    players = players.drop_duplicates()

    # Keep only complete 10-player games. ~10 games out of 37k fail this; they
    # are noise and removing them keeps the per-role row counts balanced.
    counts = players.groupby("game_id")["player_id"].transform("size")
    players = players[counts == 10].reset_index(drop=True)

    return players
