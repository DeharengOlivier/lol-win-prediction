"""Data loading, validation and basic cleaning.

This module is the single place that knows where the raw CSV files live, what
they must contain, and what "clean" means. Everything downstream consumes the
tidy DataFrame returned by :func:`load_players` and never re-checks it.

The dataset (not committed to the repo) is made of three CSVs:

- ``game_players_stats.csv`` : one row per (game, player). This is the table we
  model on. ~374k rows, 10 players per game, perfectly balanced ``win`` label.
- ``game_metadata.csv``      : one row per game (date, tournament, league...).
  Useful for context / time-based splitting, merged in optionally.
- ``game_events.csv``        : raw kill/objective events. Not used by the
  baseline model (kept out to avoid post-game outcome leakage), documented for
  completeness.

**The file is a boundary.** A CSV is arbitrary text until proven otherwise, so
:func:`validate_players` checks the whole contract once, at load time, and
raises :class:`~lolwin.errors.DataFormatError` naming the file and the offending
column. Past that point the frame is trusted: no defensive check appears
anywhere downstream, and an invalid value cannot reach the feature layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from . import ROLES
from .errors import DataFormatError

# Default location: a git-ignored ``data/`` folder in the working directory.
# Deliberately relative to the caller and not to ``__file__``: once the package
# is installed, ``__file__`` points inside site-packages, where no dataset lives.
DEFAULT_DATA_DIR = Path("data")

PLAYERS_FILE = "game_players_stats.csv"
METADATA_FILE = "game_metadata.csv"
EVENTS_FILE = "game_events.csv"

# How many players a complete match has. Rows belonging to a game that does not
# have exactly this many players are dropped as remakes or corrupted exports.
PLAYERS_PER_GAME = 10

# --- the schema this package requires of the player table --------------------

IDENTITY_COLUMNS: tuple[str, ...] = ("game_id", "player_id", "role")
LABEL_COLUMN = "win"

# The raw per-player statistics. These are the legitimate model inputs and the
# columns every derived feature is built from. ``features.BASE_FEATURES`` must
# stay a subset of this tuple; a test enforces that so the two cannot drift.
PERFORMANCE_COLUMNS: tuple[str, ...] = (
    "player_kills",
    "player_deaths",
    "player_assists",
    "total_minions_killed",
    "gold_earned",
    "level",
    "total_damage_dealt_to_champions",
    "total_damage_taken",
    "wards_placed",
    "largest_killing_spree",
    "largest_multi_kill",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    LABEL_COLUMN,
    *PERFORMANCE_COLUMNS,
)


def validate_players(players: pd.DataFrame, *, source: str) -> None:
    """Check the player table against the schema the pipeline requires.

    Complexity: O(rows x columns), one vectorised pass per rule.

    Args:
        players: The freshly read table, before any cleaning.
        source: What to name in the error message, normally the file path.

    Raises:
        DataFormatError: Any rule below is broken. The message names ``source``
            and the offending column.
    """
    if players.empty:
        raise DataFormatError(f"{source} is empty: no player rows to model on.")

    missing = [column for column in REQUIRED_COLUMNS if column not in players.columns]
    if missing:
        raise DataFormatError(
            f"{source} is missing required column(s): {', '.join(missing)}. "
            f"The player table must carry: {', '.join(REQUIRED_COLUMNS)}."
        )

    null_counts = players[list(REQUIRED_COLUMNS)].isna().sum()
    with_nulls = [column for column, count in null_counts.items() if count]
    if with_nulls:
        raise DataFormatError(
            f"{source} has null values in: {', '.join(str(c) for c in with_nulls)}. "
            "Every required column must be complete."
        )

    _validate_label(players, source=source)
    _validate_roles(players, source=source)
    _validate_statistics(players, source=source)


def _validate_label(players: pd.DataFrame, *, source: str) -> None:
    """The target must be binary: booleans, or integers already in {0, 1}."""
    label = players[LABEL_COLUMN]
    if is_bool_dtype(label):
        return
    if not is_numeric_dtype(label):
        sample = label.iloc[0]
        raise DataFormatError(
            f"{source} column '{LABEL_COLUMN}' must be boolean or 0/1, "
            f"found {label.dtype} (e.g. {sample!r})."
        )
    outside = sorted(set(label.unique()) - {0, 1})
    if outside:
        raise DataFormatError(
            f"{source} column '{LABEL_COLUMN}' must contain only 0 and 1, "
            f"found also: {outside}."
        )


def _validate_roles(players: pd.DataFrame, *, source: str) -> None:
    """Every role must be one this package trains a model for.

    An unrecognised role used to survive loading and produce an empty per-role
    split much later, where the failure said nothing about its cause.
    """
    unknown = sorted(set(players["role"].unique()) - set(ROLES))
    if unknown:
        raise DataFormatError(
            f"{source} contains unknown role(s): {', '.join(map(str, unknown))}. "
            f"Accepted roles: {', '.join(ROLES)}."
        )


def _validate_statistics(players: pd.DataFrame, *, source: str) -> None:
    """Statistics must be numeric and non-negative.

    Non-negativity is not cosmetic: ``log_gold`` is ``log1p(gold_earned)``, so a
    negative value silently produces NaN, which then propagates through feature
    scaling and surfaces as an sklearn error naming neither column nor file.
    """
    for column in PERFORMANCE_COLUMNS:
        values = players[column]
        if not is_numeric_dtype(values) or is_bool_dtype(values):
            sample = values.iloc[0]
            raise DataFormatError(
                f"{source} column '{column}' must be numeric, "
                f"found {values.dtype} (e.g. {sample!r})."
            )
        smallest = values.min()
        if smallest < 0:
            raise DataFormatError(
                f"{source} column '{column}' must be non-negative, "
                f"found {smallest}. A negative count makes log1p produce NaN."
            )


def load_players(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    with_metadata: bool = False,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the player-stats table, validate it, clean it, optionally merge metadata.

    Complexity: O(rows) after the read, which is O(file size).

    Args:
        data_dir: Folder containing the raw CSV files.
        with_metadata: If True, left-join the per-game metadata (date, league,
            tournament) on ``game_id``. The model does not use these columns,
            but they are handy for EDA and for a time-based split.
        sample_frac: If set (e.g. 0.2), keep a random subset of the *games* for
            fast iteration, so the ten players of a game stay together. Final
            reported metrics must use ``sample_frac=None``.
        random_state: Seed for reproducible sampling.

    Returns:
        A validated, cleaned DataFrame, one row per (game, player), with
        ``win`` cast to a 0/1 integer.

    Raises:
        ValueError: ``sample_frac`` is outside (0, 1]. Checked before the read,
            which costs seconds on the full dataset.
        FileNotFoundError: The expected CSV is missing, with a message telling
            the caller where to put the data.
        DataFormatError: The CSV does not match the schema of
            :func:`validate_players`.
    """
    if sample_frac is not None and not 0 < sample_frac <= 1:
        raise ValueError(f"sample_frac must be within (0, 1], got {sample_frac}.")

    data_dir = Path(data_dir)
    players_path = data_dir / PLAYERS_FILE
    if not players_path.exists():
        raise FileNotFoundError(
            f"Could not find {players_path}. Download the dataset and place the "
            f"CSV files in '{data_dir}' (see README.md, section 'Dataset')."
        )

    players = pd.read_csv(players_path)
    validate_players(players, source=str(players_path))
    players = _clean_players(players)

    if sample_frac is not None:
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

    Raises:
        FileNotFoundError: The metadata CSV is missing.
        DataFormatError: The table has no ``game_id`` to join on.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}.")
    metadata = pd.read_csv(metadata_path)
    if "game_id" not in metadata.columns:
        raise DataFormatError(
            f"{metadata_path} has no 'game_id' column, so it cannot be joined "
            "to the player table."
        )
    if "date" in metadata.columns:
        # errors="coerce": a few malformed dates should not crash the load. The
        # column is context only; nothing in the pipeline models on it.
        metadata["date"] = pd.to_datetime(metadata["date"], errors="coerce")
    return metadata


def _clean_players(players: pd.DataFrame) -> pd.DataFrame:
    """Apply basic, defensible cleaning to a table that already passed validation.

    Why each step:
    - Cast ``win`` (boolean in the raw file) to int so it is a clean 0/1 target.
    - Drop exact duplicate rows if any (defensive; the raw file is usually clean).
    - Drop the handful of games that do not have exactly ten players: they are
      remakes or corrupted rows and would distort the per-role counts.

    Complexity: O(rows) for the cast, O(rows x columns) for the deduplication.
    """
    players = players.copy()
    players[LABEL_COLUMN] = players[LABEL_COLUMN].astype(int)

    players = players.drop_duplicates()

    counts = players.groupby("game_id")["player_id"].transform("size")
    players = players[counts == PLAYERS_PER_GAME].reset_index(drop=True)

    return players
