"""The dataset on disk is an input boundary and must be validated there.

Every case below was measured against the pipeline before this battery existed.
The defect each one pins down:

- a missing ``gold_earned`` raised a bare ``KeyError`` from inside
  ``add_derived_features``, three layers away from the file that caused it;
- a missing ``role`` passed cleaning silently and only broke later;
- ``win="Victory"`` gave ``invalid literal for int() with base 10``;
- a negative ``gold_earned`` became NaN through ``log1p``, emitted four
  RuntimeWarnings, and died inside sklearn's scaler with ``Input X contains
  NaN``, naming neither the column nor the file.

The invariant: anything that reaches the feature layer has every required
column, numeric and non-negative statistics, a binary label and a known role.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lolwin.data import (
    PERFORMANCE_COLUMNS,
    REQUIRED_COLUMNS,
    load_players,
    validate_players,
)
from lolwin.errors import DataFormatError

GOOD_ROW: dict[str, object] = {
    "game_id": 1,
    "player_id": 0,
    "role": "Mid",
    "win": True,
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
}


def _table(n_games: int = 1, **override: object) -> pd.DataFrame:
    """A minimal well-formed table: ``n_games`` complete 10-player games."""
    rows = []
    for game in range(n_games):
        for player in range(10):
            row = dict(GOOD_ROW)
            row["game_id"] = game
            row["player_id"] = game * 10 + player
            row["win"] = player < 5
            row.update(override)
            rows.append(row)
    return pd.DataFrame(rows)


def _written(tmp_path, frame: pd.DataFrame):
    frame.to_csv(tmp_path / "game_players_stats.csv", index=False)
    return tmp_path


# --- the reported cases ------------------------------------------------------


@pytest.mark.parametrize("column", sorted(REQUIRED_COLUMNS))
def test_a_missing_required_column_is_named_at_the_boundary(tmp_path, column):
    frame = _table().drop(columns=[column])
    with pytest.raises(DataFormatError) as exc:
        load_players(_written(tmp_path, frame))
    message = str(exc.value)
    assert column in message
    assert "game_players_stats.csv" in message


def test_a_non_boolean_label_is_rejected_with_its_value(tmp_path):
    frame = _table(win="Victory")
    with pytest.raises(DataFormatError) as exc:
        load_players(_written(tmp_path, frame))
    assert "win" in str(exc.value)


def test_a_negative_statistic_is_rejected_instead_of_becoming_nan(tmp_path):
    # log1p(-500) is NaN. A NaN must never be representable in the feature matrix.
    frame = _table(gold_earned=-500)
    with pytest.raises(DataFormatError) as exc:
        load_players(_written(tmp_path, frame))
    assert "gold_earned" in str(exc.value)


def test_a_non_numeric_statistic_is_rejected(tmp_path):
    frame = _table(gold_earned="twelve thousand")
    with pytest.raises(DataFormatError) as exc:
        load_players(_written(tmp_path, frame))
    assert "gold_earned" in str(exc.value)


# --- the invariant, stated generally ----------------------------------------


@pytest.mark.parametrize("column", sorted(PERFORMANCE_COLUMNS))
def test_no_performance_column_may_be_negative(column):
    frame = _table()
    frame.loc[frame.index[0], column] = -1
    with pytest.raises(DataFormatError, match=column):
        validate_players(frame, source="memory")


@pytest.mark.parametrize("column", sorted(PERFORMANCE_COLUMNS))
def test_no_performance_column_may_be_null(column):
    frame = _table().astype({column: "float64"})
    frame.loc[frame.index[0], column] = float("nan")
    with pytest.raises(DataFormatError, match=column):
        validate_players(frame, source="memory")


def test_a_well_formed_table_passes(tmp_path):
    players = load_players(_written(tmp_path, _table(n_games=2)))
    assert len(players) == 20
    assert set(players["win"].unique()) <= {0, 1}


# --- boundaries --------------------------------------------------------------


def test_an_empty_table_is_rejected(tmp_path):
    empty = _table().iloc[0:0]
    with pytest.raises(DataFormatError, match="empty"):
        load_players(_written(tmp_path, empty))


def test_an_unknown_role_is_rejected_with_the_accepted_set(tmp_path):
    # Training a role that no row carries yields an empty split, far from here.
    frame = _table(role="ADC")
    with pytest.raises(DataFormatError) as exc:
        load_players(_written(tmp_path, frame))
    assert "ADC" in str(exc.value)
    assert "Support" in str(exc.value)


def test_a_label_outside_zero_one_is_rejected(tmp_path):
    frame = _table()
    frame["win"] = 2
    with pytest.raises(DataFormatError, match="win"):
        load_players(_written(tmp_path, frame))


def test_an_integer_zero_one_label_is_accepted(tmp_path):
    frame = _table()
    frame["win"] = [1] * 5 + [0] * 5
    players = load_players(_written(tmp_path, frame))
    assert set(players["win"].unique()) == {0, 1}


def test_a_float_statistic_is_accepted(tmp_path):
    # Numeric is the requirement, not integral: a derived export may be float.
    frame = _table().astype({"gold_earned": "float64"})
    assert len(load_players(_written(tmp_path, frame))) == 10


def test_zero_is_accepted_everywhere(tmp_path):
    frame = _table()
    for column in PERFORMANCE_COLUMNS:
        frame[column] = 0
    assert len(load_players(_written(tmp_path, frame))) == 10


def test_a_null_identifier_is_rejected(tmp_path):
    frame = _table().astype({"player_id": "float64"})
    frame.loc[frame.index[0], "player_id"] = float("nan")
    with pytest.raises(DataFormatError, match="player_id"):
        load_players(_written(tmp_path, frame))


# --- the argument boundary, checked before the expensive read ----------------


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_sample_frac_is_rejected_before_the_file_is_read(tmp_path, bad):
    # No CSV exists in tmp_path: reaching the read would raise FileNotFoundError
    # instead. Cheap checks come first.
    with pytest.raises(ValueError, match="sample_frac"):
        load_players(tmp_path, sample_frac=bad)


# --- the neighbouring reader -------------------------------------------------


def test_a_missing_metadata_file_says_which_one(tmp_path):
    _written(tmp_path, _table())
    with pytest.raises(FileNotFoundError, match=r"game_metadata\.csv"):
        load_players(tmp_path, with_metadata=True)


def test_metadata_without_a_date_column_still_merges(tmp_path):
    pd.DataFrame({"game_id": [0], "league_name": ["TestLeague"]}).to_csv(
        tmp_path / "game_metadata.csv", index=False
    )
    _written(tmp_path, _table())
    players = load_players(tmp_path, with_metadata=True)
    assert "league_name" in players.columns
    assert "date" not in players.columns


def test_metadata_without_a_game_id_is_rejected(tmp_path):
    pd.DataFrame({"date": ["2024-01-01"]}).to_csv(
        tmp_path / "game_metadata.csv", index=False
    )
    _written(tmp_path, _table())
    with pytest.raises(DataFormatError, match="game_id"):
        load_players(tmp_path, with_metadata=True)


# --- the schema and the feature layer must not drift apart -------------------


def test_every_base_feature_is_a_validated_column():
    """features.BASE_FEATURES may only name columns the loader guarantees.

    The loader deliberately does not import the feature layer, so this is the
    test that keeps the two definitions in step. If a new base feature is added
    without extending PERFORMANCE_COLUMNS, it would reach the model unvalidated.
    """
    from lolwin.features import BASE_FEATURES

    assert set(BASE_FEATURES) <= set(PERFORMANCE_COLUMNS)
