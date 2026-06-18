"""Shared pytest fixtures.

The tests must run fast and offline, so they build a small *synthetic* dataset
that mimics the real schema (same columns, 10 players per game, a learnable
relationship between stats and the win label). This keeps the suite independent
of the multi-hundred-MB CSVs, which are not committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``import lolwin`` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lolwin import ROLES  # noqa: E402


def _make_synthetic_players(n_games: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic player-stats table with the real column schema.

    Winners are given (noisily) better stats than losers so a model can learn a
    non-trivial relationship. Includes the leaky team-objective columns so we
    can assert the feature builder excludes them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for game_id in range(n_games):
        # Team 0 wins half the games, team 1 the other half.
        winning_team = game_id % 2
        for team_id in (0, 1):
            won = int(team_id == winning_team)
            for role in ROLES:
                base = 8 if won else 4  # winners frag more, on average
                kills = max(0, int(rng.normal(base, 3)))
                deaths = max(0, int(rng.normal(4 if won else 7, 3)))
                assists = max(0, int(rng.normal(base + 2, 3)))
                rows.append(
                    {
                        "game_id": game_id,
                        "player_id": game_id * 10 + team_id * 5 + ROLES.index(role),
                        "player_name": f"player_{team_id}_{role}",
                        "team_id": team_id,
                        "team_name": f"team_{team_id}",
                        "team_acronym": f"T{team_id}",
                        "role": role,
                        "win": bool(won),
                        "game_length": int(rng.normal(1800, 200)),
                        "champion_name": "Champ",
                        # Leaky team-objective columns (must be excluded by features).
                        "team_kills": int(rng.normal(30 if won else 15, 5)),
                        "tower_kills": (8 if won else 2),
                        "inhibitor_kills": (2 if won else 0),
                        "dragon_kills": (3 if won else 1),
                        "herald_kills": (1 if won else 0),
                        "baron_kills": (1 if won else 0),
                        # Player-performance columns (legitimate inputs).
                        "player_kills": kills,
                        "player_deaths": deaths,
                        "player_assists": assists,
                        "total_minions_killed": max(0, int(rng.normal(180, 40))),
                        "gold_earned": max(1, int(rng.normal(13000 if won else 10000, 2000))),
                        "level": max(1, int(rng.normal(15 if won else 13, 2))),
                        "total_damage_dealt": int(rng.normal(80000, 20000)),
                        "total_damage_dealt_to_champions": int(rng.normal(20000, 5000)),
                        "total_damage_taken": int(rng.normal(25000, 6000)),
                        "wards_placed": max(0, int(rng.normal(12, 4))),
                        "largest_killing_spree": max(0, int(rng.normal(4 if won else 1, 2))),
                        "largest_multi_kill": max(0, int(rng.normal(2 if won else 1, 1))),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_players() -> pd.DataFrame:
    """Session-scoped synthetic player table (raw, win as bool)."""
    return _make_synthetic_players()


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory, synthetic_players) -> Path:
    """Write the synthetic data to a temp folder as the real CSV would look."""
    d = tmp_path_factory.mktemp("data")
    synthetic_players.to_csv(d / "game_players_stats.csv", index=False)
    # Minimal metadata so with_metadata=True works.
    meta = pd.DataFrame(
        {
            "game_id": sorted(synthetic_players["game_id"].unique()),
        }
    )
    meta["date"] = "2024-01-01"
    meta["league_name"] = "TestLeague"
    meta.to_csv(d / "game_metadata.csv", index=False)
    return d
