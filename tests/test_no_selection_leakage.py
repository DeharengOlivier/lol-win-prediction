"""The train/test boundary must hold for *every* fitted decision, not just the model.

The pipeline used to call the mutual-information selector on the whole frame and
split afterwards, so the feature set was chosen while reading the labels of the
rows that later became the test set. Measured on the real dataset (seed 42,
test_size 0.2): the same fifteen features came out either way and the held-out
AUC moved by 5e-5, below seed noise. The cost was negligible; the claim
"leakage-free pipeline" was still false, and this file is what makes it true.

The invariant: nothing fitted on the training side may ever have read a test row.
"""

from __future__ import annotations

import pytest

from lolwin import ROLES
from lolwin import model as model_module
from lolwin.cli import run_pipeline
from lolwin.features import add_derived_features
from lolwin.model import DEFAULT_TEST_SIZE, split_by_game


def test_selection_never_sees_a_test_row(synthetic_data_dir, monkeypatch):
    """End to end: the frame handed to the selector holds no held-out game."""
    seen: dict[str, object] = {}
    real = model_module.select_features

    def spy(frame, *args, **kwargs):
        seen["frame"] = frame
        return real(frame, *args, **kwargs)

    monkeypatch.setattr(model_module, "select_features", spy)
    result = run_pipeline(data_dir=synthetic_data_dir, tune=False)

    assert "frame" in seen, "the pipeline did not run feature selection at all"
    selection_games = set(seen["frame"]["game_id"])  # type: ignore[index]
    test_games = set(result.split.test["game_id"])
    assert selection_games, "the selector was handed an empty frame"
    assert selection_games.isdisjoint(test_games)


def test_a_role_model_does_not_carry_its_test_set(synthetic_players):
    """A training artifact holding evaluation data is what invites leakage.

    RoleModel used to own X_test and y_test. Evaluation now takes the held-out
    frame explicitly, so a caller cannot score on data the model brought along.
    """
    split = split_by_game(add_derived_features(synthetic_players))
    features = split.select_features()
    role_model = model_module.train_role(split, "Mid", features, tune=False)
    assert not hasattr(role_model, "X_test")
    assert not hasattr(role_model, "y_test")


def test_train_and_test_share_no_game(synthetic_players):
    split = split_by_game(add_derived_features(synthetic_players))
    assert set(split.train["game_id"]).isdisjoint(set(split.test["game_id"]))
    assert len(split.train) + len(split.test) == len(synthetic_players)


def test_every_role_is_held_out_on_the_same_games(synthetic_players):
    """One split for all roles, so the Overall row averages one test set.

    The previous code split per role. With the same seed and the same game list
    the partitions happened to coincide, which made the Overall row correct by
    accident rather than by construction.
    """
    split = split_by_game(add_derived_features(synthetic_players))
    per_role = {
        role: set(split.test[split.test["role"] == role]["game_id"]) for role in ROLES
    }
    reference = per_role[ROLES[0]]
    assert all(games == reference for games in per_role.values())


def test_split_holds_out_roughly_the_requested_share(synthetic_players):
    df = add_derived_features(synthetic_players)
    split = split_by_game(df, test_size=0.3)
    share = len(split.test) / len(df)
    assert 0.25 <= share <= 0.35


def test_split_is_reproducible_for_a_given_seed(synthetic_players):
    df = add_derived_features(synthetic_players)
    first = split_by_game(df, random_state=7)
    second = split_by_game(df, random_state=7)
    assert set(first.test["game_id"]) == set(second.test["game_id"])


def test_a_different_seed_gives_a_different_split(synthetic_players):
    df = add_derived_features(synthetic_players)
    assert set(split_by_game(df, random_state=1).test["game_id"]) != set(
        split_by_game(df, random_state=2).test["game_id"]
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_unusable_test_size_is_rejected(synthetic_players, bad):
    df = add_derived_features(synthetic_players)
    with pytest.raises(ValueError, match="test_size"):
        split_by_game(df, test_size=bad)


def test_selection_from_the_split_matches_selection_on_its_training_rows(
    synthetic_players,
):
    """GameSplit.select_features is exactly select_features(split.train)."""
    from lolwin.features import select_features

    split = split_by_game(add_derived_features(synthetic_players))
    assert split.select_features() == select_features(split.train)


def test_selection_still_refuses_leaky_and_identifier_columns(synthetic_players):
    from lolwin.features import LEAKY_COLUMNS, META_COLUMNS

    split = split_by_game(add_derived_features(synthetic_players))
    features = split.select_features()
    assert set(LEAKY_COLUMNS).isdisjoint(features)
    assert set(META_COLUMNS).isdisjoint(features)


def test_tuning_only_ever_sees_training_rows(synthetic_players, monkeypatch):
    """The neighbouring fitted decision: hyperparameters, same invariant."""
    split = split_by_game(add_derived_features(synthetic_players))
    features = split.select_features()
    seen: dict[str, object] = {}
    real = model_module.tune_role

    def spy(X, y, groups, *args, **kwargs):
        seen["groups"] = set(groups)
        return real(X, y, groups, *args, **kwargs)

    monkeypatch.setattr(model_module, "tune_role", spy)
    model_module.train_role(
        split,
        "Bot",
        features,
        param_grid={
            "n_estimators": [10],
            "learning_rate": [0.1],
            "max_depth": [3],
            "subsample": [1.0],
            "colsample_bytree": [1.0],
        },
        tune=True,
    )
    assert seen["groups"].isdisjoint(set(split.test["game_id"]))  # type: ignore[union-attr]


def test_default_test_size_is_one_fifth():
    assert pytest.approx(0.2) == DEFAULT_TEST_SIZE
