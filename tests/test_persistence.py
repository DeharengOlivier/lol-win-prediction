"""A tuned run takes minutes and could not be reloaded: only re-run.

The mechanism that is supposed to save you is tested in the conditions where it
is supposed to save you, so this file covers the round trip, an interrupted
write, a partially deleted artifact and a corrupted manifest, not just the
happy path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lolwin.errors import DataFormatError
from lolwin.features import add_derived_features
from lolwin.model import split_by_game, train_all_roles
from lolwin.persistence import MANIFEST_FILE, load_models, save_models


@pytest.fixture(scope="module")
def trained(synthetic_players):
    split = split_by_game(add_derived_features(synthetic_players))
    features = split.select_features()
    return split, features, train_all_roles(split, features, tune=False)


def _proba(models, split, features):
    """Positive-class probabilities per role, in a stable order."""
    return {
        role: role_model.model.predict_proba(
            split.test.loc[split.test["role"] == role, features]
        )[:, 1]
        for role, role_model in models.items()
    }


def test_a_reloaded_model_predicts_exactly_the_same(trained, tmp_path):
    split, features, models = trained
    save_models(models, tmp_path / "artifacts")
    reloaded = load_models(tmp_path / "artifacts")

    before = _proba(models, split, features)
    after = _proba(reloaded, split, features)
    assert set(before) == set(after)
    for role in before:
        np.testing.assert_array_equal(before[role], after[role])


def test_the_manifest_records_what_is_needed_to_reuse_the_models(trained, tmp_path):
    _, features, models = trained
    save_models(models, tmp_path / "artifacts")
    reloaded = load_models(tmp_path / "artifacts")
    for role, role_model in reloaded.items():
        assert role_model.role == role
        assert role_model.features == features
        assert role_model.best_params == models[role].best_params
        assert np.isnan(role_model.cv_auc) == np.isnan(models[role].cv_auc)


def test_no_artifact_is_a_pickle(trained, tmp_path):
    """Loading a pickle runs arbitrary code, and model files get downloaded."""
    _, _, models = trained
    save_models(models, tmp_path / "artifacts")
    written = sorted(p.name for p in (tmp_path / "artifacts").iterdir())
    assert written == sorted([MANIFEST_FILE, *(f"{role}.ubj" for role in models)])


# --- the failure conditions the mechanism exists for -------------------------


def test_loading_a_missing_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere"):
        load_models(tmp_path / "nowhere")


def test_loading_a_directory_without_a_manifest_says_so(tmp_path):
    empty = tmp_path / "artifacts"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILE):
        load_models(empty)


def test_a_manifest_naming_a_missing_model_file_is_refused(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    (directory / "Bot.ubj").unlink()
    with pytest.raises(DataFormatError, match="Bot"):
        load_models(directory)


def test_a_corrupted_manifest_is_refused(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    (directory / MANIFEST_FILE).write_text("{ this is not json")
    with pytest.raises(DataFormatError, match=MANIFEST_FILE):
        load_models(directory)


def test_a_manifest_of_the_wrong_shape_is_refused(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    (directory / MANIFEST_FILE).write_text(json.dumps({"models": "not a list"}))
    with pytest.raises(DataFormatError):
        load_models(directory)


def test_saving_does_not_silently_replace_an_existing_run(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_models(models, directory)


def test_saving_with_overwrite_replaces_the_previous_run(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    save_models({"Bot": models["Bot"]}, directory, overwrite=True)
    reloaded = load_models(directory)
    assert set(reloaded) == {"Bot"}


def test_a_failed_save_leaves_the_previous_run_intact(trained, tmp_path, monkeypatch):
    """An interrupted write must not destroy the artifact it was replacing."""
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    intact = (directory / MANIFEST_FILE).read_text()

    class _Unsavable:
        def save_model(self, path):
            raise OSError("disk full")

    broken = {"Bot": models["Bot"]}
    monkeypatch.setattr(broken["Bot"], "model", _Unsavable())
    with pytest.raises(OSError, match="disk full"):
        save_models(broken, directory, overwrite=True)

    assert (directory / MANIFEST_FILE).read_text() == intact
    assert set(load_models(directory)) == set(models)


def test_no_staging_directory_survives_a_failed_save(trained, tmp_path, monkeypatch):
    _, _, models = trained
    directory = tmp_path / "artifacts"

    class _Unsavable:
        def save_model(self, path):
            raise OSError("disk full")

    broken = {"Bot": models["Bot"]}
    monkeypatch.setattr(broken["Bot"], "model", _Unsavable())
    with pytest.raises(OSError):
        save_models(broken, directory)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_swap_restores_the_previous_artifact(trained, tmp_path, monkeypatch):
    """The last step can fail too: the old artifact must come back."""
    import os

    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    intact = (directory / MANIFEST_FILE).read_text()

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 1 = move the old copy aside, 2 = move the new one in
            raise OSError("rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr("lolwin.persistence.os.replace", flaky)
    with pytest.raises(OSError, match="rename failed"):
        save_models(models, directory, overwrite=True)

    assert (directory / MANIFEST_FILE).read_text() == intact
    assert set(load_models(directory)) == set(models)
    assert not any(p.name.startswith("artifacts.") for p in tmp_path.iterdir())


def test_a_manifest_entry_without_a_role_is_refused(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    (directory / MANIFEST_FILE).write_text(json.dumps({"models": [{"file": "Bot.ubj"}]}))
    with pytest.raises(DataFormatError, match="'role'"):
        load_models(directory)


def test_a_manifest_entry_that_is_not_an_object_is_refused(trained, tmp_path):
    _, _, models = trained
    directory = tmp_path / "artifacts"
    save_models(models, directory)
    (directory / MANIFEST_FILE).write_text(json.dumps({"models": ["Bot"]}))
    with pytest.raises(DataFormatError):
        load_models(directory)


def test_saving_no_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no models"):
        save_models({}, tmp_path / "artifacts")


# --- reachable from the command line ----------------------------------------


def test_the_cli_can_persist_a_run(synthetic_data_dir, tmp_path):
    from lolwin.cli import EXIT_OK, main

    directory = tmp_path / "artifacts"
    code = main(
        [
            "--data-dir",
            str(synthetic_data_dir),
            "--no-tune",
            "--save-models",
            str(directory),
        ]
    )
    assert code == EXIT_OK
    assert set(load_models(directory)) == {"Top", "Jungle", "Mid", "Bot", "Support"}


def test_the_cli_refuses_to_clobber_an_artifact(synthetic_data_dir, tmp_path, capsys):
    from lolwin.cli import EXIT_BAD_INPUT, main

    directory = tmp_path / "artifacts"
    argv = [
        "--data-dir",
        str(synthetic_data_dir),
        "--no-tune",
        "--save-models",
        str(directory),
    ]
    main(argv)
    capsys.readouterr()
    assert main(argv) == EXIT_BAD_INPUT
    assert "overwrite" in capsys.readouterr().err
    assert main([*argv, "--overwrite"]) == 0
