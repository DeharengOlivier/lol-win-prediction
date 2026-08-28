"""Saving and reloading trained models.

A tuned run costs minutes of grid search. Without this module the only way to
get its models back was to run it again.

Two decisions worth stating:

- **Never pickle.** Unpickling executes arbitrary code, and a model artifact is
  exactly the kind of file that gets downloaded from a repository or a bucket.
  Boosters are written in XGBoost's own binary JSON format and everything else
  goes into a plain JSON manifest, so loading an artifact from a stranger can
  do no more than fail to parse.
- **The write is staged.** Models are written into a sibling directory and
  swapped into place only once every one of them succeeded, so an interrupted
  save cannot leave a half-written artifact where a usable one was.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from .errors import DataFormatError
from .model import RoleModel
from .observability import get_logger

MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1

# XGBoost's own binary JSON. Portable across versions, and not executable.
MODEL_SUFFIX = ".ubj"

logger = get_logger(__name__)


def save_models(
    models: dict[str, RoleModel],
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write every model plus a manifest into ``directory``.

    The directory is built beside its destination and moved into place at the
    end, so a failure part-way through leaves whatever was already there
    untouched.

    Complexity: O(models x trees) bytes written.

    Args:
        models: Fitted models keyed by role.
        directory: Where to write. Created if missing.
        overwrite: Required to replace an artifact that already exists.

    Returns:
        The directory that was written.

    Raises:
        ValueError: ``models`` is empty.
        FileExistsError: ``directory`` exists and ``overwrite`` is not set.
        OSError: The write failed; the previous artifact is left intact.
    """
    if not models:
        raise ValueError("Refusing to save no models.")

    destination = Path(directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists. Pass overwrite=True to replace it."
        )

    staging = destination.with_name(f"{destination.name}.saving-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        entries = []
        for role, role_model in models.items():
            filename = f"{role}{MODEL_SUFFIX}"
            role_model.model.save_model(str(staging / filename))
            entries.append(
                {
                    "role": role,
                    "file": filename,
                    "features": list(role_model.features),
                    "best_params": role_model.best_params,
                    # JSON has no NaN, and "not tuned" is worth saying explicitly.
                    "cv_auc": None
                    if math.isnan(role_model.cv_auc)
                    else role_model.cv_auc,
                }
            )
        manifest = {"manifest_version": MANIFEST_VERSION, "models": entries}
        (staging / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _swap_into_place(staging, destination)
    logger.info("saved models roles=%d path=%s", len(models), destination)
    return destination


def _swap_into_place(staging: Path, destination: Path) -> None:
    """Replace ``destination`` with ``staging``, keeping the old copy until it lands.

    Raises:
        OSError: The swap failed. ``destination`` is restored if it existed.
    """
    backup = destination.with_name(f"{destination.name}.replaced-{os.getpid()}")
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except OSError:
        if had_previous:
            os.replace(backup, destination)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if had_previous:
        shutil.rmtree(backup, ignore_errors=True)


def load_models(directory: str | Path) -> dict[str, RoleModel]:
    """Read back what :func:`save_models` wrote.

    Complexity: O(models x trees) bytes read.

    Args:
        directory: The artifact directory.

    Returns:
        The models, keyed by role, ready to predict.

    Raises:
        FileNotFoundError: The directory or its manifest is missing.
        DataFormatError: The manifest is unreadable, has the wrong shape, or
            names a model file that is not there.
    """
    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"No model directory at {source}.")

    manifest_path = source / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{source} has no {MANIFEST_FILE}, so it is not a model artifact."
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise DataFormatError(f"{manifest_path} is not valid JSON: {error}") from error

    entries = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise DataFormatError(
            f"{manifest_path} must hold a non-empty 'models' list, "
            f"found {type(entries).__name__}."
        )

    models: dict[str, RoleModel] = {}
    for entry in entries:
        models.update(_load_entry(entry, source, manifest_path))
    logger.info("loaded models roles=%d path=%s", len(models), source)
    return models


def _load_entry(entry: Any, source: Path, manifest_path: Path) -> dict[str, RoleModel]:
    """Rebuild one role from its manifest entry.

    Raises:
        DataFormatError: The entry is malformed or its model file is missing.
    """
    if not isinstance(entry, dict) or "role" not in entry or "file" not in entry:
        raise DataFormatError(
            f"{manifest_path} holds an entry without a 'role' and a 'file': {entry!r}."
        )

    role = str(entry["role"])
    model_path = source / str(entry["file"])
    if not model_path.is_file():
        raise DataFormatError(
            f"{manifest_path} names {model_path.name} for role {role}, "
            "but that file is missing from the artifact."
        )

    classifier = XGBClassifier()
    classifier.load_model(str(model_path))
    cv_auc = entry.get("cv_auc")
    return {
        role: RoleModel(
            role=role,
            model=classifier,
            features=list(entry.get("features", [])),
            best_params=dict(entry.get("best_params", {})),
            cv_auc=float("nan") if cv_auc is None else float(cv_auc),
        )
    }


__all__ = ["MANIFEST_FILE", "MODEL_SUFFIX", "load_models", "save_models"]
