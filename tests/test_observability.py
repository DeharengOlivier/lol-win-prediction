"""A library must not write to stdout, and a CLI must not raise tracebacks.

run_pipeline printed its progress directly, so importing the package into a
notebook or another program meant accepting its output on stdout with no way to
silence, redirect or level it. And main() had no exception handling at all: a
missing dataset or a malformed CSV reached the user as a traceback with exit
code 1, indistinguishable from a crash.
"""

from __future__ import annotations

import logging

import pytest

from lolwin.cli import EXIT_BAD_INPUT, EXIT_FAILED, EXIT_OK, main, run_pipeline
from lolwin.observability import LOG_LEVELS, LOGGER_NAME, configure_logging


def test_the_pipeline_writes_nothing_to_stdout(synthetic_data_dir, capsys):
    run_pipeline(data_dir=synthetic_data_dir, tune=False)
    assert capsys.readouterr().out == ""


def test_the_pipeline_reports_its_progress_to_the_logger(synthetic_data_dir, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        run_pipeline(data_dir=synthetic_data_dir, tune=False)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "rows=" in messages
    assert "held_out_games=" in messages
    assert "selected_features=" in messages


def test_the_package_logger_is_silent_by_default():
    """A library that configures logging on import steals the application's."""
    logger = logging.getLogger(LOGGER_NAME)
    assert logger.handlers, "the package must install its own NullHandler"
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert logger.level == logging.NOTSET


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_configure_logging_accepts_every_documented_level(level):
    configure_logging(level)
    assert logging.getLogger(LOGGER_NAME).level == getattr(logging, level)


def test_configure_logging_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="log level"):
        configure_logging("CHATTY")


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    before = len(logging.getLogger(LOGGER_NAME).handlers)
    configure_logging("DEBUG")
    assert len(logging.getLogger(LOGGER_NAME).handlers) == before


# --- the CLI turns failures into exit codes ---------------------------------


def test_a_successful_run_exits_zero(synthetic_data_dir, capsys):
    code = main(["--data-dir", str(synthetic_data_dir), "--no-tune"])
    assert code == EXIT_OK
    # The results table is the program's product and belongs on stdout.
    assert "| Role |" in capsys.readouterr().out


def test_a_missing_dataset_exits_two_with_a_message(tmp_path, capsys):
    code = main(["--data-dir", str(tmp_path), "--no-tune"])
    assert code == EXIT_BAD_INPUT
    assert "game_players_stats.csv" in capsys.readouterr().err


def test_a_malformed_dataset_exits_two_with_a_message(tmp_path, capsys):
    (tmp_path / "game_players_stats.csv").write_text("game_id,player_id\n1,2\n")
    code = main(["--data-dir", str(tmp_path), "--no-tune"])
    assert code == EXIT_BAD_INPUT
    assert "role" in capsys.readouterr().err


def test_a_bad_argument_exits_two_with_a_message(synthetic_data_dir, capsys):
    code = main(["--data-dir", str(synthetic_data_dir), "--sample", "3"])
    assert code == EXIT_BAD_INPUT
    assert "sample_frac" in capsys.readouterr().err


def test_an_unexpected_failure_exits_one(synthetic_data_dir, capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("the GPU melted")

    monkeypatch.setattr("lolwin.cli.train_all_roles", boom)
    code = main(["--data-dir", str(synthetic_data_dir), "--no-tune"])
    assert code == EXIT_FAILED
    assert "the GPU melted" in capsys.readouterr().err


def test_no_traceback_reaches_the_user(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "--no-tune"])
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], "WARNING"), (["-v"], "INFO"), (["-vv"], "DEBUG")],
)
def test_verbosity_flags_map_to_levels(synthetic_data_dir, flags, expected):
    main(["--data-dir", str(synthetic_data_dir), "--no-tune", *flags])
    assert logging.getLogger(LOGGER_NAME).level == getattr(logging, expected)
