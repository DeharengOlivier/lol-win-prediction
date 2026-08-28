# League of Legends: per-role match outcome prediction

Predict whether a player's team won a match from that single player's
end-of-game statistics, and surface which quantifiable factors matter most. A
separate, tuned **XGBoost** model is trained for each role (Top, Jungle, Mid,
Bot, Support).

This started as one exploratory notebook. It is now an installable, tested
Python package whose train/test boundary holds for every fitted decision, not
just for the model.

```bash
pip install -e ".[dev]"
lolwin --sample 0.2 --no-tune -v     # a fast run, progress on stderr
```

## Goals

1. Predict the binary outcome (win / loss) of a match from one player's stats.
2. Identify the most actionable variables a player should focus on.

## What makes this honest

**Team-level objective columns are leakage, and they are excluded.** The
dataset ships with `tower_kills`, `inhibitor_kills`, `team_kills`,
`baron_kills`, `dragon_kills` and `herald_kills`. They are shared by all five
players of a team and almost define the result: `tower_kills` alone has a
univariate ROC-AUC of **0.99** against the win label. Using them answers the
wrong question. The exclusion lives in `features.FORBIDDEN_FEATURES`, is applied
inside the one function that decides the model's inputs, and is enforced by
tests that try to smuggle a leaky column in as a candidate.

**Nothing fitted ever reads a held-out label.** The games are split first, once,
and every fitted decision afterwards sees only the training side:

| Decision | Fitted on |
|---|---|
| Which features to use (mutual information) | training rows |
| Which hyperparameters to use (grouped CV) | training rows |
| The model itself | training rows |
| Reported metrics | the held-out rows, touched once |

That ordering is enforced by the design, not by discipline: selection is
reachable through `GameSplit.select_features`, which has no access to
`split.test`.

**This was not always true, and the fix is reported as measured.** Until
commit `f43c9d3`, feature selection ran on the whole dataset before the split,
so it read the labels of the rows it was later scored against. Measured on the
full data before the fix:

| | selection on all rows | selection on training rows |
|---|---|---|
| features chosen | the same 15 | the same 15, different order |
| Top ROC-AUC | 0.953079 | 0.953027 |
| Bot ROC-AUC | 0.964711 | 0.964726 |
| Overall accuracy (tuned) | 0.8893 | 0.8895 |
| Overall ROC-AUC (tuned) | 0.9599 | 0.9600 |

The published numbers moved by at most 0.0002. The defect was real and the cost
of it was below seed noise: both halves of that sentence belong in the same
paragraph.

## Architecture

```
lol-win-prediction/
├── src/lolwin/
│   ├── __init__.py        # ROLES, RANDOM_STATE
│   ├── errors.py          # DataFormatError
│   ├── data.py            # the input boundary: load, validate, clean
│   ├── features.py        # derived features, MI selection, the exclusion list
│   ├── model.py           # split_by_game, GameSplit, per-role XGBoost, tuning
│   ├── evaluate.py        # accuracy, ROC-AUC, F1, confusion matrix
│   ├── persistence.py     # save and reload a trained run (never pickle)
│   ├── observability.py   # the package logger
│   └── cli.py             # `lolwin`, exit codes, the results table
├── tests/                 # 167 tests, generated data, no CSV needed
├── benchmarks/            # the script that produced the scaling table below
├── scripts/               # the notebook API guard CI runs
└── lol-win-prediction.ipynb
```

- **Each module has one job.** `data` is the only module that touches the disk
  for input; the rest are pure over DataFrames.
- **The file is a boundary.** `data.validate_players` checks the whole schema
  once and raises `DataFormatError` naming the file and the column. Past that
  line the frame is trusted, and no defensive check appears downstream.
- **The library never prints.** Progress goes to the `lolwin` logger, silent
  until `-v`. The results table is the program's product and goes to stdout.
- The notebook is a thin narrative over the package, so the logic exists once.

## Dataset

Public League of Legends esports match dataset, three CSV files:
`game_players_stats.csv` (~42 MB, the table modelled on), `game_metadata.csv`,
`game_events.csv`. They are **not** committed.

1. Download the archive (the Kaggle "Ecamania Esports" LoL dataset).
2. Extract the three CSVs into `data/` at the repository root.

The loader raises a message pointing here if they are missing, and refuses a
file that does not match the schema rather than failing three layers later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # add ",notebook" for the notebook stack
# macOS only, if XGBoost reports a libomp error:
brew install libomp
```

`requirements.txt` is a pinned lock of the exact set the published numbers were
produced on; `pyproject.toml` declares the supported floors. `tensorflow`/
`keras` is deliberately absent: the original notebook benchmarked a small Keras
net, XGBoost wins on this tabular task, and nothing here imports it.

## Run

```bash
lolwin                               # full data, grid-searched
lolwin --sample 0.2                  # 20% of games, fast iteration
lolwin --no-tune                     # skip the grid search (smoke test)
lolwin --output-csv results/metrics.csv
lolwin --save-models artifacts/      # persist a tuned run
lolwin -v                            # progress on stderr; -vv for DEBUG
python train.py                      # the same thing, older entry point
```

Exit codes: `0` success, `2` unusable input or argument, `1` anything else. No
traceback reaches the user.

Reload a saved run without retraining:

```python
from lolwin.persistence import load_models

models = load_models("artifacts")
proba = models["Bot"].model.predict_proba(rows[models["Bot"].features])[:, 1]
```

Tests, on generated data, offline, no dataset needed:

```bash
pytest -q
pytest --cov                         # 99.6% with branch coverage
```

## Results (measured, full dataset)

Per-role metrics on the held-out test set, hyperparameters chosen by grouped
stratified cross-validation on the training rows only, split by `game_id`, seed
fixed. 374,490 rows / 37,449 games, win rate 0.5000.

| Role | n_test | Accuracy | ROC-AUC | F1 |
|---|---|---|---|---|
| Top | 14,980 | 0.8797 | 0.9541 | 0.8799 |
| Jungle | 14,980 | 0.8950 | 0.9630 | 0.8954 |
| Mid | 14,980 | 0.8858 | 0.9586 | 0.8864 |
| Bot | 14,980 | 0.8987 | 0.9656 | 0.8988 |
| Support | 14,980 | 0.8882 | 0.9585 | 0.8894 |
| **Overall** | 74,900 | **0.8895** | **0.9600** | **0.8900** |

Reproduce with `lolwin -v --output-csv results/metrics_full.csv`; the run this
table came from is committed as `results/metrics_full.csv` and
`results/run_full.log`.

### Before and after the engineering pass

| Aspect | Original notebook | This version |
|---|---|---|
| Headline accuracy | ~0.87-0.90 per role | ~0.88-0.90 per role, honestly measured |
| Feature selection | on the whole dataset | on the training rows only |
| Hyperparameter selection | on the **test set** | grouped CV on train only |
| Early stopping eval set | the **test set** | removed |
| Train/test split | one random split | one grouped split by `game_id`, shared by all roles |
| Metrics | accuracy | accuracy + ROC-AUC + F1 + confusion matrix |
| Malformed input | `KeyError` from three layers in | one message naming the file and the column |
| Model reuse | none | saved and reloaded, bit-identical predictions |
| Reproducibility | partial | seeds fixed, dependencies pinned, CI on three Pythons |

The accuracy barely moved across all of that, which is the reassuring outcome:
the original already avoided *feature* leakage, so its numbers were roughly
right even though its *methodology* leaked. Fixing the methodology removed the
optimistic bias without collapsing the score.

### Most predictive factors

By mutual information and SHAP, the dominant actionable signals are **KDA
ratio, deaths, assists / kill participation, and gold efficiency**. Surviving
and converting fights into a gold lead correlate most with winning.
`wards_placed` and `total_minions_killed` carry little signal and are dropped
by the selector.

## Performance and scale

Measured by `python benchmarks/bench_pipeline.py`, one subprocess per size
because `ru_maxrss` is a lifetime high-water mark. Generated data, tuning off,
macOS on Python 3.14:

| Games | Rows | CSV | Load+validate | Split+derive | Select | Train (5 roles) | Evaluate | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| 2,000 | 20,000 | 1 MB | 0.03 s | 0.04 s | 0.46 s | 3.4 s | 0.03 s | 32 MB |
| 10,000 | 100,000 | 6 MB | 0.07 s | 0.07 s | 2.58 s | 3.4 s | 0.04 s | 136 MB |
| 40,000 | 400,000 | 23 MB | 0.27 s | 0.12 s | 11.88 s | 3.8 s | 0.08 s | 544 MB |

What those numbers mean:

- **Feature selection dominates and grows faster than linearly.** It is
  O(n log n) per candidate, from the k-nearest-neighbour entropy estimator
  behind `mutual_info_classif`. At 400k rows it is 44x the cost of loading.
- **Fixed-hyperparameter training is nearly flat** across a 20x row count:
  XGBoost's `hist` method is bounded by tree construction at these sizes, not by
  rows. Tuning multiplies that by `grid size x folds` = 32 fits per role.
- **Memory is linear**, about 1.4 KB per row across the frame, the derived
  columns and the standardised copy the MI estimator makes.

**The ceiling.** Everything is held in memory, so ~8 GB caps this at roughly
5-6 million rows (around 550k games), where selection alone would take about
three minutes. Past that, the fix is to select features on a sample and train
out of core, not to buy RAM.

The full tuned run on the real dataset takes **4 min 52 s** end to end
(165 model fits: 32 per role for the grid search, plus one final fit each).

## Limitations and how I would improve this

**These are post-game, correlational features, not a real-time predictor.** The
model sees end-of-game stats. A player with a great KDA usually won because
their team was winning, so this measures *which finished games look like wins*,
not *what to do live*. A genuine in-game predictor would use the
`game_events.csv` timeline and snapshot features at fixed timestamps (gold/XP
diff at 10/15/20 minutes), training on state available before the result is
decided.

**Single held-out split, no confidence intervals.** One grouped train/test split
is reported. The numbers are stable, but repeated splits or nested CV with mean
and standard deviation would say whether the role-to-role differences (Top vs
Bot) are significant.

**Player-versus-player symmetry inside a game.** For a given role the two
opposing players are near mirror images. The split is by `game_id` so they never
straddle the boundary; the measured leakage from not doing so was small
(<0.002 AUC), but the symmetry still makes the task easier than a truly
independent sample would be.

**The feature set is deliberately narrow.** All team-objective columns are
excluded to keep the question honest. A richer but still leak-free model could
add champion identity, role-versus-role matchup, patch, and game length as a
normaliser, since raw counts accumulate with time.

**Calibration and thresholding.** Metrics use a fixed 0.5 threshold. A product
would calibrate probabilities (isotonic or Platt), tune the threshold to the use
case, and report a calibration curve.

**Interpretability is shown for one role.** SHAP is plotted for Bot only.
Comparing SHAP across all five roles would make the "what matters per role"
claim quantitative rather than illustrative.

**No serving layer.** Models are saved and reloaded, but there is no
`predict(player_stats, role)` API, no versioning of an artifact against the data
it was trained on, and no service around it.

**Not verified.** No human review of the modelling choices, no fuzzing of the
ingestion path with real malformed exports, no load testing: none of which this
project's stakes call for, and all of which are listed here rather than left to
look like they passed.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn,
Jupyter, pytest. Tooling: ruff, mypy strict, bandit, pip-audit, GitHub Actions.

## License

MIT. See `LICENSE`.
