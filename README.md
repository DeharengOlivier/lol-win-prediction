# League of Legends - Per-role Match Outcome Prediction

Predict whether a player's team won a League of Legends match from that single
player's end-of-game statistics, and surface which quantifiable factors matter
most for winning. A separate, tuned **XGBoost** model is trained for each role
(Top, Jungle, Mid, Bot, Support).

This started as a single exploratory notebook. It has been refactored into a
small, **tested** Python package with a reproducible, leakage-free pipeline.

## Goals

1. Predict the binary outcome (win / loss) of a match from one player's stats.
2. Identify the most actionable variables a player should focus on.

## What makes this honest

The dataset ships with **team-level objective columns** (`tower_kills`,
`inhibitor_kills`, `team_kills`, `baron_kills`, `dragon_kills`, `herald_kills`).
These are shared by all five players of a team and almost *define* the result:
`tower_kills` alone has a univariate ROC-AUC of **0.99** against the win label.
Using them is **data leakage** and answers the wrong question.

The pipeline therefore models **only on a player's own performance columns** and
features derived from them. The exclusion is documented in code
(`features.LEAKY_COLUMNS`) and enforced by a unit test.

## Architecture

```
lol-win-prediction/
├── src/lolwin/            # reusable, importable package
│   ├── __init__.py        # ROLES, RANDOM_STATE
│   ├── data.py            # load + clean + (optional) merge metadata; game-level sampling
│   ├── features.py        # derived features, mutual-information selection, LEAKY_COLUMNS guard
│   ├── model.py           # per-role XGBoost: grouped split + grouped-CV ROC-AUC tuning
│   ├── evaluate.py        # accuracy + ROC-AUC + F1 + confusion matrix, results table
│   └── cli.py             # `python -m lolwin.cli` runs the whole pipeline
├── tests/                 # pytest suite (synthetic data, no CSVs needed)
├── train.py               # `python train.py` entry point (wraps the CLI)
├── lol-win-prediction.ipynb   # clean English narrative notebook over the package
├── requirements.txt
└── data/                  # git-ignored; place the CSV files here
```

Design choices:

- **Each module has one responsibility** and is independently testable.
- The notebook is a thin narrative layer that calls into the package, so the
  logic exists in exactly one place.
- Type hints, docstrings and comments explain the *why* (especially the leakage
  guard and the split discipline), not just the *what*.

## Dataset

Public League of Legends esports match dataset (3 CSV files):
`game_players_stats.csv` (~44 MB, the table we model on), `game_metadata.csv`,
`game_events.csv`. The files are **not** committed (`data/` and `*.csv` are
git-ignored).

To run the project:

1. Download the dataset archive (the Kaggle "Ecamania Esports" LoL dataset).
2. Extract the three CSVs into a `data/` folder at the repo root:

   ```
   data/
   ├── game_players_stats.csv
   ├── game_metadata.csv
   └── game_events.csv
   ```

The loader raises a clear error pointing here if the files are missing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# macOS only, if XGBoost reports a libomp error:
brew install libomp
```

`tensorflow`/`keras` is **not** required. The original notebook benchmarked a
small Keras net; it was dropped here (optional, no wheels on the local Python
3.14 toolchain, and XGBoost already wins). Everything runs without it.

## Run

Full pipeline (load -> features -> per-role tuned training -> evaluation):

```bash
python train.py                      # full data, grid-searched (a few minutes)
python train.py --sample 0.2         # 20% of games, fast iteration
python train.py --no-tune            # skip grid search (smoke test)
python train.py --output-csv results/metrics.csv
```

Notebook (clean, all-English narrative):

```bash
jupyter notebook lol-win-prediction.ipynb
```

Tests (fast, run on synthetic data, no CSVs needed):

```bash
pytest -q
```

## Results (measured, full dataset)

Final per-role metrics on a **held-out test set**, with hyperparameters chosen
by grouped stratified cross-validation on the training data only (ROC-AUC),
split by `game_id`, seed fixed. 374,490 rows / 37,449 games, perfectly balanced
label (win rate 0.5000).

| Role    | Accuracy | ROC-AUC | F1     |
|---------|----------|---------|--------|
| Top     | 0.8792   | 0.9541  | 0.8794 |
| Jungle  | 0.8951   | 0.9629  | 0.8956 |
| Mid     | 0.8856   | 0.9585  | 0.8863 |
| Bot     | 0.8978   | 0.9657  | 0.8977 |
| Support | 0.8885   | 0.9586  | 0.8898 |
| **Overall** | **0.8893** | **0.9599** | **0.8898** |

### Before vs after the engineering pass

| Aspect | Original notebook | This version |
|---|---|---|
| Headline accuracy | ~0.87-0.90 (per role) | ~0.88-0.90 (per role), **honestly measured** |
| Hyperparameter selection | on the **test set** (leakage) | grouped CV on **train only** |
| Early stopping eval set | the **test set** (leakage) | removed |
| Train/test split | single random split | grouped by `game_id` (no same-game leakage) |
| Metrics | accuracy only | accuracy + **ROC-AUC (~0.96)** + **F1** |
| Reproducibility | partial | all seeds fixed, CLI reproducible |

The accuracy barely moved, which is the honest and reassuring outcome: the
original already avoided *feature* leakage (it never used the team-objective
columns), so its numbers were roughly right even though its *methodology* leaked.
Fixing the test-set tuning removed the optimistic bias without collapsing the
score, and we now report AUC/F1 alongside accuracy.

### Most predictive factors

By mutual information and SHAP, the dominant, actionable signals are
**KDA ratio, deaths, assists / kill participation, and gold efficiency**.
Surviving (low deaths) and converting fights into a gold lead correlate most
with winning. `wards_placed` and `total_minions_killed` carry little signal and
are dropped by the MI selector.

## Limitations and how I would improve this

**These are post-game, correlational features, not a real-time predictor.**
The model sees end-of-game stats. A player with a great KDA usually won because
their team was winning, so this measures *which finished games look like wins*,
not *what to do live*. It cannot forecast the outcome at minute 15. To build a
genuine in-game predictor I would use the `game_events.csv` timeline and snapshot
features at fixed timestamps (gold/XP diff at 10/15/20 min), training on state
that is actually available before the result is decided.

**Single held-out split, no confidence intervals.** I report one grouped
train/test split per role. The numbers are stable, but I would add repeated
splits or nested CV and report mean +/- std so the role-to-role differences
(e.g. Top vs Bot) can be judged as significant or not.

**Player-vs-player symmetry inside a game.** For a given role, the two opposing
players are near-mirror images (one won, one lost). I split by `game_id` to
avoid putting both sides across the train/test boundary; the measured leakage
from not doing so was small (<0.002 AUC), but the symmetry still makes the task
slightly easier than a truly independent sample would be.

**Feature set is deliberately narrow.** I excluded all team-objective columns to
keep the question honest. A richer but still leak-free model could add champion
identity, role-vs-role matchup, patch/meta, and game length as a normaliser
(stats accumulate over time, so a 40-minute game inflates raw counts).

**Calibration and thresholding.** Metrics use a fixed 0.5 threshold. For a
product (e.g. a coaching tool) I would calibrate probabilities (isotonic /
Platt) and tune the threshold to the use case, and report a calibration curve.

**Model interpretability is single-role.** SHAP is shown for Bot only. A full
write-up would compare SHAP across all five roles to make the "what matters per
role" claim quantitative rather than illustrative.

**No model persistence / serving.** The pipeline trains and evaluates in memory.
Productionising would mean saving the per-role models + the feature list,
versioning them, and exposing a `predict(player_stats, role)` API.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn,
Jupyter, pytest.
