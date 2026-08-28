# Security policy

## Reporting a vulnerability

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/DeharengOlivier/lol-win-prediction/security/advisories/new)
rather than opening a public issue. Include what you did, what happened, and
what you expected. A proof of concept helps but is not required.

Expect an acknowledgement within a week. If the report is valid, you will be
told what the fix is and when it ships, and you will be credited in the advisory
unless you prefer otherwise.

## Scope

This is a command-line tool and a library that reads CSV files from a folder the
operator points it at, trains models in memory, and prints a table. There is no
server, no authentication, no database and no network call, so most of the usual
attack surface does not exist here. What does:

- **Anything that lets a data file control the process.** `data.validate_players`
  checks the schema at load time and refuses anything outside it. A crafted CSV
  should never do more than produce a clear error.
- **Anything that lets a model artifact control the process.** Saved models use
  XGBoost's binary JSON format and a plain JSON manifest. Nothing is pickled,
  precisely because unpickling executes arbitrary code and model files get
  downloaded. A path that reintroduces deserialization of executable data is in
  scope.
- **Anything that writes outside the paths the operator named.** `--output-csv`
  and `--save-models` are the only writes, and saving refuses to replace an
  existing artifact without `--overwrite`.
- **The supply chain.** A dependency with a known vulnerability, or a way to
  make the package execute code at install or import time.

Out of scope: the statistical validity of the models, and the behaviour of
scikit-learn, XGBoost or SHAP themselves.

## What the project does on its side

- No secret has ever been committed to this repository, and none is needed:
  nothing here authenticates to anything.
- `pip-audit` and `bandit` run in CI on every push and pull request. The pinned
  lock in `requirements.txt` is kept free of known advisories, including for the
  optional notebook stack that the pipeline never imports.
- Dependabot watches both the Python dependencies and the workflow actions.
- The dataset is never committed, so no match data is redistributed here.
