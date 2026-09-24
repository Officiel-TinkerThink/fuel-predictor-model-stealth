# fuel-predictor-training

Turns the planners' workbook into training data and, from it, model packages that the
[fuel-predictor](https://github.com/Officiel-TinkerThink/fuel-predictor) application can validate,
activate and monitor. Production never trains (its ADR 0009); this repository is where training
happens.

Read [`docs/pipeline.md`](docs/pipeline.md) first: it is the plan the code follows, stage by stage,
and the rules that decide what becomes a training row. [`docs/data-audit.md`](docs/data-audit.md)
records what the first workbook contained and what was decided about it.

## Status

| Stage | State |
|---|---|
| 1 extract | done — `fpt extract` |
| 2 tidy | done — `fpt tidy`; report, quarantine, actuals, app-import file |
| 3 split | done — `fpt ready`; the last 20 % of each unit's days is `test` |
| 4 train (`baseline-v2`) | next — until then, the **rule model** stands in (below) |
| 5 evaluate | done for the rule model — held-out MAE, bias, interval coverage |
| 6 package | done for the rule model — `fpt package-rule`, with the app's own `ModelPackageBuilder` |

## The rule model (what production serves first)

`fpt package-rule <dataset-version> --reserve` packages the planners' `Data Ratio` rule as an
ordinary model: the same pipeline, feature contract, ONNX export and package a trained model will
use, with its coefficients taken from the rule. Production accepts, activates and monitors it like
any model, and a trained candidate later replaces it by upload. Read
[`docs/rule-model.md`](docs/rule-model.md): it explains the rule, why the formula alone sits about
17 L below what's issued, and what `--reserve` does about it.

## Setup

The application is a dependency: its catalogs, feature contract and packager are imported, never
copied. For local work the checkout is expected next door as `../fuel_predictor`
(see `[tool.uv.sources]` in `pyproject.toml`); elsewhere, install it from git instead:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Run

```bash
fpt prepare "C:/Users/HP/Downloads/Data Trip Angber (updated).xlsx"
```

That is stages 1 and 2. Output lands under `data/raw/<dataset-version>/` and
`data/tidy/<dataset-version>/`, where the dataset version is derived from the workbook's contents
(`DSV-<date>-<sha256 prefix>`), so the same file always produces the same version and a re-export
produces a new one. The workbook and every output are reproducible from each other and are not
committed.

`data/tidy/<version>/` contains:

- `operations.csv` — one row per operation: unit and its type/group, date, activity mode, lifting
  hours, distance, litres issued, resolved stops, provenance (`source_sheet`, `source_row`), and the
  fuel-stick reading for that unit-day when one exists (for the report; never the label).
- `actuals.csv` — every fuel-stick reading, unit resolved, negatives flagged as `suspect`.
- `quarantine.csv` — every sheet row that was not made an operation, with the rule that excluded it.
- `app-import.csv` — the operations in the format production imports as history
  (*Impor Data Historis*), so the same rows feed both this pipeline and the app's similar-history
  search.
- `report.md` — counts per unit, litres per unit, prepared-vs-measured where both exist, stop
  resolution and the most frequent unresolved names, quarantine by reason.

## Collecting future data

`templates/Template Operasi Harian.xlsx` (regenerate with `fpt template`) is what the field should fill
from now on: one row per unit per day, plan and actual in separate columns, the fuel stick's four
raw numbers, dropdowns for names, number-only columns. Its *Kamus Kolom* sheet says why each
column exists.

## Checks

```bash
ruff check src tests && ruff format --check src tests && mypy && pytest
```
