# Training pipeline

Training runs here, outside the production application, and hands production a model package
(ADR 0009 in `fuel-predictor`). Production never trains; it validates, activates, and monitors.
This document is the plan for that pipeline: the stages, what each one reads and writes, and the
rules that decide what becomes training data. It is written before the code so that the code can be
checked against it.

```
workbook (.xlsx)
   │  1. extract      one CSV per source sheet, values untouched, source row numbers kept
   ▼
data/raw/<dataset-version>/
   │  2. tidy         one row per operation; names resolved; quarantine with reasons
   ▼
data/tidy/<dataset-version>/  operations.csv · actuals.csv · quarantine.csv · report.md · app-import.csv
   │  3. split        time-ordered train / test, never random
   ▼
data/features/<dataset-version>/
   │  4. train        the feature contract production supports; MLflow run per candidate
   ▼
models/<model-version>/
   │  5. evaluate     untouched test set; overall and per-category metrics; interval coverage
   ▼
   │  6. package      model.onnx|model.skops + manifest + reference statistics + smoke tests
   ▼
packages/<model-version>.zip   →   uploaded to production (Unggah Paket), promoted by hand
```

Every stage is a command (`fpt <stage> …`) that reads the previous stage's directory and writes its
own. Nothing is edited in place, so a stage can be re-run and its output compared.

## Contracts this pipeline must honour

These live in the `fuel-predictor` repository and are imported, not copied, so there is exactly one
definition of each:

| Contract | Where | What it fixes |
|---|---|---|
| Daily operation | ADR 0001, `domain/daily_operation.py` | The unit of a row: one unit's complete activity sequence for a day, distance and lifting hours, out-and-back. |
| Prepared vs actual fuel | ADR 0002 | The label is *prepared* fuel (litres issued). Actual consumption is a different quantity, kept in a different table, never mixed into the label. |
| Feature contract | `application/prediction_features.py` (`baseline-v2`) | The feature names and order a model is scored with. A package declaring a contract production does not support is rejected at upload. |
| Vehicle lineage | ADR 0015, `kendaraan-angber.csv` | Unit → type → group. Training resolves the same catalog production does; the package records the catalog fingerprint. |
| Model package | ADR 0009, `schemas/model-package/*.json`, `packaging/model_packager.py` | Archive members, manifest fields, checksums, ONNX/skops only. |
| Import format | `SpreadsheetHistoricalDatasetSourceReader` | The CSV production imports as history (`Kategori ANGBER, Kendaraan, Mode Aktivitas, Jam Lifting, Jarak Total (km), Bahan Bakar Disiapkan (L), Sumber Jarak`, plus `Tanggal`). |

## Stage 1 — extract

Reads the workbook and writes one CSV per sheet we use, values as they are in the cells, plus a
`source_row` column so every later row can be traced back to a cell. Also records the workbook's
SHA-256 and the sheet list in `data/raw/<version>/source.json`.

Sheets used, and why:

| Sheet | Role |
|---|---|
| `Prime Mover`, `Truck Crane 01`, `Truck Crane 02`, `Whellcrane`, `TrontonWinch` | Operation rows (one per unit-day): date, litres issued, distance, lifting hours, activity text, trip text, from/to. **The training data.** |
| `Data Fuel Stick`, `Mentah - Data Fuel Stick` | Measured consumption per unit-day (mileage, initial/finish/refuel/drain). **Actuals**, kept separately. The `Mentah` sheet has date-typing damage; the cleaned one is used. |
| `Dim_Kendaraan`, `Peta_Nama_Sumber` | The fleet and its alias map. Cross-checked against production's catalog, which is authoritative. |
| `Data Lokasi`, `Missing Data` | Location names and coordinates, for resolving trip text into stops. |
| `Data Ratio` | The ratios the planners use today (km/L, L/h, safety factor) and the **vehicle types** (e.g. Scania P410 6x6 vs UD Quester 6x4 among the VTs). The baseline to beat, and the source for the catalog's `tipe` column. |
| `Fakta_BBM_Harian` | Daily litres per unit for 2025–2026, all 23 units, no distance or hours. Not operation-level; used only for context and for the VT question below. |

Not used: `Analisis`, `Origin Analisis Files`, `Journal`, `Set Up *`, `Dim_Tanggal` — reports and
lookups derived from the above.

## Stage 2 — tidy

Turns the per-unit sheets into `operations.csv`, one row per operation, with these rules. Every row
that fails a rule goes to `quarantine.csv` with the rule's name; nothing is silently dropped, and
the report counts each reason.

**Row selection**

- A sheet row is an operation only if it carries work: litres, distance, hours or activity text.
  The sheets are pre-created calendars, so most rows are blank.
- `Kegiatan` = `STANDBY` (any case) is not an operation. Quarantined as `standby`, kept for the
  record: a standby day with litres issued is a fact worth knowing, not a training row.

**Vehicle**

- Sheet name gives the unit, except `TrontonWinch`, where the `Angber` column distinguishes
  `OFT Tronton` from `OFT Winch Truck`. Blank `Angber` there → `vehicle_unknown`.
- Names are resolved through production's catalog (canonical name + aliases), the same resolver
  production uses at import. Sheet spellings that the catalog lacks (`Whellcrane`, `OFT Tronton`)
  are mapped here once, in `SHEET_VEHICLES`, and that mapping is asserted against the catalog at
  run time so a renamed unit cannot slip through.

**Label** — `prepared_fuel_liters` = `Liter`. Must be a number > 0; otherwise `no_fuel_issued`.
Litres with no work recorded (no distance, no hours) → `fuel_without_work`: real, but nothing to
learn from.

**Distance** — `total_distance_km` = `D (Km)` parsed leniently (`'80.8.'` → 80.8, comma decimals).
`Jarak - Data FS` (fuel-stick mileage) is carried as `fuel_stick_km` for cross-checking, never as
the distance. A row with hours but no distance is a pure lifting day and keeps distance 0 with
`distance_source = manual`; the app's domain requires distance > 0, so such rows are exported with
`total_distance_km` blank and are quarantined from the *app import* only (`lifting_only`), while
they stay in the training table.

**Activity mode** — derived from the numbers, not the text, because the text is free:
`lifting_hours > 0` and `distance > 0` → `transport_and_lifting`; hours only → `lifting`;
distance only → `transport`. The text is kept as `activity_text` for a human to check the derivation.

**Stops** — `Trip` split on ` - `, `-`, `,` into tokens, each resolved against the location catalog
(case, spacing, punctuation, numeral style insensitive). Resolved names go to `stop_sequence`
joined by ` → `; unresolved tokens are kept in `unresolved_stops` and counted in the report. Stops
are optional for training and for the app import; they are what makes the similar-history search
useful later.

**Actuals** — `actuals.csv` from `Data Fuel Stick`: unit (resolved), date, mileage, consumption.
Joined to operations on (unit, date) as `actual_fuel_liters` and `actual_km` *for the report only*.
Negative consumption (220 rows) is kept and flagged, not corrected.

**Provenance** — every operation row keeps `source_sheet`, `source_row`, and the dataset version.

## Stage 3 — split

Time-ordered: the last 20% of operations by date are the test set, never seen by training or
model selection. A random split would let a model learn June to predict June; the question the
model must answer is "given what we knew, how much fuel will *next* week's operation need".
Per-unit counts of both sets are printed so a unit that exists only in the test set is noticed.

## Stage 4 — train

The first candidate is the contract production already serves, `baseline-v2`: linear regression
over `vehicle_category`, `vehicle`, `activity_mode`, `distance_source`, `total_distance_km`,
`lifting_hours`. Same features, trained on the real workbook instead of the 12-row demo. That is
deliberately unambitious: it proves the whole path — workbook → package → production — before any
modelling decision is taken.

The second candidate is the reason the lineage work exists: `baseline-v3`, pooling by vehicle type
and group with per-level distance and lifting-hour slopes. It needs production to add `baseline-v3`
to `LINEAGE_AWARE_FEATURE_VERSIONS` and its supported contracts first; the packager will refuse to
build it until then.

Every run is an MLflow run (local `mlruns/`) with parameters, metrics, the dataset version and the
catalog fingerprint.

## Stage 5 — evaluate

On the untouched test set: MAE, RMSE, sMAPE, and coverage of the prediction interval (the share of
test rows whose prepared fuel falls inside the interval the package will claim). Overall and per
category (only `ANGBER` today, so also per vehicle group, which is what the planners will look at).
The `Data Ratio` heuristic (km ÷ ratio + hours × ratio + safety) is evaluated on the same rows as
the baseline to beat; a candidate that does not beat it is not packaged.

## Stage 6 — package

Uses `fuel_predictor.packaging.model_packager.ModelPackageBuilder` — imported from the app, never
reimplemented — to write `<model-version>.zip` with `manifest.json`, `reference-statistics.json`
(per-feature summaries of the training rows, for drift), `smoke-tests.json` (a handful of test rows
with the model's own predictions, so production can verify the model it loaded is the one that was
evaluated), and the ONNX (preferred) or skops artefact. The manifest carries the catalog fingerprint
computed from the same catalog the features were resolved with.

Upload is manual: production's *Unggah Paket* page validates the archive and an administrator
promotes it (ADR 0004). This pipeline stops at the file.

## What the initial dataset is, and is not

- **Scope:** the five sheeted units, February–August 2026. Roughly 500 usable operations. Enough for
  a linear baseline per unit; not enough to trust per-unit slopes, which is exactly why `v3`
  pools by type and group.
- **The VTs are not in it.** Thirteen vacuum trucks have daily litres but no trip rows, so there is no
  operation to predict for them yet. Two follow-ups, both outside this repo's first cut: (a) get
  the VT trip sheets, or (b) accept a coarser VT model on daily litres with route as the only
  feature. Their types from `Data Ratio` should go into production's catalog `tipe` column
  regardless — that is a catalog edit, not a training question.
- **Actuals stay out of the label.** Fuel-stick consumption is joined for the report so we can see,
  today, how far "prepared" sits from "burned" per unit. That gap is the calibration question ADR
  0002 defers; it is answered with a separate model or a correction factor, later, not by
  swapping the label now.

## Targets (added 2026-09-23, after the live-sheet audit)

Two targets exist and they are different questions:

| Target | Source | Rows | What a model on it is |
|---|---|---|---|
| **Issued litres** | unit sheets `Liter`, `Analisis` | 400 operations (6 units) | The planners' allocation rule, learnt back — with fitted per-unit rates and an honest uncertainty. What the app predicts today (ADR 0002). |
| **Measured consumption** | fuel stick, repaired | 1,786 clean unit-days (16 units, 1,392 VT) | What the unit actually burns for a given distance and lifting time. The basis for recommending *less* than today's allocation. |

Model shape for both, in this order:

1. **Structured linear** (first): `litres = base(unit) + km × rate(type→group) + lifting_hours × rate(crane)`,
   lifting term structurally zero for non-lifting units, rates pooled unit → type → group (ADR 0015).
   Interpretable, needs little data, and its coefficients are directly comparable with `Data Ratio`.
2. **Fixed lifting rate** (the field's rule, as a variant to compare, not the default): subtract
   `hours × L/h` and model the remainder on km. Equivalent to (1) with the lifting rate fixed instead
   of fitted; worth it only if fitted rates prove unstable.
3. **Gradient boosting** only once measured data per unit is well past a few hundred days.

Every recorded actual in the app becomes a new measured row, so retraining on (2)/(3) improves by itself.
