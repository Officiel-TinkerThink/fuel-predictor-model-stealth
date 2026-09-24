# The rule model: the planners' rule, served as a model

Until a trained candidate beats it, production predicts with the planners' own rule from the
`Data Ratio` tab. It is packaged exactly like a trained model, so production validates, activates,
monitors and rolls it back the same way. When a trained model is ready, it replaces the rule with
an upload, not a code change.

```bash
fpt prepare "data/inbox/Data Trip Angber.xlsx"   # stages 1-2
fpt ready <dataset-version>                      # stage 3: train-issued.csv with its split
fpt package-rule <dataset-version> --reserve     # → packages/<model-version>.zip + reports/rule/
```

Upload the zip on production's **Unggah Kandidat** page and promote it on **Pengelolaan Model**
(ADR 0004: promotion stays manual).

## What the rule is

```
litres = base(unit) + km × litres_per_km(unit) + lifting_hours × litres_per_hour(unit)
```

| Term | From the `Data Ratio` tab |
|---|---|
| `litres_per_km` | 1 ÷ *Drive Ratio (KM/L)* |
| `litres_per_hour` | *Lift Ratio (L/Jam)*; zero for units that don't lift |
| `base` | *Safety Factor (L)*, plus the reserve if `--reserve` is set (below) |

Units the tab doesn't list use their group's median (VT 14 and VT 15 use the vacuum trucks'). A
group with no listed unit (the forklifts) and a name that isn't in the catalog use the fleet
median. Nothing ever predicts zero litres per km.

## Why it looks like a model

`rule_model.rule_pipeline` builds the same scikit-learn pipeline a trained candidate uses:

```
vehicle one-hot + km + lifting hours  →  pairwise interactions  →  linear regression
```

The regression's coefficients are **set** from the rule, not fitted: the "is VT 01 × km" term holds
VT 01's litres per km, "is Truck Crane 01 × lifting hours" holds its lift ratio, and the intercept
is the fleet's base. It reads the app's own feature contract (`baseline-v2`), is exported to ONNX,
and is packaged by the app's `ModelPackageBuilder`. To production it is simply a model version, named
`rule-data-ratio-…`.

**Moving to a trained model** means calling `pipeline.fit(features, issued_litres)` (or a
regularised regressor in place of `LinearRegression`, pooling by type and group as
`docs/pipeline.md` plans), then packaging the same way. The feature contract, export, package,
upload and monitoring stay the same.

## The two variants, and their numbers

Both were evaluated on the 60 held-out operations of `DSV-20260924-8485d134`, the last 20 % of each
unit's days. Nothing was fitted on those rows.

| Package | MAE | Bias | Within ± MAE |
|---|---|---|---|
| `rule-data-ratio-…` (the formula exactly as the sheet writes it) | 16.8 L | −16.8 L | 60 % |
| `rule-data-ratio-reserve-…` (formula + each unit's usual reserve) | 5.2 L | −0.5 L | 68 % |

**The formula alone is about 17 L below what planners actually issue**, for every unit and in both
splits. The sheet's own *Rekomendasi BBM* column is also above the formula (by about 10 L), so
planners add a reserve that the ratios don't write down. `--reserve` adds that reserve: for each
unit, the median of litres issued minus the formula over its **training** rows (the fleet median
for units without issued rows). The ratios themselves stay the planners'.

Two things production enforces that matter here:

- **Promotion needs MAE ≤ `FUEL_PREDICTOR_MAX_ACTIVE_MODEL_MAE_LITERS` (default 5 L).** The reserve
  variant is at 5.2 L, the formula alone at 16.8 L. Promoting either one means raising that
  setting, which is the owner's decision. The same value also sets the monitoring alert threshold.
- **The ± band on every estimate is the package's MAE.** The recommended allocation is the
  estimate plus the larger of that band and the safety margin. With the formula alone, the estimate
  sits 17 L low and only the band brings the allocation back up.

## Limits

- The vacuum trucks have no issued-litre operations in the workbook, so neither variant is checked
  on them. Their reserve is borrowed from the cranes and trucks.
- The rule predicts **issued** litres, the app's target (ADR 0002). Measured consumption is lower
  for the cranes (`docs/exploration.md` F2–F3); recommending less than today's issue is the trained
  model's job, on the measured target.
