# Exploration: findings and the decisions they lead to

Dataset `DSV-20260921-06c3f413` (the client's Google Sheet, exported 2026-09-21). Every number
below is reproduced by `fpt explore <version>`, which writes the charts and tables to
`reports/exploration/<version>/exploration.md`; section numbers (§) refer to that report. The
exploration reads only the clean tables (`fpt ready`), so it describes exactly what a model is
fitted on. When a finding showed data to be implausible, it became an erase rule in `ready.py`
rather than a note here.

## What the clean data is

| Table | Rows | Units | Target | Period |
|---|---|---|---|---|
| `train-issued.csv` | 285 operations | 6 (3 cranes, 3 trucks) | litres issued for the operation | Feb–Aug 2026 |
| `train-measured.csv` | 1,268 unit-days | 14 (3 cranes, 2 trucks, 9 VTs) | litres the fuel stick measured that day | Mar–Aug 2026 |

928 rows were erased on the way, every one listed in `removed.csv` with its rule.

## Findings

### F1 — Issued litres are the planners' formula (§2, §4)

For the trucks, issued litres are nearly a straight line in typed km (R² 0.86–0.97). For the cranes,
km plus lifting hours reproduce the planners' own rates: 11.1 / 11.5 / 15.7 L per lifting hour
fitted, against 10.6 / 10.6 / 14.4 in `Data Ratio`. A model on this target learns the allocation
rule back, with per-unit adjustments and an honest error band — useful, and exactly what the app
promises today (ADR 0002), but not a measurement of need.

### F2 — Measured consumption agrees with the planners' km-per-litre, except where lifting is (§5)

Measured km/L sits within 15% of `Data Ratio` for Prime Mover (0.97×), VT 01 (1.04×), VT 04 (0.94×),
VT 09 (1.03× — the 5-tonne unit) and others. That independent agreement is the best evidence the
repaired fuel-stick data is sound. The three cranes read 0.34–0.73× their ratio, because their fuel
also pays for lifting, which km/L does not count.

### F3 — Typed lifting hours do not show up in measured fuel (§4)

Fitted on measured litres, the extra fuel per typed lifting hour is −0.1 L (Truck Crane 01),
−3.3 L (Truck Crane 02) and 5.7 L (Wheel Crane), against 10.6–14.4 L in the rule. Either lifting
burns far less than allocated, or the typed hours are planned rather than actual. Either way the
lifting term must be **fitted from measured data**, not fixed at the rule's rate — and this is the
largest single source of over-allocation to show the client.

### F4 — Daily readings are noisy; weekly totals are not (§10)

Per unit, km and tracker hours explain R² 0.12–0.97 of *daily* measured litres. Summed per week,
the same units reach 0.58–1.00 (VT 06: 0.12 → 0.65, VT 05: 0.14 → 0.58, VT 08: 0.30 → 0.80). A
daily figure is the difference of two stick readings plus refuels and drains; its reading error
cancels over a week. So part of any daily error is measurement noise no model can remove.

### F5 — Units drift month to month (§6)

VT km/L moves between 1.30 (April) and 1.82 (July); issued km/L for trucks rises from 1.07 to 1.43
over the year (allocations got leaner). The relationship is not stationary.

### F6 — Error grows mildly with distance (§8); the target is not heavily skewed (§1)

Mean absolute residual rises from 6.6 L (shortest quarter of days) to 9.7 L (longest). Skew is
0.04–0.82 for issued and 0.41–1.44 for measured. A stationary day still burns fuel (67 zero-km days,
17 L on average — idle and pumping).

### F7 — Features overlap (§7)

Tracker drive and working hours correlate 0.71; working and idle 0.76. km is the strongest single
predictor of measured litres; tracker hours add little beyond it.

### F8 — Data per unit is thin and uneven (§9)

VTs have ~100–130 measured days each; the cranes 20–50; Truck Crane 02 only 20 after cleaning.
The fuel stick for the Prime Mover stops in early June.

## Decisions

### Data treatment

| Decision | Because |
|---|---|
| Erase, don't flag, anything implausible; log it in `removed.csv` | Agreed with the owner; the log keeps it reviewable. |
| Rules added from this exploration: fuel-stick km vs tracker km > 1.5× apart; typed km on a day both odometers say the unit never moved; a whole non-lifting unit whose km/L is > 3× off its group (VT 11: 6.4 km/L for a 12-tonne truck) | F2 shows the odometers and ratios normally agree; disagreement means a bad reading. Lifting units are exempt from the km/L rule (F2). |
| **No log transform** of the target | Skew is mild (F6), and fuel is additive — base + km × rate + hours × rate; a log would turn that into a product and lose the interpretable rates. |
| Keep zero-km days | They are real fuel (F6); dropping them would teach the model a stationary unit burns nothing. |
| **Time-ordered split, per unit**: each unit's last 20% of days is its test set | F5 (drift) rules out random splits; F8 rules out a single fleet-wide date (the Prime Mover had no test days). |

### Features and model shape

- Measured target: `gps_km`, one hours feature (drive or working, chosen by cross-validation — F7
  says not both), and `lifting_hours` for the three cranes only.
- Issued target: `distance_km`, `lifting_hours` (cranes only), `activity_mode`.
- Rates **pooled unit → type → group** (ADR 0015) with regularisation, because per-unit data is thin
  (F8) and per-unit weekly fits give unstable rates (VT 06 even a negative L/km).
- Feature construction is shared code between training and the app; learned transforms live inside
  the exported pipeline (see `docs/pipeline.md`).

### How error is measured

| Metric | Why |
|---|---|
| **MAE in litres** (primary) | The planner thinks in litres; robust to the few large days. |
| **WAPE** = Σ\|error\| ÷ Σ actual | Compares groups of different size (VT 09 burns 15 L, Wheel Crane 90); MAPE is not used — small and zero-km days make it explode. |
| **Bias** (mean error) | A model can have a fine MAE and still under-allocate every day. |
| **Under-allocation rate** of the recommended allocation, and **interval coverage** (80%, 90%) | Running a unit dry costs more than a few litres left over; the recommendation is an upper bound, so what matters is how often reality exceeds it. |
| **Weekly MAE** alongside daily (measured target) | F4: daily error has a noise floor from the stick; weekly error is the skill that matters for planning refills. |
| Every metric per group and per unit, and against **two baselines**: the planners' `Data Ratio` formula and a per-unit median | A model that does not beat the rule in use today has no case for replacing it. |

Model selection uses rolling-origin cross-validation inside the training split only; the test split is
opened once, for the final comparison.
