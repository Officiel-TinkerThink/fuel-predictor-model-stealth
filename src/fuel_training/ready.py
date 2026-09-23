"""Stage 3: training-ready tables. Anything doubtful is erased, not flagged.

Two targets, two tables (docs/pipeline.md § Targets):

- `train-issued.csv` - litres issued per typed operation (the six sheeted
  units). What the app predicts today (ADR 0002).
- `train-measured.csv` - litres the fuel stick measured per unit-day, with the
  tracker's km and hours. The whole tracked fleet, the vacuum trucks included.

`Analisis` litres are deliberately not a target: 2025 is a fixed daily quota
and the 2026 vacuum-truck figures are tank refills every few days, so neither
says what a day's work needed.

A row that breaks a rule is removed from the table it would have been in and
written to `removed.csv` with the rule's name, so the erasure is itself
reviewable. The tables hold only rows that passed every rule.

Each table carries a `split` column: the last 20 % of each unit's own days is
`test`, never used to fit or choose a model. Per unit, not one fleet-wide
date: the fuel stick for some units stops earlier than for others, and a
single cutoff left the Prime Mover with no test days at all (exploration §9).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Only these units lift; any other unit's lifting hours are an entry error,
# and these units' lifting hours are required to explain their fuel.
LIFTING_UNITS = frozenset({"Truck Crane 01", "Truck Crane 02", "Wheel Crane"})
TEST_SHARE = 0.2
# A row this many robust standard deviations away from what the unit's own
# km and hours predict is treated as an entry or reading error.
OUTLIER_LIMIT = 3.5
MIN_ROWS_FOR_OUTLIER_CHECK = 15
DISTANCE_DISAGREEMENT = 2.0
ODOMETER_DISAGREEMENT = 1.5
# A unit whose overall km per litre sits this far from its group's typical
# unit (either way) is not measuring what its neighbours measure: a broken or
# mislabelled fuel stick. Measured km/L otherwise matches the planners' own
# ratios closely (Prime Mover 1.18 vs 1.2, VT 01 1.57 vs 1.5, VT 09 3.93 vs 3.8).
UNIT_KM_PER_LITRE_LIMIT = 3.0

# The tables carry every column worth having, not only today's features:
# which ones a model reads is the training stage's choice, and a column
# dropped here could not be chosen later without rebuilding the data.
ISSUED_COLUMNS = [
    "date",
    "vehicle",
    "vehicle_type",
    "vehicle_group",
    "activity_mode",
    "distance_km",
    "lifting_hours",
    "gps_km",
    "stop_sequence",
    "activity_text",
    "source_sheet",
    "source_row",
    "issued_litres",
    "split",
]
MEASURED_COLUMNS = [
    "date",
    "vehicle",
    "vehicle_type",
    "vehicle_group",
    "gps_km",
    "gps_usage_hours",
    "gps_drive_hours",
    "gps_working_hours",
    "gps_idle_hours",
    "lifting_hours",
    "typed_km",
    "fs_mileage_km",
    "fs_refuel_litres",
    "fs_drain_litres",
    "measured_litres",
    "split",
]


@dataclass(slots=True)
class Removals:
    rows: list[dict[str, object]]

    def take(self, frame: pd.DataFrame, mask: pd.Series, table: str, rule: str) -> pd.DataFrame:
        """Remove the rows `mask` selects, recording each with its rule."""
        for _, row in frame[mask].iterrows():
            self.rows.append(
                {"table": table, "rule": rule, "date": row["date"], "vehicle": row["vehicle"]}
            )
        kept: pd.DataFrame = frame[~mask]
        return kept

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["table", "rule", "date", "vehicle"])


# --- the rules -------------------------------------------------------------------------


def robust_outliers(frame: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
    """Per unit: fit the target on the features by least squares and mark rows
    whose residual is beyond OUTLIER_LIMIT robust standard deviations (MAD).
    Units with too few rows are not judged."""
    marked = pd.Series(False, index=frame.index)
    for _, part in frame.groupby("vehicle"):
        if len(part) < MIN_ROWS_FOR_OUTLIER_CHECK:
            continue
        design = np.column_stack(
            [np.ones(len(part))] + [part[f].fillna(0).to_numpy(float) for f in features]
        )
        observed = part[target].to_numpy(float)
        coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
        residual = observed - design @ coefficients
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if spread <= 0:
            continue
        marked.loc[part.index] = np.abs(residual - np.median(residual)) > OUTLIER_LIMIT * spread
    return marked


def implausible_units(frame: pd.DataFrame, km: str, litres: str) -> pd.Series:
    """Every row of a unit whose km/L is UNIT_KM_PER_LITRE_LIMIT times off the
    median unit of its group."""
    totals = frame.groupby(["vehicle", "vehicle_group"])[[km, litres]].sum()
    per_unit = totals[km] / totals[litres]
    group_median = per_unit.groupby(level="vehicle_group").transform("median")
    off = (per_unit / group_median).pipe(
        lambda r: (r > UNIT_KM_PER_LITRE_LIMIT) | (r < 1 / UNIT_KM_PER_LITRE_LIMIT)
    )
    bad_units = {str(index[0]) for index, is_off in zip(off.index, off, strict=True) if is_off}
    return frame["vehicle"].isin(bad_units)


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Time-ordered per unit: each unit's latest TEST_SHARE of days is its test set."""
    result = frame.copy()
    result["split"] = "train"
    for _, part in result.groupby("vehicle"):
        days = sorted(part["date"].unique())
        if len(days) < 5:
            continue
        cutoff = days[int(len(days) * (1 - TEST_SHARE))]
        result.loc[part.index[part["date"] >= cutoff], "split"] = "test"
    return result


# --- the tables --------------------------------------------------------------------------


def issued_table(operations: pd.DataFrame, gps: pd.DataFrame, removals: Removals) -> pd.DataFrame:
    table = operations.rename(
        columns={
            "operation_date": "date",
            "total_distance_km": "distance_km",
            "prepared_fuel_liters": "issued_litres",
        }
    )[
        [
            "date",
            "vehicle",
            "vehicle_type",
            "vehicle_group",
            "activity_mode",
            "distance_km",
            "lifting_hours",
            "stop_sequence",
            "activity_text",
            "source_sheet",
            "source_row",
            "issued_litres",
        ]
    ]
    table = table.merge(gps, on=["vehicle", "date"], how="left")
    name = "train-issued"
    lifts = table["vehicle"].isin(LIFTING_UNITS)
    table = removals.take(
        table, ~lifts & (table["lifting_hours"].fillna(0) > 0), name, "lifting_on_non_lifting_unit"
    )
    lifts = table["vehicle"].isin(LIFTING_UNITS)
    table = removals.take(
        table, lifts & ~(table["lifting_hours"].fillna(0) > 0), name, "crane_without_lifting_hours"
    )
    ratio = table["gps_km"] / table["distance_km"]
    disagree = (table["gps_km"] > 0) & (table["distance_km"] > 0)
    disagree &= (ratio > DISTANCE_DISAGREEMENT) | (ratio < 1 / DISTANCE_DISAGREEMENT)
    table = removals.take(table, disagree, name, "typed_km_disagrees_with_tracker")
    table["distance_km"] = table["distance_km"].fillna(0.0)
    table["lifting_hours"] = table["lifting_hours"].fillna(0.0)
    outliers = robust_outliers(table, "issued_litres", ["distance_km", "lifting_hours"])
    table = removals.take(table, outliers, name, "outlier_for_unit")
    return assign_split(table)[ISSUED_COLUMNS].reset_index(drop=True)


def measured_table(
    unit_days: pd.DataFrame, operations: pd.DataFrame, removals: Removals
) -> pd.DataFrame:
    table = unit_days[unit_days["fs_consumption_litres"].notna()].rename(
        columns={"fs_consumption_litres": "measured_litres"}
    )
    name = "train-measured"
    issues = table["issues"].fillna("")
    for rule in (
        "fs_negative_consumption",
        "fs_consumption_above_tank",
        "fs_km_per_litre_implausible",
    ):
        table = removals.take(table, issues.loc[table.index].str.contains(rule), name, rule)
    table = removals.take(table, table["measured_litres"] <= 0, name, "no_consumption_measured")
    table = removals.take(table, table["gps_km"].isna(), name, "no_tracker_record")
    # Two odometers for the same day: the fuel stick's mileage and the tracker.
    # When they disagree, one of them is wrong and the day cannot be trusted.
    both = (table["gps_km"] > 0) & (table["fs_mileage_km"] > 0)
    odometers = table["fs_mileage_km"] / table["gps_km"]
    table = removals.take(
        table,
        both & ((odometers > ODOMETER_DISAGREEMENT) | (odometers < 1 / ODOMETER_DISAGREEMENT)),
        name,
        "fuel_stick_km_disagrees_with_tracker",
    )
    # The planners typed a distance, but tracker and stick both say the unit
    # never moved: the day contradicts itself.
    table = removals.take(
        table,
        (table["typed_km"].fillna(0) > 0)
        & (table["gps_km"] == 0)
        & (table["fs_mileage_km"].fillna(0) == 0),
        name,
        "typed_km_but_no_movement_recorded",
    )
    for column in (
        "gps_km",
        "gps_usage_hours",
        "gps_drive_hours",
        "gps_working_hours",
        "gps_idle_hours",
    ):
        table[column] = table[column].fillna(0.0)
    # Lifting hours are only known from a typed operation. A lifting unit's
    # day without one cannot be explained, so it is not learnt from; every
    # other unit lifts zero hours by definition.
    typed = operations.rename(columns={"operation_date": "date"})[
        ["vehicle", "date", "lifting_hours"]
    ].rename(columns={"lifting_hours": "typed_lifting_hours"})
    table = table.merge(typed, on=["vehicle", "date"], how="left")
    lifts = table["vehicle"].isin(LIFTING_UNITS)
    table = removals.take(
        table, lifts & table["typed_lifting_hours"].isna(), name, "crane_day_without_typed_hours"
    )
    lifts = table["vehicle"].isin(LIFTING_UNITS)
    table["lifting_hours"] = np.where(lifts, table["typed_lifting_hours"].fillna(0.0), 0.0)
    # km per litre means nothing for a lifting unit, whose fuel also pays for
    # lifting: judged on it, the Wheel Crane looks broken when it is not.
    table = removals.take(
        table,
        implausible_units(table, "gps_km", "measured_litres")
        & ~table["vehicle"].isin(LIFTING_UNITS),
        name,
        "unit_km_per_litre_implausible",
    )
    outliers = robust_outliers(table, "measured_litres", ["gps_km", "gps_working_hours"])
    table = removals.take(table, outliers, name, "outlier_for_unit")
    return assign_split(table)[MEASURED_COLUMNS].reset_index(drop=True)


# --- the run -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadyResult:
    directory: Path
    issued: pd.DataFrame
    measured: pd.DataFrame
    removed: pd.DataFrame


def build(tidy_directory: Path, ready_root: Path) -> ReadyResult:
    operations = pd.read_csv(tidy_directory / "operations.csv")
    unit_days = pd.read_csv(tidy_directory / "unit-days.csv")
    gps = unit_days[["vehicle", "date", "gps_km"]].dropna(subset=["gps_km"])
    removals = Removals([])
    issued = issued_table(operations, gps, removals)
    measured = measured_table(unit_days, operations, removals)
    removed = removals.frame()
    directory = ready_root / tidy_directory.name
    directory.mkdir(parents=True, exist_ok=True)
    issued.to_csv(directory / "train-issued.csv", index=False)
    measured.to_csv(directory / "train-measured.csv", index=False)
    removed.to_csv(directory / "removed.csv", index=False)
    (directory / "README.md").write_text(
        report(tidy_directory.name, issued, measured, removed), encoding="utf-8"
    )
    return ReadyResult(directory, issued, measured, removed)


def _summary(table: pd.DataFrame, target: str) -> list[str]:
    lines = ["| Unit | Group | Train | Test | Mean litres |", "|---|---|---|---|---|"]
    for (unit, group), part in table.groupby(["vehicle", "vehicle_group"]):
        splits = part["split"].value_counts()
        lines.append(
            f"| {unit} | {group} | {splits.get('train', 0)} | {splits.get('test', 0)} "
            f"| {part[target].mean():.1f} |"
        )
    return lines


def report(
    version: str, issued: pd.DataFrame, measured: pd.DataFrame, removed: pd.DataFrame
) -> str:
    lines = [
        f"# Training-ready data — {version}",
        "",
        "Only rows that passed every rule are here; everything erased is in `removed.csv`.",
        "The last 20% of days is `split = test`: never fit or choose a model on it.",
        "",
        f"## train-issued.csv — {len(issued)} operations (target: `issued_litres`)",
        "",
        "Features: vehicle (+ type, group), activity_mode, distance_km, lifting_hours "
        "(non-zero only for Truck Crane 01/02 and Wheel Crane).",
        "",
        *_summary(issued, "issued_litres"),
        "",
        f"## train-measured.csv — {len(measured)} unit-days (target: `measured_litres`)",
        "",
        "Features: vehicle (+ type, group), gps_km, gps drive / working / idle hours, "
        "lifting_hours (cranes, from the typed operation).",
        "",
        *_summary(measured, "measured_litres"),
        "",
        "## Erased",
        "",
        "| Table | Rule | Rows |",
        "|---|---|---|",
    ]
    counts = removed.groupby(["table", "rule"]).size().reset_index(name="rows")
    for _, row in counts.iterrows():
        lines.append(f"| {row['table']} | {row['rule']} | {row['rows']} |")
    return "\n".join(lines) + "\n"
