"""Stage 2b: one clean row per unit per day, from every source the client's sheet has.

The operation sheets (stage 2) cover six units with typed distance and lifting
hours. The client's workbook holds three more sources that cover the whole
fleet, the vacuum trucks included:

- `Analisis`, 2026 block: litres issued per unit per day.
- `Data GPS`: the tracker's daily km and drive / working / idle hours.
- `Mentah - Data Fuel Stick`: the dip-stick readings, i.e. measured consumption.

Exporting the sheet turned many decimals into dates: a cell holding `30.5`
comes back as 30 May, `1.4` as 1 April. `repair_number` reverses that
(day.month), and is checked against the sheet's own cleaned copy of the fuel
stick, where it agrees on 2,087 of 2,088 mileages and every consumption.

Nothing here decides what the model learns from. Each row carries every value
side by side, and `issues` names each rule a value broke; stage 3 chooses.
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from fuel_training.catalog import Catalogs
from fuel_training.extract import raw_frame
from fuel_training.tidy import fuel_stick_vehicle, parse_date, parse_duration_hours, parse_number

# What a value may be before it is called implausible rather than learnt from.
MAX_DAILY_LITRES = 400.0  # more than a tank
MAX_DAILY_KM = 1000.0
MAX_HOURS = 24.0
MIN_KM_PER_LITRE, MAX_KM_PER_LITRE = 0.05, 10.0
# Typed and tracked distance for the same day, beyond this ratio either way.
DISTANCE_DISAGREEMENT = 2.0

_EXPORTED_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?: 00:00:00)?$")


def repair_number(text: object) -> float | None:
    """A numeric cell, undoing the export's decimal-to-date damage.

    `2026-05-30 00:00:00` in a numeric column was `30.5`; `2026-04-01` was
    `1.4`. A real number passes through `parse_number` unchanged.
    """
    if text is None:
        return None
    cleaned = str(text).strip()
    if match := _EXPORTED_DATE.match(cleaned):
        _, month, day = (int(part) for part in match.groups())
        return float(f"{day}.{month}")
    return parse_number(cleaned)


# --- sources ------------------------------------------------------------------------


def issued_litres(raw_directory: Path, catalogs: Catalogs) -> pd.DataFrame:
    """The 2026 block of Analisis: the third table headed `Tanggal`."""
    sheet = raw_frame(raw_directory, "Analisis")
    position = next(
        position for position in range(len(sheet)) if (sheet.iloc[position] == "Tanggal").sum() >= 3
    )
    header = sheet.iloc[position]
    columns = [str(column) for column in sheet.columns]
    tanggal_columns = [column for column in columns if header[column] == "Tanggal"]
    block_columns = columns[columns.index(tanggal_columns[2]) :]
    records: list[dict[str, object]] = []
    for _, row in sheet.iloc[position + 1 :].iterrows():
        day = parse_date(row[block_columns[0]])
        if day is None:
            continue
        for column in block_columns[2:]:
            written = str(header[column]).strip()
            if not written or written in {"Total", "BBM Komulatif", "nan"}:
                continue
            vehicle = catalogs.vehicle(written)
            if vehicle is None:
                continue
            litres = repair_number(row[column])
            if litres is None:
                continue
            records.append({"vehicle": vehicle, "date": day, "issued_litres": litres})
    return pd.DataFrame(records, columns=["vehicle", "date", "issued_litres"])


def gps_days(raw_directory: Path, catalogs: Catalogs) -> pd.DataFrame:
    """Data GPS summed per unit-day (a unit occasionally has two trip rows)."""
    sheet = raw_frame(raw_directory, "Data GPS")
    unit_column = sheet.columns[2]  # "Nama Switch": the unit as the sheet names it
    records: list[dict[str, object]] = []
    for _, row in sheet.iterrows():
        written = str(row[unit_column]).strip()
        vehicle = fuel_stick_vehicle(written, catalogs) if written else None
        day = parse_date(str(row["Start Time"])[:19])
        if vehicle is None or day is None:
            continue
        records.append(
            {
                "vehicle": vehicle,
                "date": day,
                "gps_km": repair_number(row["Mileage Summary(KM)"]),
                "gps_usage_hours": parse_duration_hours(row["Vehicle Usage(hh:mm:ss)"]),
                "gps_drive_hours": parse_duration_hours(row["Drive(hh:mm:ss)"]),
                "gps_working_hours": parse_duration_hours(row["Working(hh:mm:ss)"]),
                "gps_idle_hours": parse_duration_hours(row["Idle(hh:mm:ss)"]),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.groupby(["vehicle", "date"], as_index=False).sum(min_count=1)


def fuel_stick_days(raw_directory: Path, catalogs: Catalogs) -> pd.DataFrame:
    """The dip-stick readings, per unit-day.

    The sheet keeps two copies: `Mentah` as the stick logger wrote it, and
    `Data Fuel Stick`, the same rows with correct dates. The export damaged
    `Mentah`'s numbers (repairable, see `repair_number`) and its dates, where a
    date-typed cell cannot say on its own whether day and month were swapped.
    So the clean tab is read when present - its values equal repaired
    `Mentah` - and `Mentah` is only the fallback.
    """
    clean = (raw_directory / "data-fuel-stick.csv").exists()
    sheet = raw_frame(raw_directory, "Data Fuel Stick" if clean else "Mentah - Data Fuel Stick")
    records: list[dict[str, object]] = []
    for _, row in sheet.iterrows():
        vehicle = fuel_stick_vehicle(row["Vehicle Code"], catalogs)
        day = parse_date(row["Date"]) if clean else _fuel_stick_date(row["Date"])
        if vehicle is None or day is None:
            continue
        records.append(
            {
                "vehicle": vehicle,
                "date": day,
                "fs_mileage_km": repair_number(row["Mileage (KM)"]),
                "fs_initial_litres": repair_number(row["Initial Fuel (L)"]),
                "fs_finish_litres": repair_number(row["Finish Fuel (L)"]),
                "fs_refuel_litres": repair_number(row["Refuel (L)"]),
                "fs_drain_litres": repair_number(row["Drain (L)"]),
                "fs_consumption_litres": repair_number(row["Fuel Consumption"]),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.groupby(["vehicle", "date"], as_index=False).sum(min_count=1)


def _fuel_stick_date(text: object) -> date | None:
    cleaned = str(text).strip()
    if match := _EXPORTED_DATE.match(cleaned):
        year, month, day = (int(part) for part in match.groups())
        try:
            return date(year, day, month)  # day and month were swapped
        except ValueError:
            return None
    return parse_date(cleaned)


def typed_operations(tidy_directory: Path) -> pd.DataFrame:
    operations = pd.read_csv(tidy_directory / "operations.csv")
    operations["date"] = pd.to_datetime(operations["operation_date"]).dt.date
    return operations[
        ["vehicle", "date", "total_distance_km", "lifting_hours", "activity_mode", "activity_text"]
    ].rename(columns={"total_distance_km": "typed_km"})


# --- the table ---------------------------------------------------------------------------


def unit_days(raw_directory: Path, tidy_directory: Path, catalogs: Catalogs) -> pd.DataFrame:
    keys = ["vehicle", "date"]
    table = issued_litres(raw_directory, catalogs)
    for part in (
        gps_days(raw_directory, catalogs),
        fuel_stick_days(raw_directory, catalogs),
        typed_operations(tidy_directory),
    ):
        if not part.empty:
            table = table.merge(part, on=keys, how="outer")
    # A day exists only if something happened on it: litres issued, the
    # tracker moving, a stick reading, or a typed operation. The Analisis
    # calendar pre-fills the rest of the year with zeros.
    activity = ["issued_litres", "gps_km", "gps_usage_hours", "fs_consumption_litres", "typed_km"]
    activity = [column for column in activity if column in table.columns]
    happened = (table[activity].fillna(0) != 0).any(axis=1) | table["lifting_hours"].notna()
    table = table[happened]
    observed = ["gps_km", "fs_consumption_litres", "typed_km"]
    last_day = (
        table.dropna(how="all", subset=[c for c in observed if c in table]).loc[:, "date"].max()
    )
    table = table[table["date"] <= last_day]
    lineage = {option.name: option.lineage for option in catalogs.vehicles.options()}
    table.insert(1, "vehicle_group", table["vehicle"].map(lambda v: lineage[v].group))
    table.insert(1, "vehicle_type", table["vehicle"].map(lambda v: lineage[v].type))
    table["issues"] = table.apply(_issues, axis=1)
    result: pd.DataFrame = table.sort_values(keys).reset_index(drop=True)
    return result


def _number(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Only between two positive values; anything else says nothing."""
    if not numerator or not denominator or numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


def _issues(row: pd.Series) -> str:
    found: list[str] = []
    litres, gps_km = _number(row, "issued_litres"), _number(row, "gps_km")
    typed_km, lifting = _number(row, "typed_km"), _number(row, "lifting_hours")
    consumed, stick_km = _number(row, "fs_consumption_litres"), _number(row, "fs_mileage_km")
    usage = _number(row, "gps_usage_hours")
    if litres is not None and litres > MAX_DAILY_LITRES:
        found.append("issued_above_tank")
    if gps_km is not None and gps_km > MAX_DAILY_KM:
        found.append("gps_km_implausible")
    if usage is not None and usage > MAX_HOURS:
        found.append("gps_hours_implausible")
    if consumed is not None and consumed < 0:
        found.append("fs_negative_consumption")
    if consumed is not None and consumed > MAX_DAILY_LITRES:
        found.append("fs_consumption_above_tank")
    km_per_litre = _ratio(stick_km, consumed)
    if km_per_litre is not None and not MIN_KM_PER_LITRE <= km_per_litre <= MAX_KM_PER_LITRE:
        found.append("fs_km_per_litre_implausible")
    tracked_over_typed = _ratio(gps_km, typed_km)
    if tracked_over_typed is not None and not (
        1 / DISTANCE_DISAGREEMENT <= tracked_over_typed <= DISTANCE_DISAGREEMENT
    ):
        found.append("typed_vs_gps_km_disagree")
    if litres and litres > 0 and not any(v and v > 0 for v in (gps_km, typed_km, lifting)):
        found.append("issued_without_recorded_work")
    return " ".join(found)


# --- the run -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitDaysResult:
    path: Path
    table: pd.DataFrame


def build(raw_directory: Path, tidy_directory: Path, catalogs: Catalogs) -> UnitDaysResult:
    table = unit_days(raw_directory, tidy_directory, catalogs)
    path = tidy_directory / "unit-days.csv"
    table.to_csv(path, index=False)
    (tidy_directory / "unit-days-report.md").write_text(report(table), encoding="utf-8")
    return UnitDaysResult(path, table)


def report(table: pd.DataFrame) -> str:
    lines = ["# Unit-day table", "", f"- Rows: **{len(table)}** (unit × day with any value)", ""]
    header = "| Unit | Group | Days | Issued >0 | GPS km >0 | Fuel stick | Typed op | Clean |"
    lines += [header, "|---|---|---|---|---|---|---|---|"]
    for (unit, group), part in table.groupby(["vehicle", "vehicle_group"]):
        issued = int((part["issued_litres"] > 0).sum())
        gps = int((part["gps_km"] > 0).sum()) if "gps_km" in part else 0
        stick = int(part["fs_consumption_litres"].notna().sum())
        typed = int((part["typed_km"].notna() | part["lifting_hours"].notna()).sum())
        clean = int(((part["issued_litres"] > 0) & (part["issues"] == "")).sum())
        lines.append(
            f"| {unit} | {group} | {len(part)} | {issued} | {gps} | {stick} | {typed} | {clean} |"
        )
    lines += ["", "Clean = litres issued and no rule broken.", "", "## Issues", ""]
    counts = table["issues"].str.split().explode().dropna().value_counts()
    lines += [f"- `{issue}` × {count}" for issue, count in counts.items()]
    return "\n".join(lines) + "\n"
