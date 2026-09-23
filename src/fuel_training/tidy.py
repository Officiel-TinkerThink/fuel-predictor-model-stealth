"""Stage 2: one row per operation, names resolved, everything else quarantined with a reason.

The rules are the ones `docs/pipeline.md` states. Each is a small function so
it can be tested on a handful of values and changed without re-reading the
workbook. Nothing is dropped silently: a row that is not an operation goes to
the quarantine table with the name of the rule that sent it there, and the
report counts every reason.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from fuel_training.catalog import SHEET_VEHICLES, Catalogs
from fuel_training.extract import OPERATION_SHEETS, raw_frame

STANDBY = "standby"
NO_DATE = "no_date"
VEHICLE_UNKNOWN = "vehicle_unknown"
NO_FUEL_ISSUED = "no_fuel_issued"
FUEL_WITHOUT_WORK = "fuel_without_work"
IMPLAUSIBLE = "implausible_value"
LIFTING_ONLY = "lifting_only"  # app import only: the domain needs distance > 0

# A day has 24 hours, and no unit leaves the field: anything beyond these is a
# typo (a year in a numeric cell shows up as 2026 km) and is not learnt from.
MAX_DISTANCE_KM = 1000.0
MAX_LIFTING_HOURS = 24.0

OPERATION_COLUMNS = (
    "dataset_version",
    "source_sheet",
    "source_row",
    "operation_date",
    "vehicle",
    "vehicle_type",
    "vehicle_group",
    "activity_mode",
    "lifting_hours",
    "total_distance_km",
    "distance_source",
    "prepared_fuel_liters",
    "fuel_stick_km",
    "gps_km",
    "gps_drive_hours",
    "stop_sequence",
    "unresolved_stops",
    "activity_text",
    "trip_text",
    "actual_fuel_liters",
    "actual_km",
)


# --- parsing the cells -----------------------------------------------------------


_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?")


def parse_number(text: object) -> float | None:
    """Lenient: '80.8.' is 80.8, '1,5' is 1.5, blanks and words are None.

    The sheets are typed by hand. A trailing dot or a comma decimal is a
    typo, not a different number; a word in a numeric column is not a number.
    """
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", ".")
    if not cleaned or cleaned.lower() in {"nan", "none", "-"}:
        return None
    match = _NUMBER.match(cleaned)
    if match is None:
        return None
    value = float(match.group(0))
    return value


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


def parse_date(text: object) -> date | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned.lower() == "nan":
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    return None


_DURATION = re.compile(r"^(\d+):(\d{2}):(\d{2})$")


def parse_duration_hours(text: object) -> float | None:
    """'04:30:52' from the tracker is 4.51 hours; anything else is None."""
    if text is None:
        return None
    match = _DURATION.match(str(text).strip())
    if match is None:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return round(hours + minutes / 60 + seconds / 3600, 2)


def derive_activity_mode(distance_km: float | None, lifting_hours: float | None) -> str | None:
    """From the numbers, not the free text: what the day contained."""
    moved = distance_km is not None and distance_km > 0
    lifted = lifting_hours is not None and lifting_hours > 0
    if moved and lifted:
        return "transport_and_lifting"
    if lifted:
        return "lifting"
    if moved:
        return "transport"
    return None


_TRIP_SEPARATORS = re.compile(r"\s+-\s+|\s*[,;–]\s*|\s+-|-\s+")
_NOTE = re.compile(r"\(.*?\)")


def split_trip(text: object) -> list[str]:
    """Stops as the planner wrote them, in order, notes in brackets removed.

    Split on spaced hyphens, commas and semicolons only: a bare hyphen is
    part of many names ("KRG-12", "Bel-50"), so it is left alone here and
    tried as a second-chance split in `resolve_stops` when a token is
    unknown.
    """
    if text is None:
        return []
    cleaned = _NOTE.sub(" ", str(text)).strip()
    if not cleaned or cleaned.lower() == "nan":
        return []
    return [token.strip(" .") for token in _TRIP_SEPARATORS.split(cleaned) if token.strip(" .")]


@dataclass(frozen=True, slots=True)
class ResolvedStops:
    resolved: tuple[str, ...]
    unresolved: tuple[str, ...]


def resolve_stops(tokens: Iterable[str], catalogs: Catalogs) -> ResolvedStops:
    resolved: list[str] = []
    unresolved: list[str] = []
    for token in tokens:
        name = catalogs.location(token)
        if name is not None:
            resolved.append(name)
            continue
        # "Pool-KRG10-Pool": a bare-hyphen chain of known names.
        parts = [part for part in token.split("-") if part.strip()]
        names = [catalogs.location(part) for part in parts] if len(parts) > 1 else []
        if names and all(names):
            resolved.extend(name for name in names if name)
        else:
            unresolved.append(token)
    return ResolvedStops(tuple(resolved), tuple(unresolved))


# --- the operations table ----------------------------------------------------------


@dataclass(slots=True)
class Quarantine:
    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, sheet: str, row: int, reason: str, detail: str = "") -> None:
        self.rows.append(
            {"source_sheet": sheet, "source_row": row, "reason": reason, "detail": detail}
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["source_sheet", "source_row", "reason", "detail"])


def _has_work(row: pd.Series) -> bool:
    return any(
        str(row.get(column, "")).strip() for column in ("Liter", "D (Km)", "L (Jam)", "Kegiatan")
    )


def _vehicle_for(sheet: str, row: pd.Series, catalogs: Catalogs) -> str | None:
    written = str(row.get("Angber", "")).strip()
    if sheet == "TrontonWinch":
        return catalogs.vehicle(written) if written else None
    return catalogs.vehicle(SHEET_VEHICLES.get(sheet, sheet))


def tidy_operations(
    raw_directory: Path, version: str, catalogs: Catalogs
) -> tuple[pd.DataFrame, Quarantine]:
    quarantine = Quarantine()
    records: list[dict[str, object]] = []
    for sheet in OPERATION_SHEETS:
        frame = raw_frame(raw_directory, sheet)
        for _, row in frame.iterrows():
            source_row = int(row["source_row"])
            if not _has_work(row):
                continue  # a blank calendar day is not a row of anything
            activity_text = str(row.get("Kegiatan", "")).strip()
            if activity_text.strip().casefold() == STANDBY:
                quarantine.add(sheet, source_row, STANDBY, activity_text)
                continue
            operation_date = parse_date(row.get("Tanggal"))
            if operation_date is None:
                quarantine.add(sheet, source_row, NO_DATE, str(row.get("Tanggal", "")))
                continue
            vehicle = _vehicle_for(sheet, row, catalogs)
            if vehicle is None:
                quarantine.add(sheet, source_row, VEHICLE_UNKNOWN, str(row.get("Angber", "")))
                continue
            liters = parse_number(row.get("Liter"))
            if liters is None or liters <= 0:
                quarantine.add(sheet, source_row, NO_FUEL_ISSUED, str(row.get("Liter", "")))
                continue
            distance = parse_number(row.get("D (Km)"))
            hours = parse_number(row.get("L (Jam)"))
            if (distance is not None and distance > MAX_DISTANCE_KM) or (
                hours is not None and hours > MAX_LIFTING_HOURS
            ):
                quarantine.add(sheet, source_row, IMPLAUSIBLE, f"km={distance} jam={hours}")
                continue
            mode = derive_activity_mode(distance, hours)
            if mode is None:
                quarantine.add(
                    sheet,
                    source_row,
                    FUEL_WITHOUT_WORK,
                    f"liter={liters} km={row.get('D (Km)', '')!s} jam={row.get('L (Jam)', '')!s}",
                )
                continue
            stops = resolve_stops(split_trip(row.get("Trip")), catalogs)
            lineage = catalogs.lineage(vehicle)
            fuel_stick_km = parse_number(_fuel_stick_column(row))
            gps_km = repair_number(row.get("Mileage Summary(KM)"))
            gps_drive_hours = parse_duration_hours(row.get("Drive(hh:mm:ss)"))
            records.append(
                {
                    "dataset_version": version,
                    "source_sheet": sheet,
                    "source_row": source_row,
                    "operation_date": operation_date.isoformat(),
                    "vehicle": vehicle,
                    "vehicle_type": lineage.type,
                    "vehicle_group": lineage.group,
                    "activity_mode": mode,
                    "lifting_hours": hours if hours and hours > 0 else None,
                    "total_distance_km": distance if distance and distance > 0 else None,
                    "distance_source": "manual",
                    "prepared_fuel_liters": liters,
                    "fuel_stick_km": fuel_stick_km,
                    # Telematics, when the export carries it: an objective
                    # distance beside the typed one, never in its place.
                    "gps_km": gps_km if gps_km and gps_km > 0 else None,
                    "gps_drive_hours": gps_drive_hours,
                    "stop_sequence": " → ".join(stops.resolved),
                    "unresolved_stops": " | ".join(stops.unresolved),
                    "activity_text": activity_text,
                    "trip_text": str(row.get("Trip", "")).strip(),
                    "actual_fuel_liters": None,
                    "actual_km": None,
                }
            )
    operations = pd.DataFrame(records, columns=list(OPERATION_COLUMNS))
    operations = operations.sort_values(["operation_date", "vehicle", "source_row"]).reset_index(
        drop=True
    )
    return operations, quarantine


def _fuel_stick_column(row: pd.Series) -> object:
    for column in row.index:
        if str(column).startswith("Jarak") and "FS" in str(column):
            return row[column]
    return None


# --- actuals: the fuel stick ---------------------------------------------------------

_VT_CODE = re.compile(r"^VT[-\s]?(\d+)", re.IGNORECASE)
_TC_CODE = re.compile(r"^TC[-\s]?(\d+)", re.IGNORECASE)
_FUEL_STICK_NAMES = {
    "whell crane kato": "Wheel Crane",
    "winch truck": "Winch Truck",
    "prime mover": "Prime Mover",
    "truck crane 01": "Truck Crane 01",
    "truck crane 02": "Truck Crane 02",
}


def fuel_stick_vehicle(code: object, catalogs: Catalogs) -> str | None:
    """'VT-05 BG 8798 DR' is VT 05; 'TC-02 BG 8285 DO' is Truck Crane 02: the
    code before the number plate names the unit."""
    written = str(code or "").strip()
    if not written:
        return None
    if match := _VT_CODE.match(written):
        return catalogs.vehicle(f"VT {int(match.group(1)):02d}")
    if match := _TC_CODE.match(written):
        return catalogs.vehicle(f"Truck Crane {int(match.group(1)):02d}")
    for prefix, name in _FUEL_STICK_NAMES.items():
        if written.casefold().startswith(prefix):
            return catalogs.vehicle(name)
    return catalogs.vehicle(written)


def tidy_actuals(raw_directory: Path, version: str, catalogs: Catalogs) -> pd.DataFrame:
    frame = raw_frame(raw_directory, "Data Fuel Stick")
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        vehicle = fuel_stick_vehicle(row.get("Vehicle Code"), catalogs)
        day = parse_date(row.get("Date"))
        if vehicle is None or day is None:
            continue
        consumption = parse_number(row.get("Fuel Consumption"))
        records.append(
            {
                "dataset_version": version,
                "source_row": int(row["source_row"]),
                "vehicle": vehicle,
                "vehicle_code": str(row.get("Vehicle Code", "")).strip(),
                "reading_date": day.isoformat(),
                "mileage_km": parse_number(row.get("Mileage (KM)")),
                "initial_fuel_liters": parse_number(row.get("Initial Fuel (L)")),
                "finish_fuel_liters": parse_number(row.get("Finish Fuel (L)")),
                "refuel_liters": parse_number(row.get("Refuel (L)")),
                "drain_liters": parse_number(row.get("Drain (L)")),
                "actual_fuel_liters": consumption,
                # A negative reading is a measurement problem, kept and flagged.
                "suspect": consumption is not None and consumption < 0,
            }
        )
    return pd.DataFrame(records)


def join_actuals(operations: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """For the report: what the fuel stick measured on the same unit-day.
    Never the label - prepared and actual are different quantities (ADR 0002)."""
    if actuals.empty or operations.empty:
        return operations
    readings = (
        actuals[~actuals["suspect"]]
        .groupby(["vehicle", "reading_date"], as_index=False)
        .agg(actual_fuel_liters=("actual_fuel_liters", "sum"), actual_km=("mileage_km", "sum"))
        .rename(columns={"reading_date": "operation_date"})
    )
    merged = operations.drop(columns=["actual_fuel_liters", "actual_km"]).merge(
        readings, on=["vehicle", "operation_date"], how="left"
    )
    return merged[list(OPERATION_COLUMNS)]


# --- the app's import file ----------------------------------------------------------

APP_IMPORT_COLUMNS = (
    "Tanggal",
    "Kategori ANGBER",
    "Kendaraan",
    "Mode Aktivitas",
    "Jam Lifting",
    "Jarak Total (km)",
    "Bahan Bakar Disiapkan (L)",
    "Sumber Jarak",
)


def app_import(operations: pd.DataFrame, quarantine: Quarantine) -> pd.DataFrame:
    """The CSV production imports as history. Production's domain needs a
    distance > 0, so lifting-only days are left out of this file (and only
    this file) with a quarantine reason."""
    rows: list[dict[str, object]] = []
    for _, row in operations.iterrows():
        if row["total_distance_km"] is None or pd.isna(row["total_distance_km"]):
            quarantine.add(str(row["source_sheet"]), int(row["source_row"]), LIFTING_ONLY)
            continue
        hours = row["lifting_hours"]
        rows.append(
            {
                "Tanggal": row["operation_date"],
                "Kategori ANGBER": "ANGBER",
                "Kendaraan": row["vehicle"],
                "Mode Aktivitas": row["activity_mode"],
                "Jam Lifting": "" if hours is None or pd.isna(hours) else hours,
                "Jarak Total (km)": row["total_distance_km"],
                "Bahan Bakar Disiapkan (L)": row["prepared_fuel_liters"],
                "Sumber Jarak": "manual",
            }
        )
    return pd.DataFrame(rows, columns=list(APP_IMPORT_COLUMNS))


# --- the run ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TidyResult:
    version: str
    directory: Path
    operations: pd.DataFrame
    actuals: pd.DataFrame
    quarantine: pd.DataFrame
    app_rows: int


def tidy(raw_directory: Path, tidy_root: Path, catalogs: Catalogs) -> TidyResult:
    version = raw_directory.name
    operations, quarantine = tidy_operations(raw_directory, version, catalogs)
    actuals = tidy_actuals(raw_directory, version, catalogs)
    operations = join_actuals(operations, actuals)
    importable = app_import(operations, quarantine)
    directory = tidy_root / version
    directory.mkdir(parents=True, exist_ok=True)
    operations.to_csv(directory / "operations.csv", index=False)
    actuals.to_csv(directory / "actuals.csv", index=False)
    quarantine_frame = quarantine.frame()
    quarantine_frame.to_csv(directory / "quarantine.csv", index=False)
    importable.to_csv(directory / "app-import.csv", index=False)
    (directory / "report.md").write_text(
        report(version, operations, actuals, quarantine_frame, len(importable), catalogs),
        encoding="utf-8",
    )
    return TidyResult(version, directory, operations, actuals, quarantine_frame, len(importable))


def report(
    version: str,
    operations: pd.DataFrame,
    actuals: pd.DataFrame,
    quarantine: pd.DataFrame,
    app_rows: int,
    catalogs: Catalogs,
) -> str:
    suspect = int(actuals["suspect"].sum()) if len(actuals) else 0
    lines = [
        f"# Tidy report — {version}",
        "",
        f"- Operations: **{len(operations)}** (app-importable: {app_rows})",
        f"- Quarantined: {len(quarantine)}",
        f"- Fuel-stick readings: {len(actuals)} (suspect: {suspect})",
        f"- Catalog fingerprint: `{catalogs.fingerprint()}`",
        "",
    ]
    if len(operations):
        lines += _units_section(operations)
        lines += _fuel_section(operations)
        lines += _gps_section(operations)
        lines += _actuals_section(operations)
        lines += _stops_section(operations)
    if len(quarantine):
        lines += ["## Quarantine by reason", "", "| Reason | Rows |", "|---|---|"]
        counts = quarantine["reason"].value_counts()
        lines += [f"| {reason} | {count} |" for reason, count in counts.items()]
        lines.append("")
    return "\n".join(lines)


def _table(title: str, header: list[str], rows: list[list[object]]) -> list[str]:
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return [*lines, ""]


def _units_section(operations: pd.DataFrame) -> list[str]:
    rows: list[list[object]] = []
    for (unit, vtype, group), part in operations.groupby(
        ["vehicle", "vehicle_type", "vehicle_group"]
    ):
        modes = part["activity_mode"].value_counts()
        rows.append(
            [
                unit,
                vtype,
                group,
                len(part),
                f"{part['operation_date'].min()} → {part['operation_date'].max()}",
                modes.get("transport", 0),
                modes.get("lifting", 0),
                modes.get("transport_and_lifting", 0),
                int((part["stop_sequence"] != "").sum()),
                int(part["actual_fuel_liters"].notna().sum()),
            ]
        )
    header = ["Unit", "Type", "Group", "Rows", "Dates", "Transport", "Lifting", "Both"]
    return _table("Operations by unit", [*header, "With stops", "With actuals"], rows)


def _fuel_section(operations: pd.DataFrame) -> list[str]:
    rows: list[list[object]] = []
    for unit, part in operations.groupby("vehicle"):
        fuel = part["prepared_fuel_liters"]
        rows.append(
            [
                unit,
                f"{fuel.min():.0f}",
                f"{fuel.median():.0f}",
                f"{fuel.mean():.1f}",
                f"{fuel.max():.0f}",
            ]
        )
    return _table("Prepared fuel per unit (litres)", ["Unit", "Min", "Median", "Mean", "Max"], rows)


def _gps_section(operations: pd.DataFrame) -> list[str]:
    both = operations.dropna(subset=["gps_km", "total_distance_km"])
    if both.empty:
        return []
    rows: list[list[object]] = []
    for unit, part in both.groupby("vehicle"):
        ratio = (part["gps_km"] / part["total_distance_km"]).median()
        rows.append([unit, len(part), f"{ratio:.2f}"])
    header = ["Unit", "Days with both", "GPS km ÷ typed km (median)"]
    return _table("Typed distance against the tracker, where both exist", header, rows)


def _actuals_section(operations: pd.DataFrame) -> list[str]:
    both = operations.dropna(subset=["actual_fuel_liters"])
    if both.empty:
        return []
    rows: list[list[object]] = []
    for unit, part in both.groupby("vehicle"):
        prepared = part["prepared_fuel_liters"].mean()
        measured = part["actual_fuel_liters"].mean()
        rows.append([unit, len(part), f"{prepared:.1f}", f"{measured:.1f}"])
    header = ["Unit", "Days", "Prepared (mean)", "Measured (mean)"]
    return _table("Prepared vs measured, where both exist", header, rows)


def _stops_section(operations: pd.DataFrame) -> list[str]:
    with_trip = int((operations["trip_text"] != "").sum())
    complete = (operations["stop_sequence"] != "") & (operations["unresolved_stops"] == "")
    cells = operations["unresolved_stops"].loc[lambda s: s != ""]
    tokens = pd.Series([token for cell in cells for token in str(cell).split(" | ")])
    lines = [
        "## Stops",
        "",
        f"- Rows with a trip text: {with_trip}",
        f"- Rows with every stop resolved: {int(complete.sum())}",
        f"- Distinct unresolved stop names: {tokens.nunique() if len(tokens) else 0}",
        "",
    ]
    if len(tokens):
        lines += ["Most frequent unresolved names (add to the location catalog, or an alias):", ""]
        lines += [f"- `{name}` × {count}" for name, count in tokens.value_counts().head(25).items()]
        lines.append("")
    return lines
