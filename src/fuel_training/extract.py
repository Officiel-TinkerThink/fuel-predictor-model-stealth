"""Stage 1: the workbook, sheet by sheet, into CSVs that still say where each value came from.

Values are written as they are in the cells. Interpretation - what is a
number, which sheet is which unit - is stage 2's job, so that a rule can be
changed and re-run without touching the workbook again.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# The sheets stage 2 reads, and the header row of each (0-based). The
# per-unit sheets carry an empty first column, which pandas names "Unnamed: 0".
OPERATION_SHEETS: tuple[str, ...] = (
    "Prime Mover",
    "Truck Crane 01",
    "Truck Crane 02",
    "Whellcrane",
    "TrontonWinch",
)
# Sheets that come and go between exports: the client's live workbook carries
# the telematics summary (VTS) but not the lookup tables the earlier file had.
# Each is extracted when present and listed as missing otherwise; stage 2 only
# requires "Data Fuel Stick".
SUPPORT_SHEETS: dict[str, int] = {
    "Data Fuel Stick": 0,
    "Data Lokasi": 0,
    "Missing Data": 2,
    "Data Ratio": 2,
    "VTS": 0,
    "Mentah - Data Fuel Stick": 0,
    # Header rows the sheet does not start with: found by content (see
    # _header_row), or kept positional (-1) when a sheet is several side-by-side
    # tables, as Analisis is.
    "Data GPS": -2,
    "Analisis": -1,
    "Dim_Kendaraan": 0,
    "Peta_Nama_Sumber": 0,
    "Fakta_BBM_Harian": 0,
}


@dataclass(frozen=True, slots=True)
class Extraction:
    version: str
    directory: Path
    sheets: tuple[str, ...]
    missing: tuple[str, ...] = ()


def dataset_version(workbook: Path) -> str:
    """`DSV-<yyyymmdd>-<8 hex of the workbook's SHA-256>`: the same file always
    yields the same version, and a re-export of the workbook yields a new one."""
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    day = datetime.fromtimestamp(workbook.stat().st_mtime, tz=UTC).strftime("%Y%m%d")
    return f"DSV-{day}-{digest[:8]}"


def _header_row(workbook: Path, sheet: str, marker: str = "Vehicle Code") -> int:
    probe = pd.read_excel(workbook, sheet_name=sheet, header=None, dtype=object, nrows=50)
    for position in range(len(probe)):
        if any(str(value).strip() == marker for value in probe.iloc[position].tolist()):
            return position
    raise LookupError(f"No header row containing {marker!r} in sheet {sheet!r}.")


def _sheet_frame(workbook: Path, sheet: str, header: int) -> pd.DataFrame:
    if header == -1:
        frame = pd.read_excel(workbook, sheet_name=sheet, header=None, dtype=object)
        frame.columns = [f"c{index}" for index in range(frame.shape[1])]
        frame.insert(0, "source_row", range(1, 1 + len(frame)))
        return frame
    if header == -2:
        header = _header_row(workbook, sheet)
    frame = pd.read_excel(workbook, sheet_name=sheet, header=header, dtype=object)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    # 1-based row in the sheet, as a person would read it off Excel.
    frame.insert(0, "source_row", range(header + 2, header + 2 + len(frame)))
    return frame


def extract(workbook: Path, raw_root: Path) -> Extraction:
    version = dataset_version(workbook)
    directory = raw_root / version
    directory.mkdir(parents=True, exist_ok=True)
    present = set(pd.ExcelFile(workbook).sheet_names)
    absent = [sheet for sheet in OPERATION_SHEETS if sheet not in present]
    if absent:
        raise LookupError(f"Workbook lacks the operation sheets {absent}; nothing to train on.")
    written: list[str] = []
    missing: list[str] = []
    for sheet in OPERATION_SHEETS:
        frame = _sheet_frame(workbook, sheet, 0)
        frame.to_csv(directory / f"{_slug(sheet)}.csv", index=False)
        written.append(sheet)
    for sheet, header in SUPPORT_SHEETS.items():
        if sheet not in present:
            missing.append(sheet)
            continue
        frame = _sheet_frame(workbook, sheet, header)
        frame.to_csv(directory / f"{_slug(sheet)}.csv", index=False)
        written.append(sheet)
    (directory / "source.json").write_text(
        json.dumps(
            {
                "dataset_version": version,
                "workbook": workbook.name,
                "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                "workbook_bytes": workbook.stat().st_size,
                "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sheets": written,
                "missing_sheets": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Extraction(
        version=version, directory=directory, sheets=tuple(written), missing=tuple(missing)
    )


def _slug(sheet: str) -> str:
    return sheet.lower().replace(" - ", "-").replace(" ", "-").replace("_", "-")


def raw_frame(directory: Path, sheet: str) -> pd.DataFrame:
    return pd.read_csv(directory / f"{_slug(sheet)}.csv", dtype=object, keep_default_na=False)
