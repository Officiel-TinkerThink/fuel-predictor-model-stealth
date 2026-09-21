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
SUPPORT_SHEETS: dict[str, int] = {
    "Data Fuel Stick": 0,
    "Data Lokasi": 0,
    "Missing Data": 2,
    "Data Ratio": 2,
    "Dim_Kendaraan": 0,
    "Peta_Nama_Sumber": 0,
    "Fakta_BBM_Harian": 0,
}


@dataclass(frozen=True, slots=True)
class Extraction:
    version: str
    directory: Path
    sheets: tuple[str, ...]


def dataset_version(workbook: Path) -> str:
    """`DSV-<yyyymmdd>-<8 hex of the workbook's SHA-256>`: the same file always
    yields the same version, and a re-export of the workbook yields a new one."""
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    day = datetime.fromtimestamp(workbook.stat().st_mtime, tz=UTC).strftime("%Y%m%d")
    return f"DSV-{day}-{digest[:8]}"


def _sheet_frame(workbook: Path, sheet: str, header: int) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet, header=header, dtype=object)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    # 1-based row in the sheet, as a person would read it off Excel.
    frame.insert(0, "source_row", range(header + 2, header + 2 + len(frame)))
    return frame


def extract(workbook: Path, raw_root: Path) -> Extraction:
    version = dataset_version(workbook)
    directory = raw_root / version
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for sheet in OPERATION_SHEETS:
        frame = _sheet_frame(workbook, sheet, 0)
        frame.to_csv(directory / f"{_slug(sheet)}.csv", index=False)
        written.append(sheet)
    for sheet, header in SUPPORT_SHEETS.items():
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Extraction(version=version, directory=directory, sheets=tuple(written))


def _slug(sheet: str) -> str:
    return sheet.lower().replace(" - ", "-").replace(" ", "-").replace("_", "-")


def raw_frame(directory: Path, sheet: str) -> pd.DataFrame:
    return pd.read_csv(directory / f"{_slug(sheet)}.csv", dtype=object, keep_default_na=False)
