"""The rules that turn a sheet row into a training row, on a handful of values each."""

from pathlib import Path

import pandas as pd
import pytest

from fuel_training.catalog import Catalogs
from fuel_training.extract import _slug
from fuel_training.tidy import (
    FUEL_WITHOUT_WORK,
    IMPLAUSIBLE,
    NO_FUEL_ISSUED,
    STANDBY,
    VEHICLE_UNKNOWN,
    app_import,
    derive_activity_mode,
    fuel_stick_vehicle,
    parse_date,
    parse_number,
    repair_number,
    resolve_stops,
    split_trip,
    tidy_operations,
)


@pytest.fixture(scope="module")
def catalogs() -> Catalogs:
    return Catalogs.load()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("80.8.", 80.8),
        ("1,5", 1.5),
        (" 42 ", 42.0),
        ("", None),
        ("nan", None),
        ("STANDBY", None),
        (None, None),
    ],
)
def test_numbers_are_read_leniently(text: object, expected: float | None) -> None:
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["2026-02-09 00:00:00", "2026-02-09", "2/9/2026"])
def test_dates_in_the_sheets_forms_are_read(text: str) -> None:
    parsed = parse_date(text)
    assert parsed is not None
    assert parsed.isoformat() == "2026-02-09"


@pytest.mark.parametrize(
    ("km", "hours", "mode"),
    [
        (83, 1, "transport_and_lifting"),
        (None, 2, "lifting"),
        (55, None, "transport"),
        (0, 0, None),
        (None, None, None),
    ],
)
def test_the_activity_mode_comes_from_the_numbers(
    km: float | None, hours: float | None, mode: str | None
) -> None:
    assert derive_activity_mode(km, hours) == mode


def test_a_trip_text_splits_on_spaced_hyphens_and_commas_but_keeps_hyphenated_names() -> None:
    assert split_trip("Pool Limau - SKG.01 - SKG. 02 - Pool Limau") == [
        "Pool Limau",
        "SKG.01",
        "SKG. 02",
        "Pool Limau",
    ]
    assert split_trip("Pool-KRG10-KRG-19-Pool, Yard Limau (isi bbm)") == [
        "Pool-KRG10-KRG-19-Pool",
        "Yard Limau",
    ]
    assert split_trip("") == [] and split_trip(None) == []


def test_stops_resolve_through_the_production_catalog_and_report_what_did_not(
    catalogs: Catalogs,
) -> None:
    result = resolve_stops(["pool limau", "SP. II", "KRG 12", "tugu nanas"], catalogs)

    assert result.resolved == ("POOL LIMAU", "SP-II", "KRG-012")
    assert result.unresolved == ("tugu nanas",)


@pytest.mark.parametrize(
    ("code", "unit"),
    [
        ("VT-05 BG 8798 DR", "VT 05"),
        ("VT-09 Pilona BG 8085 DW", "VT 09"),
        ("VT-11", "VT 11"),
        ("TC-02 BG 8285 DO", "Truck Crane 02"),
        ("Whell Crane Kato", "Wheel Crane"),
        ("Winch Truck BG 8305 DR", "Winch Truck"),
        ("Prime Mover", "Prime Mover"),
        ("Bulldozer 07", None),
    ],
)
def test_fuel_stick_codes_name_the_unit_before_the_plate(
    code: str, unit: str | None, catalogs: Catalogs
) -> None:
    assert fuel_stick_vehicle(code, catalogs) == unit


def _raw_sheet(directory: Path, sheet: str, rows: list[dict[str, object]]) -> None:
    columns = [
        "source_row",
        "Tanggal",
        "Angber",
        "Liter",
        "D (Km)",
        "L (Jam)",
        "Kegiatan",
        "Trip",
        "Jarak - Data FS",
    ]
    frame = pd.DataFrame([{c: r.get(c, "") for c in columns} for r in rows], columns=columns)
    frame.to_csv(directory / f"{_slug(sheet)}.csv", index=False)


def test_rows_become_operations_or_are_quarantined_with_a_reason(
    tmp_path: Path, catalogs: Catalogs
) -> None:
    for sheet in ("Prime Mover", "Truck Crane 01", "Truck Crane 02", "Whellcrane"):
        _raw_sheet(tmp_path, sheet, [])
    _raw_sheet(
        tmp_path,
        "Truck Crane 01",
        [
            {
                "source_row": 2,
                "Tanggal": "2026-02-10",
                "Liter": "62",
                "D (Km)": "85",
                "L (Jam)": "1",
                "Kegiatan": "Lifting",
                "Trip": "Pool Limau - KRG 12 - Pool Limau",
            },
            {
                "source_row": 3,
                "Tanggal": "2026-02-11",
                "Liter": "25",
                "D (Km)": "",
                "L (Jam)": "2",
                "Kegiatan": "Lifting di WS",
            },
            {
                "source_row": 4,
                "Tanggal": "2026-02-12",
                "Liter": "",
                "D (Km)": "",
                "L (Jam)": "",
                "Kegiatan": "",
            },
            {
                "source_row": 5,
                "Tanggal": "2026-02-13",
                "Liter": "40",
                "D (Km)": "",
                "L (Jam)": "",
                "Kegiatan": "Standby",
            },
            {
                "source_row": 6,
                "Tanggal": "2026-02-14",
                "Liter": "40",
                "D (Km)": "",
                "L (Jam)": "",
                "Kegiatan": "Mobilisasi",
            },
            {
                "source_row": 7,
                "Tanggal": "2026-02-15",
                "Liter": "0",
                "D (Km)": "30",
                "L (Jam)": "",
                "Kegiatan": "Mobilisasi",
            },
            {
                "source_row": 8,
                "Tanggal": "2026-02-16",
                "Liter": "21",
                "D (Km)": "2026",
                "L (Jam)": "1",
                "Kegiatan": "Lifting",
            },
        ],
    )
    _raw_sheet(
        tmp_path,
        "TrontonWinch",
        [
            {
                "source_row": 2,
                "Tanggal": "2026-02-07",
                "Angber": "OFT Tronton",
                "Liter": "66",
                "D (Km)": "55.0",
                "Kegiatan": "Mobilisasi",
            },
            {
                "source_row": 3,
                "Tanggal": "2026-02-08",
                "Angber": "OFT Winch Truck",
                "Liter": "60",
                "D (Km)": "50",
                "Kegiatan": "Mobilisasi",
            },
            {
                "source_row": 4,
                "Tanggal": "2026-02-09",
                "Angber": "",
                "Liter": "10",
                "D (Km)": "5",
                "Kegiatan": "Mobilisasi",
            },
        ],
    )

    operations, quarantine = tidy_operations(tmp_path, "DSV-TEST", catalogs)

    assert operations["vehicle"].tolist() == [
        "Oil Field Truck",
        "Winch Truck",
        "Truck Crane 01",
        "Truck Crane 01",
    ]
    crane_day = operations.iloc[2]
    assert crane_day["activity_mode"] == "transport_and_lifting"
    assert crane_day["stop_sequence"] == "POOL LIMAU → KRG-012 → POOL LIMAU"
    assert (crane_day["vehicle_type"], crane_day["vehicle_group"]) == ("Crane", "Crane")
    lifting_only = operations.iloc[3]
    assert lifting_only["activity_mode"] == "lifting" and pd.isna(lifting_only["total_distance_km"])
    reasons = quarantine.frame().set_index("source_row")["reason"].to_dict()
    assert reasons == {
        5: STANDBY,
        6: FUEL_WITHOUT_WORK,
        7: NO_FUEL_ISSUED,
        8: IMPLAUSIBLE,
        4: VEHICLE_UNKNOWN,
    }

    importable = app_import(operations, quarantine)
    assert len(importable) == 3  # the lifting-only day has no distance for the app's domain
    assert list(importable.columns) == [
        "Tanggal",
        "Kategori ANGBER",
        "Kendaraan",
        "Mode Aktivitas",
        "Jam Lifting",
        "Jarak Total (km)",
        "Bahan Bakar Disiapkan (L)",
        "Sumber Jarak",
    ]
    assert quarantine.frame()["reason"].tolist().count("lifting_only") == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-05-30 00:00:00", 30.5),
        ("2026-04-01", 1.4),
        ("2026-12-03 00:00:00", 3.12),
        ("62.6", 62.6),
        ("-37", -37.0),
        ("", None),
    ],
)
def test_the_exports_decimal_to_date_damage_is_undone(text: str, expected: float | None) -> None:
    """Sheets read 30.5 as 30 May; the number is the day, a dot, the month."""
    assert repair_number(text) == expected
