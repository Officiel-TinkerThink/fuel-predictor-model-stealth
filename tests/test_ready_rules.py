"""What gets erased before training, on small hand-made tables."""

import pandas as pd

from fuel_training.ready import (
    Removals,
    assign_split,
    implausible_units,
    issued_table,
    robust_outliers,
)


def _operations(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "vehicle_type": "Crane",
        "vehicle_group": "Crane",
        "activity_mode": "transport",
        "stop_sequence": "",
        "activity_text": "",
        "source_sheet": "Truck Crane 01",
        "source_row": 2,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_lifting_is_only_believed_on_the_three_lifting_units() -> None:
    operations = _operations(
        [
            {
                "operation_date": "2026-03-01",
                "vehicle": "Prime Mover",
                "total_distance_km": 40.0,
                "lifting_hours": 2.0,
                "prepared_fuel_liters": 50.0,
            },
            {
                "operation_date": "2026-03-01",
                "vehicle": "Truck Crane 01",
                "total_distance_km": 30.0,
                "lifting_hours": None,
                "prepared_fuel_liters": 40.0,
            },
            {
                "operation_date": "2026-03-02",
                "vehicle": "Truck Crane 01",
                "total_distance_km": 30.0,
                "lifting_hours": 2.0,
                "prepared_fuel_liters": 45.0,
            },
        ]
    )
    removals = Removals([])

    table = issued_table(operations, pd.DataFrame(columns=["vehicle", "date", "gps_km"]), removals)

    assert table["vehicle"].tolist() == ["Truck Crane 01"]
    assert sorted(removals.frame()["rule"]) == [
        "crane_without_lifting_hours",
        "lifting_on_non_lifting_unit",
    ]


def test_a_typed_distance_the_tracker_contradicts_is_erased() -> None:
    operations = _operations(
        [
            {
                "operation_date": "2026-03-01",
                "vehicle": "Prime Mover",
                "total_distance_km": 100.0,
                "lifting_hours": None,
                "prepared_fuel_liters": 80.0,
            },
            {
                "operation_date": "2026-03-02",
                "vehicle": "Prime Mover",
                "total_distance_km": 50.0,
                "lifting_hours": None,
                "prepared_fuel_liters": 45.0,
            },
        ]
    )
    gps = pd.DataFrame(
        {
            "vehicle": ["Prime Mover"] * 2,
            "date": ["2026-03-01", "2026-03-02"],
            "gps_km": [30.0, 48.0],
        }
    )
    removals = Removals([])

    table = issued_table(operations, gps, removals)

    assert table["date"].tolist() == ["2026-03-02"]
    assert removals.frame()["rule"].tolist() == ["typed_km_disagrees_with_tracker"]


def test_a_row_far_from_what_the_units_own_km_predict_is_an_outlier() -> None:
    km = list(range(10, 40))
    litres = [5 + 0.5 * k for k in km]
    litres[7] = 400.0
    frame = pd.DataFrame({"vehicle": "VT 01", "km": km, "litres": litres})

    marked = robust_outliers(frame, "litres", ["km"])

    assert marked.tolist().count(True) == 1 and bool(marked.iloc[7])


def test_each_units_latest_fifth_of_days_is_its_test_set() -> None:
    """A unit whose readings stop early still gets test days of its own."""
    early = pd.DataFrame(
        {"date": [f"2026-03-{d:02d}" for d in range(1, 11)], "vehicle": "Prime Mover"}
    )
    late = pd.DataFrame({"date": [f"2026-08-{d:02d}" for d in range(1, 11)], "vehicle": "VT 01"})

    split = assign_split(pd.concat([early, late], ignore_index=True))

    for unit in ("Prime Mover", "VT 01"):
        assert split.loc[split["vehicle"] == unit, "split"].tolist() == ["train"] * 8 + ["test"] * 2


def test_a_unit_measuring_far_off_its_group_is_erased_whole() -> None:
    rows = []
    for unit, km_per_litre in (("VT 01", 1.5), ("VT 02", 1.3), ("VT 03", 1.4), ("VT 11", 6.4)):
        for day in range(5):
            rows.append(
                {
                    "vehicle": unit,
                    "vehicle_group": "Vacuum Truck",
                    "km": 40.0,
                    "litres": 40.0 / km_per_litre,
                    "day": day,
                }
            )
    frame = pd.DataFrame(rows)

    marked = implausible_units(frame, "km", "litres")

    assert set(frame.loc[marked, "vehicle"]) == {"VT 11"}
