"""The planners' rule, shaped as a model production accepts and serves.

Built on small hand-made ratios so the tests need no workbook. The last test
is the one that matters: the real application validates the package, an
administrator activates it, and a prediction for a planned operation is the
rule's own arithmetic.
"""

import json
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fuel_predictor.application.prediction_features import feature_values
from fuel_predictor.application.vehicles import VehicleLineage
from fuel_predictor.domain.daily_operation import (
    ActivityMode,
    DailyOperation,
    DistanceSource,
    VehicleCategory,
)
from fuel_predictor.infrastructure.packaged_vehicle_catalog import PackagedVehicleCatalog
from fuel_predictor.main import create_app

from fuel_training.planner import PlannerRatio
from fuel_training.rule_model import (
    FEATURE_SCHEMA,
    RuleTable,
    feature_frame,
    reserve_from,
    rule_pipeline,
)
from fuel_training.rule_package import build, onnx_predict, to_onnx

_RATIOS = {
    "Truck Crane 01": PlannerRatio(km_per_litre=2.4, litres_per_lifting_hour=10.6, safety_litres=1),
    "Prime Mover": PlannerRatio(km_per_litre=1.2, litres_per_lifting_hour=0.0, safety_litres=1),
    "VT 01": PlannerRatio(km_per_litre=1.5, litres_per_lifting_hour=0.0, safety_litres=1),
    "VT 09": PlannerRatio(km_per_litre=3.8, litres_per_lifting_hour=0.0, safety_litres=1),
}
_CASES = [
    ("Truck Crane 01", 30.0, 2.0),
    ("Prime Mover", 50.0, 0.0),
    ("VT 01", 45.0, 0.0),
    # In the catalog, not in the ratios: its group's median.
    ("VT 14", 40.0, 0.0),
    # Not in the catalog at all: the fleet's median, never zero.
    ("Crane Sewa", 20.0, 1.0),
]


def _table() -> RuleTable:
    return RuleTable.from_ratios(_RATIOS, PackagedVehicleCatalog().options())


def test_the_feature_schema_is_the_apps_feature_contract() -> None:
    operation = DailyOperation(
        operation_id="OPR-X",
        vehicle_category=VehicleCategory.ANGBER,
        vehicle="VT 01",
        activity_mode=ActivityMode.TRANSPORT,
        lifting_hours=None,
        total_distance_km=10,
        distance_source=DistanceSource.MANUAL,
    )

    contract = feature_values(operation, VehicleLineage.unknown())

    assert [name for name, _ in FEATURE_SCHEMA] == list(contract)


def test_a_unit_with_ratios_follows_the_planners_formula() -> None:
    table = _table()

    # 30 km / 2.4 km/L + 2 h x 10.6 L/h + 1 L safety
    assert table.litres("Truck Crane 01", 30, 2) == pytest.approx(12.5 + 21.2 + 1)


def test_units_without_ratios_fall_back_to_their_group_then_the_fleet() -> None:
    table = _table()

    vt14 = table.units["VT 14"]
    assert vt14.source == "Vacuum Truck median"
    assert vt14.litres_per_km == pytest.approx((1 / 1.5 + 1 / 3.8) / 2)
    assert table.rule_for("Crane Sewa").source == "fleet median"
    assert table.rule_for("Crane Sewa").litres_per_km > 0


def test_the_pipeline_is_the_rule_for_every_kind_of_unit() -> None:
    table = _table()
    frame = feature_frame(
        {"vehicle": unit, "total_distance_km": km, "lifting_hours": hours}
        for unit, km, hours in _CASES
    )

    predicted = rule_pipeline(table).predict(frame)

    assert list(predicted) == pytest.approx([table.litres(*case) for case in _CASES])


def test_the_exported_onnx_answers_like_the_rule() -> None:
    table = _table()
    frame = feature_frame(
        {"vehicle": unit, "total_distance_km": km, "lifting_hours": hours}
        for unit, km, hours in _CASES
    )

    predicted = onnx_predict(to_onnx(rule_pipeline(table)), frame)

    assert list(predicted) == pytest.approx([table.litres(*case) for case in _CASES], abs=1e-3)


def test_the_reserve_is_what_training_rows_were_issued_above_the_formula() -> None:
    table = _table()
    training = pd.DataFrame(
        {
            "vehicle": ["Prime Mover"] * 3 + ["VT 01"],
            "distance_km": [60.0, 60.0, 60.0, 30.0],
            "lifting_hours": [0.0] * 4,
            # Formula: Prime Mover 60/1.2+1 = 51; VT 01 30/1.5+1 = 21.
            "issued_litres": [61.0, 66.0, 71.0, 31.0],
        }
    )

    per_unit, fleet = reserve_from(training, table)
    reserved = table.with_reserve(per_unit, fleet)

    assert per_unit == {"Prime Mover": 15.0, "VT 01": 10.0}
    assert fleet == pytest.approx(12.5)
    assert reserved.litres("Prime Mover", 60, 0) == pytest.approx(66.0)
    assert reserved.litres("VT 14", 0, 0) == pytest.approx(table.litres("VT 14", 0, 0) + 12.5)


# --- the application accepts it ---------------------------------------------------

_ADMIN = ("admin", "kata-sandi-admin-1")


def _issued(table: RuleTable) -> pd.DataFrame:
    """Operations the rule gets right, 40 of them held out: enough for the
    app's minimum test set, and an MAE the promotion policy accepts."""
    rows = []
    for index in range(80):
        unit, km, hours = _CASES[index % 3]
        km += index % 7
        rows.append(
            {
                "vehicle": unit,
                "activity_mode": "transport_and_lifting" if hours else "transport",
                "distance_km": km,
                "lifting_hours": hours,
                "issued_litres": table.litres(unit, km, hours),
                "split": "test" if index >= 40 else "train",
            }
        )
    return pd.DataFrame(rows)


def _csrf(html: str) -> str:
    marker = 'name="csrf_token" value="'
    start = html.index(marker) + len(marker)
    return html[start : html.index('"', start)]


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FUEL_PREDICTOR_MODEL_ARTIFACT_DIRECTORY", str(tmp_path / "packages"))
    monkeypatch.setenv("FUEL_PREDICTOR_MLFLOW_TRACKING_DIRECTORY", str(tmp_path / "mlruns"))
    app = create_app(
        database_path=tmp_path / "app.sqlite3",
        bootstrap_administrator=_ADMIN,
        vehicle_catalog=PackagedVehicleCatalog(),
    )
    with TestClient(app) as client:
        client.post(
            "/masuk",
            data={
                "username": _ADMIN[0],
                "password": _ADMIN[1],
                "csrf_token": _csrf(client.get("/masuk").text),
            },
            follow_redirects=False,
        )
        yield client


def _package(table: RuleTable) -> bytes:
    return build(
        table,
        model_version="rule-data-ratio-test",
        dataset_version="DSV-TEST",
        issued=_issued(table),
        parameter_rows=len(_RATIOS),
        catalog_fingerprint="0" * 64,
        trained_at=datetime(2026, 9, 24, tzinfo=UTC),
    ).archive


def test_the_package_carries_the_apps_contract_and_a_smoke_case_per_unit() -> None:
    table = _table()

    with zipfile.ZipFile(BytesIO(_package(table))) as archive:
        manifest: dict[str, Any] = json.loads(archive.read("manifest.json"))
        smoke: dict[str, Any] = json.loads(archive.read("smoke-tests.json"))

    assert manifest["model_format"] == "onnx"
    assert manifest["feature_contract_version"] == "baseline-v2"
    assert manifest["test_set_size"] == 40
    assert len(smoke["cases"]) == len(table.units) + 1


def test_the_app_validates_activates_and_serves_the_rule(app_client: TestClient) -> None:
    table = _table()

    upload = app_client.post(
        "/model/unggah",
        files={"file": ("rule.zip", _package(table), "application/zip")},
        data={"csrf_token": _csrf(app_client.get("/model/unggah").text)},
    )
    assert upload.status_code == 201, upload.text
    page = app_client.get("/pengelolaan-model").text
    marker = 'action="/kandidat-model/'
    start = page.index(marker) + len(marker)
    model_version_id = page[start : page.index("/promosikan", start)]
    activation = app_client.post(
        f"/kandidat-model/{model_version_id}/promosikan",
        data={"csrf_token": _csrf(page)},
    )
    assert activation.status_code == 200, activation.text

    operation = app_client.post(
        "/api/v1/daily-operations",
        json={
            "vehicle_category": "ANGBER",
            "vehicle": "Truck Crane 01",
            "activity_mode": "transport_and_lifting",
            "lifting_hours": 2,
            "total_distance_km": 30,
            "distance_source": "manual",
        },
    ).json()
    prediction = app_client.post(
        f"/api/v1/daily-operations/{operation['operation_id']}/predictions"
    )

    assert prediction.status_code == 201, prediction.text
    body = prediction.json()
    assert body["model"]["model_version_id"] == "rule-data-ratio-test"
    assert body["estimated_fuel_requirement_liters"] == pytest.approx(
        table.litres("Truck Crane 01", 30, 2), abs=0.01
    )
