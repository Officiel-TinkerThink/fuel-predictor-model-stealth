"""The rule model as a production package: ONNX, honest metrics, reference
statistics, smoke tests - built by the app's own `ModelPackageBuilder`.

Metrics come from the held-out split of `train-issued.csv` and are computed
with the app's own `calculate_performance_metrics`, so the ± band production
shows next to every estimate (the package's MAE) is what the rule actually
achieved on operations it never saw.
"""

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
from fuel_predictor.application.actual_fuel import (
    PerformanceMetrics,
    PredictionOutcome,
    calculate_performance_metrics,
)
from fuel_predictor.domain.daily_operation import VehicleCategory
from fuel_predictor.packaging.model_packager import ModelPackageBuilder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline

from fuel_training.rule_model import (
    FEATURE_CONTRACT,
    FEATURE_SCHEMA,
    RuleTable,
    feature_frame,
    rule_pipeline,
)

# What production's runtime check expects (its supported list), and an opset
# every onnxruntime from 1.20 on loads.
RUNTIME_COMPATIBILITY = "onnxruntime-1.20"
_OPSETS = {"": 17, "ai.onnx.ml": 3}
# An ONNX session for a few hundred linear terms stays far below this; the
# figure only feeds production's activation capacity check.
_EXPECTED_MEMORY_BYTES = 64 * 1024 * 1024
_SMOKE_KM = 40.0
_SMOKE_HOURS = 2.0


@dataclass(frozen=True, slots=True)
class RulePackage:
    model_version: str
    archive: bytes
    metrics: PerformanceMetrics
    test_rows: pd.DataFrame


def to_onnx(pipeline: Pipeline) -> bytes:
    inputs = [
        (name, StringTensorType([None, 1]) if kind == "string" else FloatTensorType([None, 1]))
        for name, kind in FEATURE_SCHEMA
    ]
    model = convert_sklearn(pipeline, initial_types=inputs, target_opset=_OPSETS)
    return bytes(model.SerializeToString())


def onnx_predict(model_bytes: bytes, frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Score with the exported artefact itself, one named input per feature -
    the way production's loader feeds it."""
    session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feeds: dict[str, Any] = {}
    for name, kind in FEATURE_SCHEMA:
        column = frame[name].to_numpy().reshape(-1, 1)
        feeds[name] = column.astype(object) if kind == "string" else column.astype(np.float32)
    return np.asarray(session.run(None, feeds)[0], dtype=np.float64).reshape(-1)


def issued_features(issued: pd.DataFrame) -> pd.DataFrame:
    """`train-issued.csv` rows in the app's feature contract."""
    return feature_frame(
        {
            "vehicle": unit,
            "activity_mode": mode,
            "total_distance_km": km,
            "lifting_hours": hours,
        }
        for unit, mode, km, hours in zip(
            issued["vehicle"].astype(str),
            issued["activity_mode"].astype(str),
            issued["distance_km"].to_numpy(dtype=float),
            issued["lifting_hours"].to_numpy(dtype=float),
            strict=True,
        )
    )


def evaluate(
    model_bytes: bytes, issued_test: pd.DataFrame
) -> tuple[PerformanceMetrics, pd.DataFrame]:
    """Metrics on the held-out rows, with the interval production will show
    (estimate ± the package's MAE), computed by production's own code."""
    predicted = onnx_predict(model_bytes, issued_features(issued_test))
    actual = issued_test["issued_litres"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(predicted - actual)))
    outcomes = [
        PredictionOutcome(
            vehicle_category=VehicleCategory.ANGBER,
            estimated_fuel_requirement_liters=float(estimate),
            uncertainty_lower_liters=max(0.0, float(estimate) - mae),
            uncertainty_upper_liters=float(estimate) + mae,
            actual_fuel_liters=float(litres),
        )
        for estimate, litres in zip(predicted, actual, strict=True)
    ]
    scored = issued_test.assign(predicted_litres=predicted, error=predicted - actual)
    return calculate_performance_metrics(outcomes), scored


def reference_statistics(features: pd.DataFrame) -> dict[str, Any]:
    """The drift baseline: what the rule is expected to be asked about."""
    summary: dict[str, Any] = {}
    for name, kind in FEATURE_SCHEMA:
        column = features[name]
        if kind == "string":
            shares = column.value_counts(normalize=True)
            summary[name] = {
                "kind": "categorical",
                "frequencies": {str(level): float(share) for level, share in shares.items()},
            }
        else:
            values = column.astype(float)
            summary[name] = {
                "kind": "numeric",
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=0)),
                "quantiles": {q: float(values.quantile(float(q))) for q in ("0.25", "0.5", "0.75")},
            }
    return {"row_count": len(features), "features": summary}


def smoke_tests(table: RuleTable) -> dict[str, Any]:
    """One case per unit and one for a name nobody knows. The expected value
    is the rule's arithmetic, not the model's output: a package whose
    coefficients drifted from the rule fails production's replay."""
    cases = []
    for unit in [*sorted(table.units), "Unit tidak dikenal"]:
        hours = _SMOKE_HOURS if table.rule_for(unit).litres_per_hour > 0 else 0.0
        features = feature_frame(
            [
                {
                    "vehicle": unit,
                    "activity_mode": "transport_and_lifting" if hours else "transport",
                    "total_distance_km": _SMOKE_KM,
                    "lifting_hours": hours,
                }
            ]
        ).iloc[0]
        cases.append(
            {
                "name": f"{unit}: {_SMOKE_KM:g} km, {hours:g} jam lifting",
                "features": {name: _plain(features[name]) for name, _ in FEATURE_SCHEMA},
                "expected_prediction": round(table.litres(unit, _SMOKE_KM, hours), 4),
                "tolerance": 0.01,
            }
        )
    return {"cases": cases}


def build(
    table: RuleTable,
    *,
    model_version: str,
    dataset_version: str,
    issued: pd.DataFrame,
    parameter_rows: int,
    catalog_fingerprint: str,
    trained_at: datetime | None = None,
) -> RulePackage:
    """The package, from a rule table and the ready `train-issued.csv`.

    `parameter_rows` is what `training_row_count` reports: the rows the rule's
    numbers came from (the ratio rows, plus the training split if a reserve was
    taken from it). Nothing was fitted on the test split.
    """
    test = issued[issued["split"] == "test"]
    train = issued[issued["split"] == "train"]
    if test.empty:
        raise ValueError("train-issued.csv has no test rows; there is nothing to evaluate on.")
    model_bytes = to_onnx(rule_pipeline(table))
    metrics, scored = evaluate(model_bytes, test)
    overall = _metrics(metrics)
    builder = ModelPackageBuilder(
        model_version=model_version,
        model_format="onnx",
        runtime_compatibility_version=RUNTIME_COMPATIBILITY,
        feature_contract_version=FEATURE_CONTRACT,
        feature_schema=[{"name": name, "type": kind} for name, kind in FEATURE_SCHEMA],
        target_name="prepared_fuel_liters",
        target_unit="liters",
        training_dataset_version=dataset_version,
        trained_at=trained_at or datetime.now(UTC),
        source_revision=_source_revision(),
        metrics={"overall": overall, "by_category": [{"category": "ANGBER", **overall}]},
        test_set_size=len(test),
        training_row_count=parameter_rows,
        expected_memory_bytes=_EXPECTED_MEMORY_BYTES,
        catalog_fingerprint=catalog_fingerprint,
    )
    archive = builder.build(
        model_bytes,
        reference_statistics(issued_features(train if not train.empty else test)),
        smoke_tests(table),
    )
    return RulePackage(
        model_version=model_version, archive=archive, metrics=metrics, test_rows=scored
    )


def _metrics(metrics: PerformanceMetrics) -> dict[str, float]:
    values = {
        "mae": metrics.mae_liters,
        "rmse": metrics.rmse_liters,
        "smape_percent": metrics.smape_percent,
        "interval_coverage_percent": metrics.interval_coverage_percent,
    }
    if any(value is None for value in values.values()):
        raise ValueError("No test rows could be scored.")
    return {key: round(float(value), 4) for key, value in values.items() if value is not None}


def _plain(value: object) -> str | float:
    return float(value) if isinstance(value, (int, float, np.number)) else str(value)


def _source_revision() -> str:
    """The commit this package was built from, marked when the tree was dirty."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def write(package: RulePackage, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{package.model_version}.zip"
    path.write_bytes(package.archive)
    return path


def report(table: RuleTable, package: RulePackage) -> str:
    """What the rule is, unit by unit, and how it did on the held-out rows."""
    metrics = package.metrics
    lines = [
        f"# {package.model_version}",
        "",
        "The planners' `Data Ratio` rule packaged as a production model (docs/rule-model.md).",
        "",
        f"Held-out operations: {len(package.test_rows)} · MAE {metrics.mae_liters:.2f} L · "
        f"RMSE {metrics.rmse_liters:.2f} L · sMAPE {metrics.smape_percent:.1f} % · "
        f"within ± MAE {metrics.interval_coverage_percent:.0f} % · "
        f"bias {package.test_rows['error'].mean():+.2f} L",
        "",
        "Only units with issued-litre operations are in the held-out rows. A unit's reserve",
        "marked *own* comes from its training rows; *reserve* alone is the fleet's median gap,",
        "borrowed - the vacuum trucks have no issued operations yet, so theirs is unverified.",
        "",
        "## The rule per unit",
        "",
        "| Unit | L per km | L per lifting hour | Base L | From |",
        "|---|---|---|---|---|",
    ]
    for unit in sorted(table.units):
        rule = table.units[unit]
        lines.append(
            f"| {unit} | {rule.litres_per_km:.3f} | {rule.litres_per_hour:g} | "
            f"{rule.base_litres:.1f} | {rule.source} |"
        )
    default = table.default
    lines.append(
        f"| *(unit not in the catalog)* | {default.litres_per_km:.3f} | "
        f"{default.litres_per_hour:g} | {default.base_litres:.1f} | {default.source} |"
    )
    per_unit = package.test_rows.groupby("vehicle")["error"].agg(
        rows="count", bias="mean", mae=lambda error: error.abs().mean()
    )
    lines += [
        "",
        "## Held-out error per unit",
        "",
        "| Unit | Rows | Bias L | MAE L |",
        "|---|---|---|---|",
        *(
            f"| {unit} | {int(row.rows)} | {row.bias:+.1f} | {row.mae:.1f} |"
            for unit, row in per_unit.iterrows()
        ),
        "",
    ]
    return "\n".join(lines)
