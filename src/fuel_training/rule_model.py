"""The planners' `Data Ratio` rule, shaped as the model production serves.

    litres = base(unit) + km x litres_per_km(unit) + lifting_hours x litres_per_hour(unit)

`litres_per_km` is 1 / the sheet's drive ratio (km per litre), `litres_per_hour`
its lift ratio, and `base` its safety litres plus - optionally - a reserve: the
litres planners typically issue on top of the formula, taken from the training
split (docs/rule-model.md).

The rule is built as exactly the pipeline a trained candidate will be:

    vehicle one-hot + km + hours -> pairwise interactions -> linear regression

with the regression's coefficients *set* from the rule instead of fitted. The
interaction of "is VT 01" with km carries VT 01's own litres per km, and so on.
Production cannot tell it from a trained model: same feature contract, same
ONNX shape, same package. Moving to a trained model later is `fit` on this
pipeline and nothing else changes.
"""

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from fuel_predictor.application.prediction_features import FEATURE_VERSION
from fuel_predictor.application.vehicles import UNKNOWN_VEHICLE, VehicleOption
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from fuel_training.planner import PlannerRatio

# The app's feature contract (`baseline-v2`), in the order production feeds it.
# Imported names would be nicer, but the contract is a function, not a table;
# tests/test_rule_model.py asserts this list against it.
FEATURE_CONTRACT = FEATURE_VERSION
FEATURE_SCHEMA: tuple[tuple[str, str], ...] = (
    ("vehicle_category", "string"),
    ("vehicle", "string"),
    ("activity_mode", "string"),
    ("distance_source", "string"),
    ("total_distance_km", "number"),
    ("lifting_hours", "number"),
)
_KM = "total_distance_km"
_HOURS = "lifting_hours"


@dataclass(frozen=True, slots=True)
class UnitRule:
    """One unit's rule, and where its numbers came from."""

    litres_per_km: float
    litres_per_hour: float
    base_litres: float
    source: str

    def litres(self, km: float, lifting_hours: float) -> float:
        return self.base_litres + km * self.litres_per_km + lifting_hours * self.litres_per_hour


@dataclass(frozen=True, slots=True)
class RuleTable:
    """Every catalog unit's rule, plus the one used for a unit nobody knows.

    Units the `Data Ratio` tab lists use their own ratios. Units it does not
    (VT 14, the forklifts) use their group's median, and a name that is not in
    the catalog at all uses the fleet's median - never zero litres per km.
    """

    units: Mapping[str, UnitRule]
    default: UnitRule

    def rule_for(self, vehicle: str) -> UnitRule:
        return self.units.get(vehicle, self.default)

    def litres(self, vehicle: str, km: float, lifting_hours: float) -> float:
        return self.rule_for(vehicle).litres(km, lifting_hours)

    @classmethod
    def from_ratios(
        cls, ratios: Mapping[str, PlannerRatio], fleet: Iterable[VehicleOption]
    ) -> "RuleTable":
        fleet = tuple(fleet)
        own = {
            unit: UnitRule(
                litres_per_km=1 / ratio.km_per_litre,
                litres_per_hour=ratio.litres_per_lifting_hour,
                base_litres=ratio.safety_litres,
                source="Data Ratio",
            )
            for unit, ratio in ratios.items()
        }
        if not own:
            raise ValueError("The Data Ratio tab gave no usable ratio; there is no rule to build.")
        default = _median_rule(own.values(), "fleet median")
        units = dict(own)
        for option in fleet:
            if option.name in units:
                continue
            kin = [own[o.name] for o in fleet if o.group == option.group and o.name in own]
            units[option.name] = (
                _median_rule(kin, f"{option.group} median") if kin else replace(default)
            )
        return cls(units=units, default=default)

    def with_reserve(self, reserve: Mapping[str, float], fallback: float) -> "RuleTable":
        """Add a reserve to every unit's base: its own where one is known,
        `fallback` otherwise (and for a unit nobody knows)."""
        return RuleTable(
            units={
                unit: replace(
                    rule,
                    base_litres=rule.base_litres + reserve.get(unit, fallback),
                    source=rule.source + (" + own reserve" if unit in reserve else " + reserve"),
                )
                for unit, rule in self.units.items()
            },
            default=replace(
                self.default,
                base_litres=self.default.base_litres + fallback,
                source="fleet median + reserve",
            ),
        )


def _median_rule(rules: Iterable[UnitRule], source: str) -> UnitRule:
    rules = list(rules)
    lifting = [rule.litres_per_hour for rule in rules if rule.litres_per_hour > 0]
    return UnitRule(
        litres_per_km=statistics.median(rule.litres_per_km for rule in rules),
        # A group that does not lift keeps zero; a unit nobody knows is priced
        # like the lifting units if lifting hours are typed for it.
        litres_per_hour=statistics.median(lifting) if lifting else 0.0,
        base_litres=statistics.median(rule.base_litres for rule in rules),
        source=source,
    )


def reserve_from(training: pd.DataFrame, table: RuleTable) -> tuple[dict[str, float], float]:
    """Litres issued above the formula, per unit, from the training split only.

    The median, not the mean: a handful of large issues (a double day, a
    refill typed as one operation) should not raise every future allocation.
    Units with no issued rows get the fleet's median gap.
    """
    predicted = [
        table.litres(unit, km, hours)
        for unit, km, hours in zip(
            training["vehicle"].astype(str),
            training["distance_km"].to_numpy(dtype=float),
            training["lifting_hours"].to_numpy(dtype=float),
            strict=True,
        )
    ]
    gap = training["issued_litres"].to_numpy(dtype=float) - np.asarray(predicted)
    frame = pd.DataFrame({"vehicle": training["vehicle"].to_numpy(), "gap": gap})
    per_unit = {
        str(unit): float(value) for unit, value in frame.groupby("vehicle")["gap"].median().items()
    }
    return per_unit, float(np.median(gap))


def rule_pipeline(table: RuleTable) -> Pipeline:
    """The rule as a scikit-learn pipeline with its coefficients set, not fitted."""
    units = sorted(table.units)
    pipeline = Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "unit",
                            OneHotEncoder(
                                categories=[units], handle_unknown="ignore", sparse_output=False
                            ),
                            ["vehicle"],
                        ),
                        ("measure", "passthrough", [_KM, _HOURS]),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "interact",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("regress", LinearRegression()),
        ]
    )
    # Fit the two transformers only, on one row per unit, so the feature
    # layout is fixed; the regression is given its coefficients below.
    layout = feature_frame([{"vehicle": unit, _KM: 1.0, _HOURS: 1.0} for unit in units])
    pipeline[:-1].fit(layout)
    names = list(pipeline[:-1].get_feature_names_out())

    km, hours = f"measure__{_KM}", f"measure__{_HOURS}"
    default = table.default
    coefficients = {km: default.litres_per_km, hours: default.litres_per_hour}
    for unit in units:
        rule = table.units[unit]
        is_unit = f"unit__vehicle_{unit}"
        coefficients[is_unit] = rule.base_litres - default.base_litres
        coefficients[f"{is_unit} {km}"] = rule.litres_per_km - default.litres_per_km
        coefficients[f"{is_unit} {hours}"] = rule.litres_per_hour - default.litres_per_hour
    missing = set(coefficients) - set(names)
    if missing:
        raise RuntimeError(f"Feature layout changed; not found: {sorted(missing)[:3]}")

    regress: LinearRegression = pipeline.named_steps["regress"]
    regress.coef_ = np.array([coefficients.get(name, 0.0) for name in names])
    regress.intercept_ = default.base_litres
    regress.n_features_in_ = len(names)
    return pipeline


def feature_frame(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Rows in the app's feature contract, filling what a caller left out the
    way production's operations fill it."""
    defaults: dict[str, object] = {
        "vehicle_category": "ANGBER",
        "vehicle": UNKNOWN_VEHICLE,
        "activity_mode": "transport",
        "distance_source": "manual",
        _KM: 0.0,
        _HOURS: 0.0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    return frame[[name for name, _ in FEATURE_SCHEMA]]
