"""The planners' current rule, from the `Data Ratio` tab: the baseline any model must beat.

    litres = km / drive_ratio (km per litre) + lifting hours x lift_ratio (L/h) + safety

Read from the extracted sheet, not typed in here, so a change the planners make
to their ratios reaches the comparison on the next run.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fuel_training.catalog import Catalogs
from fuel_training.extract import raw_frame
from fuel_training.tidy import parse_number

# `Data Ratio` spellings the catalog does not carry as aliases.
_RATIO_NAMES = {"Whinch Truck OFT": "Winch Truck", "Tronton OFT": "Oil Field Truck"}


@dataclass(frozen=True, slots=True)
class PlannerRatio:
    km_per_litre: float
    litres_per_lifting_hour: float
    safety_litres: float

    def litres(self, km: float, lifting_hours: float) -> float:
        return (
            km / self.km_per_litre
            + lifting_hours * self.litres_per_lifting_hour
            + (self.safety_litres)
        )


def planner_ratios(raw_directory: Path, catalogs: Catalogs) -> dict[str, PlannerRatio]:
    sheet = raw_frame(raw_directory, "Data Ratio")
    columns = {str(c).replace("\n", " "): c for c in sheet.columns}
    drive, lift, safety = (
        columns["Drive Ratio (KM/L)"],
        columns["Lift Ratio (L/Jam)"],
        columns["Safety Factor (L)"],
    )
    ratios: dict[str, PlannerRatio] = {}
    for _, row in sheet.iterrows():
        written = str(row["Nama Kendaraan"]).split(" - ")[0].strip()
        vehicle = catalogs.vehicle(_RATIO_NAMES.get(written, written)) if written else None
        km_per_litre = parse_number(row[drive])
        if vehicle is None or not km_per_litre:
            continue
        ratios[vehicle] = PlannerRatio(
            km_per_litre=km_per_litre,
            litres_per_lifting_hour=parse_number(row[lift]) or 0.0,
            safety_litres=parse_number(row[safety]) or 0.0,
        )
    return ratios


def planner_prediction(
    frame: pd.DataFrame, ratios: dict[str, PlannerRatio], km_column: str
) -> pd.Series:
    """The rule applied to each row; NaN for a unit the planners have no ratio for."""

    def one(row: pd.Series) -> float:
        ratio = ratios.get(str(row["vehicle"]))
        if ratio is None:
            return float("nan")
        return ratio.litres(float(row[km_column]), float(row["lifting_hours"]))

    return frame.apply(one, axis=1)
