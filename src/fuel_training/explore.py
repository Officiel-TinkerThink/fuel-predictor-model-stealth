"""Exploration of the training-ready tables: the evidence behind every treatment decision.

Writes `reports/exploration/<version>/`: one chart per question and
`exploration.md` with the numbers. The *decisions* drawn from it are argued in
`docs/exploration.md`; this module only measures, so re-running it on a new
dataset version shows whether those decisions still hold.

It reads the clean tables only (`data/ready/<version>/`): what it describes is
exactly what a model will be fitted on.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats  # noqa: E402

from fuel_training.planner import PlannerRatio  # noqa: E402

FEATURES = ["gps_km", "gps_usage_hours", "gps_drive_hours", "gps_working_hours", "gps_idle_hours"]


def _save(figure: Figure, directory: Path, name: str) -> str:
    figure.tight_layout()
    figure.savefig(directory / name, dpi=110)
    plt.close(figure)
    return name


def _grid(count: int) -> tuple[Figure, list[Axes]]:
    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows), squeeze=False)
    flat = list(axes.ravel())
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat


# --- one function per question --------------------------------------------------------


def distributions(issued: pd.DataFrame, measured: pd.DataFrame, out: Path) -> tuple[str, list[str]]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (frame, target, title) in zip(
        axes,
        (
            (issued, "issued_litres", "Issued litres per operation"),
            (measured, "measured_litres", "Measured litres per unit-day"),
        ),
        strict=True,
    ):
        for group, part in frame.groupby("vehicle_group"):
            axis.hist(part[target], bins=30, alpha=0.55, label=str(group))
        axis.set_title(title)
        axis.set_xlabel("litres")
        axis.legend()
    lines = [
        "| Table | Group | n | mean | median | std | CV | skew |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, frame, target in (
        ("issued", issued, "issued_litres"),
        ("measured", measured, "measured_litres"),
    ):
        for group, part in frame.groupby("vehicle_group"):
            y = part[target]
            lines.append(
                f"| {name} | {group} | {len(y)} | {y.mean():.1f} | {y.median():.1f} "
                f"| {y.std():.1f} "
                f"| {y.std() / y.mean():.2f} | {stats.skew(y):.2f} |"
            )
    return _save(figure, out, "01-target-distributions.png"), lines


def litres_against_km(
    frame: pd.DataFrame, target: str, km: str, out: Path, name: str
) -> tuple[str, list[str]]:
    units = sorted(frame["vehicle"].unique())
    figure, axes = _grid(len(units))
    lines = [
        "| Unit | n | L per km (slope) | litres at 0 km (intercept) | R² |",
        "|---|---|---|---|---|",
    ]
    for axis, unit in zip(axes, units, strict=False):
        part = frame[frame["vehicle"] == unit]
        axis.scatter(part[km], part[target], s=10, alpha=0.6)
        if part[km].nunique() > 1:
            fit = stats.linregress(part[km], part[target])
            xs = np.linspace(0, part[km].max(), 20)
            axis.plot(xs, fit.intercept + fit.slope * xs, color="black", linewidth=1)
            lines.append(
                f"| {unit} | {len(part)} | {fit.slope:.2f} | {fit.intercept:.1f} "
                f"| {fit.rvalue**2:.2f} |"
            )
        axis.set_title(unit, fontsize=9)
        axis.set_xlabel("km", fontsize=8)
        axis.set_ylabel("litres", fontsize=8)
    return _save(figure, out, name), lines


def lifting(issued: pd.DataFrame, measured: pd.DataFrame, out: Path) -> tuple[str, list[str]]:
    cranes = sorted(issued.loc[issued["lifting_hours"] > 0, "vehicle"].unique())
    figure, axes = plt.subplots(2, len(cranes), figsize=(4 * len(cranes), 7), squeeze=False)
    lines = [
        "Litres ≈ a + b·km + c·lifting_hours, fitted per crane:",
        "",
        "| Crane | Table | n | L/km | L per lifting hour |",
        "|---|---|---|---|---|",
    ]
    for column, crane in enumerate(cranes):
        for row, (frame, target, km, label) in enumerate(
            (
                (issued, "issued_litres", "distance_km", "issued"),
                (measured, "measured_litres", "gps_km", "measured"),
            )
        ):
            part = frame[(frame["vehicle"] == crane)]
            axis = axes[row][column]
            axis.scatter(part["lifting_hours"], part[target], s=12, alpha=0.6)
            axis.set_title(f"{crane} — {label}", fontsize=9)
            axis.set_xlabel("lifting hours", fontsize=8)
            if len(part) > 3:
                design = np.column_stack([np.ones(len(part)), part[km], part["lifting_hours"]])
                coef, *_ = np.linalg.lstsq(design, part[target].to_numpy(float), rcond=None)
                lines.append(f"| {crane} | {label} | {len(part)} | {coef[1]:.2f} | {coef[2]:.2f} |")
    return _save(figure, out, "04-cranes-lifting.png"), lines


def ratio_check(
    measured: pd.DataFrame, ratios: dict[str, PlannerRatio], out: Path
) -> tuple[str, list[str]]:
    totals = measured.groupby("vehicle")[["gps_km", "measured_litres"]].sum()
    per_unit = (totals["gps_km"] / totals["measured_litres"]).sort_index()
    planned = pd.Series(
        {u: ratios[u].km_per_litre if u in ratios else np.nan for u in per_unit.index}
    )
    figure, axis = plt.subplots(figsize=(11, 4))
    x = np.arange(len(per_unit))
    axis.bar(x - 0.2, per_unit, width=0.4, label="measured (fuel stick ÷ GPS)")
    axis.bar(x + 0.2, planned, width=0.4, label="planners' Data Ratio")
    axis.set_xticks(x, per_unit.index, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("km per litre")
    axis.legend()
    lines = ["| Unit | measured km/L | planners' km/L | ratio |", "|---|---|---|---|"]
    for unit in per_unit.index:
        p = planned[unit]
        lines.append(
            f"| {unit} | {per_unit[unit]:.2f} | {p:.2f} | {per_unit[unit] / p:.2f} |"
            if not np.isnan(p)
            else f"| {unit} | {per_unit[unit]:.2f} | — | — |"
        )
    return _save(figure, out, "05-km-per-litre-vs-planners.png"), lines


def drift(issued: pd.DataFrame, measured: pd.DataFrame, out: Path) -> tuple[str, list[str]]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    lines = ["| Table | Group | " + " | ".join(["month → km per litre"]) + " |", "|---|---|---|"]
    for axis, (frame, target, km, title) in zip(
        axes,
        (
            (issued, "issued_litres", "distance_km", "Issued: km per issued litre"),
            (measured, "measured_litres", "gps_km", "Measured: km per litre burnt"),
        ),
        strict=True,
    ):
        month = pd.to_datetime(frame["date"]).dt.to_period("M")
        for group, part in frame.groupby("vehicle_group"):
            by_month = part.groupby(month.loc[part.index])[[km, target]].sum()
            series = by_month[km] / by_month[target]
            axis.plot(series.index.astype(str), series.to_numpy(), marker="o", label=str(group))
            lines.append(
                f"| {title.split(':')[0]} | {group} | "
                + ", ".join(f"{m}: {v:.2f}" for m, v in series.items())
                + " |"
            )
        axis.set_title(title)
        axis.legend()
        axis.tick_params(axis="x", rotation=45)
    return _save(figure, out, "06-monthly-drift.png"), lines


def features(measured: pd.DataFrame, out: Path) -> tuple[str, list[str]]:
    columns = [*FEATURES, "lifting_hours", "measured_litres"]
    corr = measured[columns].corr()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(columns)), columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            axis.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image)
    zero_km = measured[measured["gps_km"] == 0]
    lines = [
        "Correlation (measured table): see chart.",
        "",
        f"- Days with 0 GPS km: {len(zero_km)}, "
        f"mean measured litres {zero_km['measured_litres'].mean():.1f}"
        " — a unit that does not move still burns fuel (idle, pumping).",
    ]
    weekday = measured.groupby(pd.to_datetime(measured["date"]).dt.dayofweek)[
        "measured_litres"
    ].mean()
    lines.append(
        "- Mean measured litres by weekday (Mon=0): "
        + ", ".join(f"{d}: {v:.1f}" for d, v in weekday.items())
    )
    return _save(figure, out, "07-feature-correlation.png"), lines


def heteroscedasticity(measured: pd.DataFrame, out: Path) -> tuple[str, list[str]]:
    residuals = []
    for _, part in measured.groupby("vehicle"):
        fit = stats.linregress(part["gps_km"], part["measured_litres"])
        residuals.append(
            pd.DataFrame(
                {
                    "km": part["gps_km"],
                    "residual": part["measured_litres"]
                    - (fit.intercept + fit.slope * part["gps_km"]),
                }
            )
        )
    frame = pd.concat(residuals)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(frame["km"], frame["residual"], s=6, alpha=0.4)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("GPS km")
    axis.set_ylabel("residual litres (per-unit linear fit)")
    bins = pd.qcut(frame["km"], 4)
    spread = frame.groupby(bins, observed=True)["residual"].apply(lambda s: s.abs().mean())
    lines = [
        "Mean |residual| by km quartile: " + ", ".join(f"{b}: {v:.1f} L" for b, v in spread.items())
    ]
    return _save(figure, out, "08-residual-spread.png"), lines


def _r_squared(target: np.ndarray, columns: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones(len(target)), *columns])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(1 - residual.var() / target.var())


def reading_noise(measured: pd.DataFrame) -> list[str]:
    """Daily against weekly: if a week explains far more than a day, the daily
    scatter is mostly stick-reading error, which cancels over a week."""
    frame = measured.assign(week=pd.to_datetime(measured["date"]).dt.to_period("W"))
    weekly = (
        frame.groupby(["vehicle", "week"])
        .agg(
            litres=("measured_litres", "sum"),
            km=("gps_km", "sum"),
            drive=("gps_drive_hours", "sum"),
            idle=("gps_idle_hours", "sum"),
            days=("date", "count"),
        )
        .reset_index()
    )
    weekly = weekly[weekly["days"] >= 4]
    lines = ["| Unit | daily R² | weekly R² | full weeks |", "|---|---|---|---|"]
    for unit, part in frame.groupby("vehicle"):
        daily = _r_squared(
            part["measured_litres"].to_numpy(float),
            [part[c].to_numpy(float) for c in ("gps_km", "gps_drive_hours", "gps_idle_hours")],
        )
        weeks = weekly[weekly["vehicle"] == unit]
        if len(weeks) < 6:
            lines.append(f"| {unit} | {daily:.2f} | — | {len(weeks)} |")
            continue
        week_r2 = _r_squared(
            weeks["litres"].to_numpy(float),
            [weeks[c].to_numpy(float) for c in ("days", "km", "drive", "idle")],
        )
        lines.append(f"| {unit} | {daily:.2f} | {week_r2:.2f} | {len(weeks)} |")
    return lines


def split_coverage(issued: pd.DataFrame, measured: pd.DataFrame) -> list[str]:
    lines = ["| Table | Unit | train | test | last date |", "|---|---|---|---|---|"]
    for name, frame in (("issued", issued), ("measured", measured)):
        for unit, part in frame.groupby("vehicle"):
            counts = part["split"].value_counts()
            lines.append(
                f"| {name} | {unit} | {counts.get('train', 0)} | {counts.get('test', 0)} "
                f"| {part['date'].max()} |"
            )
    return lines


# --- the run --------------------------------------------------------------------------------


def explore(ready_directory: Path, out_root: Path, ratios: dict[str, PlannerRatio]) -> Path:
    issued = pd.read_csv(ready_directory / "train-issued.csv")
    measured = pd.read_csv(ready_directory / "train-measured.csv")
    out = out_root / ready_directory.name
    out.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str | None, list[str]]] = []
    chart, lines = distributions(issued, measured, out)
    sections.append(("1. What is being predicted: target distributions", chart, lines))
    chart, lines = litres_against_km(
        issued, "issued_litres", "distance_km", out, "02-issued-vs-km.png"
    )
    sections.append(("2. Issued litres against typed km, per unit", chart, lines))
    chart, lines = litres_against_km(
        measured, "measured_litres", "gps_km", out, "03-measured-vs-km.png"
    )
    sections.append(("3. Measured litres against GPS km, per unit", chart, lines))
    chart, lines = lifting(issued, measured, out)
    sections.append(("4. The three lifting units: litres against lifting hours", chart, lines))
    chart, lines = ratio_check(measured, ratios, out)
    sections.append(("5. Measured km per litre against the planners' ratios", chart, lines))
    chart, lines = drift(issued, measured, out)
    sections.append(("6. Drift: km per litre by month", chart, lines))
    chart, lines = features(measured, out)
    sections.append(("7. Candidate features (measured table)", chart, lines))
    chart, lines = heteroscedasticity(measured, out)
    sections.append(("8. Does the error grow with distance?", chart, lines))
    sections.append(("9. Train / test coverage per unit", None, split_coverage(issued, measured)))
    sections.append(
        ("10. Reading noise: km and hours explain a day, and a week", None, reading_noise(measured))
    )
    text = [
        f"# Exploration — {ready_directory.name}",
        "",
        f"Clean tables only: {len(issued)} issued operations, {len(measured)} measured unit-days.",
        "",
    ]
    for title, section_chart, section_lines in sections:
        text += [f"## {title}", ""]
        if section_chart:
            text += [f"![{title}]({section_chart})", ""]
        text += [*section_lines, ""]
    report = out / "exploration.md"
    report.write_text("\n".join(text), encoding="utf-8")
    return report
