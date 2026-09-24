"""`fpt`: one command per pipeline stage (docs/pipeline.md)."""

from pathlib import Path

import typer

from fuel_training.catalog import Catalogs
from fuel_training.extract import extract as run_extract
from fuel_training.tidy import tidy as run_tidy
from fuel_training.unit_days import build as run_unit_days

app = typer.Typer(add_completion=False, no_args_is_help=True)
DATA = Path("data")


@app.command()
def extract(workbook: Path) -> None:
    """Stage 1: the workbook's sheets into data/raw/<dataset-version>/."""
    result = run_extract(workbook, DATA / "raw")
    typer.echo(f"{result.version}: {len(result.sheets)} sheets → {result.directory}")


@app.command()
def tidy(version: str) -> None:
    """Stage 2: operations, actuals, quarantine and the app import into data/tidy/<version>/."""
    raw = DATA / "raw" / version
    if not raw.is_dir():
        raise typer.BadParameter(f"{raw} does not exist; run `fpt extract` first.")
    result = run_tidy(raw, DATA / "tidy", Catalogs.load())
    typer.echo(
        f"{result.version}: {len(result.operations)} operations "
        f"({result.app_rows} app-importable), {len(result.quarantine)} quarantined, "
        f"{len(result.actuals)} fuel-stick readings → {result.directory}"
    )
    typer.echo(f"Report: {result.directory / 'report.md'}")


@app.command()
def prepare(workbook: Path) -> None:
    """Stages 1 and 2 in one go."""
    extraction = run_extract(workbook, DATA / "raw")
    catalogs = Catalogs.load()
    result = run_tidy(extraction.directory, DATA / "tidy", catalogs)
    days = run_unit_days(extraction.directory, result.directory, catalogs)
    typer.echo(f"{len(days.table)} unit-days → {days.path}")
    typer.echo(
        f"{result.version}: {len(result.operations)} operations "
        f"({result.app_rows} app-importable), {len(result.quarantine)} quarantined"
    )
    typer.echo(f"Report: {result.directory / 'report.md'}")


@app.command("unit-days")
def unit_days(version: str) -> None:
    """Stage 2b: one row per unit per day from issued litres, GPS and the fuel stick."""
    result = run_unit_days(DATA / "raw" / version, DATA / "tidy" / version, Catalogs.load())
    typer.echo(f"{len(result.table)} unit-days → {result.path}")


@app.command()
def ready(version: str) -> None:
    """Stage 3: training-ready tables (doubtful rows erased) into data/ready/<version>/."""
    from fuel_training.ready import build as run_ready

    result = run_ready(DATA / "tidy" / version, DATA / "ready")
    typer.echo(
        f"train-issued {len(result.issued)} · train-measured {len(result.measured)} · "
        f"erased {len(result.removed)} → {result.directory}"
    )


@app.command()
def explore(version: str) -> None:
    """Exploration of the clean tables: charts and numbers into reports/exploration/<version>/."""
    from fuel_training.explore import explore as run_explore
    from fuel_training.planner import planner_ratios

    ratios = planner_ratios(DATA / "raw" / version, Catalogs.load())
    report = run_explore(DATA / "ready" / version, Path("reports") / "exploration", ratios)
    typer.echo(f"Report: {report}")


@app.command("package-rule")
def package_rule(
    version: str,
    reserve: bool = typer.Option(
        False,
        "--reserve/--no-reserve",
        help="Add the litres planners issue above the formula, per unit, from the training split.",
    ),
) -> None:
    """The planners' Data Ratio rule as a production model package (docs/rule-model.md)."""
    import pandas as pd

    from fuel_training.planner import planner_ratios
    from fuel_training.rule_model import RuleTable, reserve_from
    from fuel_training.rule_package import build, report, write

    issued_path = DATA / "ready" / version / "train-issued.csv"
    if not issued_path.is_file():
        raise typer.BadParameter(f"{issued_path} does not exist; run `fpt ready` first.")
    catalogs = Catalogs.load()
    ratios = planner_ratios(DATA / "raw" / version, catalogs)
    table = RuleTable.from_ratios(ratios, catalogs.vehicles.options())
    issued = pd.read_csv(issued_path)
    parameter_rows = len(ratios)
    if reserve:
        training = issued[issued["split"] == "train"]
        per_unit, fleet = reserve_from(training, table)
        table = table.with_reserve(per_unit, fleet)
        parameter_rows += len(training)
    model_version = f"rule-data-ratio{'-reserve' if reserve else ''}-{version}"
    package = build(
        table,
        model_version=model_version,
        dataset_version=version,
        issued=issued,
        parameter_rows=parameter_rows,
        catalog_fingerprint=catalogs.fingerprint(),
    )
    path = write(package, Path("packages"))
    report_path = Path("reports") / "rule" / f"{model_version}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(table, package), encoding="utf-8")
    metrics = package.metrics
    typer.echo(
        f"{model_version}: MAE {metrics.mae_liters:.2f} L on {len(package.test_rows)} "
        f"held-out operations → {path}"
    )
    typer.echo(f"Report: {report_path}")


@app.command()
def template(out: Path = Path("templates/Template Operasi Harian.xlsx")) -> None:
    """The data-entry template for the field, with dropdowns and a column dictionary."""
    from fuel_training.template import write_template

    typer.echo(f"Template: {write_template(out, Catalogs.load())}")
