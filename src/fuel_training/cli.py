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


@app.command()
def template(out: Path = Path("templates/Template Operasi Harian.xlsx")) -> None:
    """The data-entry template for the field, with dropdowns and a column dictionary."""
    from fuel_training.template import write_template

    typer.echo(f"Template: {write_template(out, Catalogs.load())}")
