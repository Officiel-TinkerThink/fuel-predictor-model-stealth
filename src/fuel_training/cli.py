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
