"""SentrixDataEngine CLI.

V1 exposes `formats`, `exporters`, and `validate` (re-gate an existing dataset).
End-to-end `materialize` is driven from Python in V1 because it needs an
in-memory SyncResult (a Session manifest alone does not carry the timeline grid);
a manifest-only entry point lands in V2 once timelines are persisted on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__
from .exporters import registered_exporters

app = typer.Typer(add_completion=False, help="SentrixDataEngine — dataset materialization")


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


@app.command()
def formats() -> None:
    """List available export formats."""
    for name in registered_exporters():
        typer.echo(name)


@app.command()
def validate(dataset: Path = typer.Option(..., help="Path to a version=… dataset dir")) -> None:
    """Print the stored QA verdict for an already-packaged dataset."""
    qa_path = Path(dataset) / "qa_report.json"
    if not qa_path.exists():
        typer.echo(f"no qa_report.json under {dataset}", err=True)
        raise typer.Exit(code=1)
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    typer.echo(f"verdict: {data.get('gate_verdict')}")
    typer.echo(f"detail:  {data.get('detail')}")


@app.command()
def inspect(dataset: Path = typer.Option(..., help="Path to a version=… dataset dir")) -> None:
    """Print summary statistics for a packaged dataset."""
    from .inspect import summarize_dataset
    typer.echo(json.dumps(summarize_dataset(Path(dataset)), indent=2))


@app.command()
def diff(a: Path = typer.Option(..., help="dataset version dir A"),
         b: Path = typer.Option(..., help="dataset version dir B")) -> None:
    """Diff two packaged dataset versions."""
    from .inspect import diff_datasets
    typer.echo(json.dumps(diff_datasets(Path(a), Path(b)), indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
