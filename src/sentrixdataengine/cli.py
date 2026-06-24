"""SentrixDataEngine CLI.

`materialize` (DE-CLI-1) turns a persisted SyncResult bundle (SentrixSync SYNC-1:
sync_result.json + arrays.npz + session.json) into a packaged dataset with no live
Sync process — the timeline grid is now carried on disk. `formats`, `validate`,
`inspect`, and `diff` operate on already-packaged datasets.
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
def materialize(
    bundle: Path = typer.Option(..., help="persisted SyncResult bundle dir (SentrixSync SYNC-1)"),
    out: Path = typer.Option(..., help="Gold output root"),
    formats: str = typer.Option("parquet", help="comma-separated Gold formats (see `formats`)"),
    version: str = typer.Option("0.1.0", help="dataset version"),
    dataset_id: str = typer.Option(None, help="override dataset id (default derived from session)"),
) -> None:
    """Materialize a packaged dataset from a persisted SyncResult bundle.

    Reads the bundle's SyncResult + Session (the Session's stream_refs point at the
    producer payload artifacts), runs the full resolve→materialize→validate→export
    →package pipeline, and writes Silver + Gold + manifest + provenance under `out`.
    No live SentrixSync process required.
    """
    from sentrixsync import load_session, load_sync_result

    from . import MaterializationRequest, Pipeline

    if not Path(bundle).exists():
        typer.echo(f"bundle not found: {bundle}", err=True)
        raise typer.Exit(code=2)
    sync_result = load_sync_result(bundle)
    session = load_session(bundle)
    if session is None:
        typer.echo(
            f"bundle {bundle} has no session.json; a Session (device descriptors + "
            "stream_refs) is required to materialize. Re-persist with "
            "save_sync_result(..., session=session).", err=True)
        raise typer.Exit(code=2)

    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    result = Pipeline().run(MaterializationRequest(
        sync_result=sync_result, session=session, out_root=Path(out),
        formats=fmts, version=version, dataset_id=dataset_id))

    typer.echo(json.dumps({
        "verdict": result.qa.gate_verdict,
        "dataset": str(result.layout.base),
        "silver": result.extras.get("silver_path"),
        "formats": list(fmts),
        "content_hash": result.content_hash,
        "manifest": str(result.layout.manifest_path),
        "provenance": str(result.layout.provenance_path),
    }, indent=2))


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
