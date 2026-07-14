"""Deterministic compact artifacts and SVG figures for retrieval-CV Step 12."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analysis import (
    AnalysisBundle,
    EXPECTED_DATASET_INPUT_FILES,
    KS,
    PRIMARY_REGIME,
)


REPORT_SCHEMA_VERSION = 1
REPORT_PROTOCOL = "arr_retrieval_case_first_report_v1"
_COLORS = {
    ("flat_masked", "local_unique"): "#2563eb",
    ("flat_masked", "global_uniform"): "#06b6d4",
    ("structured", "local_unique"): "#dc2626",
    ("structured", "global_uniform"): "#f59e0b",
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Cannot publish an empty CSV artifact")
    fieldnames = list(rows[0])
    expected = set(fieldnames)
    if any(set(row) != expected for row in rows):
        raise ValueError("CSV rows do not share one exact schema")
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(
        destination,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if value is None else value for key, value in row.items()})
    return destination.getvalue().encode("utf-8")


def _svg(width: int, height: int, body: Sequence[str], *, title: str) -> bytes:
    payload = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}'
        '.title{font-size:18px;font-weight:700}.axis{font-size:11px}'
        '.label{font-size:12px}.note{font-size:10px;fill:#526071}</style>',
        *body,
        "</svg>",
    ]
    return ("\n".join(payload) + "\n").encode("utf-8")


def _finite(values: Sequence[float], *, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _forest_plot(contrasts: Sequence[Mapping[str, Any]]) -> bytes:
    if len(contrasts) != 5:
        raise ValueError("Forest plot requires exactly five prespecified contrasts")
    values = _finite(
        [
            float(record[key])
            for record in contrasts
            for key in (
                "estimate",
                "case_bootstrap_lower",
                "case_bootstrap_upper",
                "hierarchical_lower",
                "hierarchical_upper",
            )
        ],
        name="forest values",
    )
    low = min([0.0, *values])
    high = max([0.0, *values])
    padding = max((high - low) * 0.12, 0.01)
    low -= padding
    high += padding
    left, right, top, row_height = 345, 920, 72, 62
    width, height = 980, top + row_height * len(contrasts) + 78

    def x(value: float) -> float:
        return left + (value - low) * (right - left) / (high - low)

    body = [
        '<text x="24" y="30" class="title">Primary paired case contrasts (Hit@20, fold-global)</text>',
        f'<line x1="{x(0):.2f}" y1="50" x2="{x(0):.2f}" y2="{height - 55}" stroke="#7b8794" stroke-dasharray="4 4"/>',
    ]
    for position, record in enumerate(contrasts):
        y = top + position * row_height
        label = html.escape(str(record["label"]))
        lower = float(record["case_bootstrap_lower"])
        upper = float(record["case_bootstrap_upper"])
        h_lower = float(record["hierarchical_lower"])
        h_upper = float(record["hierarchical_upper"])
        estimate = float(record["estimate"])
        body.extend(
            [
                f'<text x="24" y="{y + 4}" class="label">{label}</text>',
                f'<line x1="{x(h_lower):.2f}" y1="{y + 12}" x2="{x(h_upper):.2f}" y2="{y + 12}" stroke="#94a3b8" stroke-width="3"/>',
                f'<line x1="{x(lower):.2f}" y1="{y}" x2="{x(upper):.2f}" y2="{y}" stroke="#1f4e79" stroke-width="4"/>',
                f'<circle cx="{x(estimate):.2f}" cy="{y}" r="5" fill="#1f4e79"/>',
            ]
        )
    axis_y = height - 45
    for tick in range(6):
        value = low + tick * (high - low) / 5
        tick_x = x(value)
        body.extend(
            [
                f'<line x1="{tick_x:.2f}" y1="{axis_y - 5}" x2="{tick_x:.2f}" y2="{axis_y}" stroke="#172033"/>',
                f'<text x="{tick_x:.2f}" y="{axis_y + 16}" text-anchor="middle" class="axis">{value:.3f}</text>',
            ]
        )
    body.append(
        f'<text x="{(left + right) / 2:.2f}" y="{height - 8}" text-anchor="middle" class="note">Dark: paired case bootstrap; gray: hierarchical case/seed sensitivity</text>'
    )
    return _svg(width, height, body, title="Primary retrieval contrast forest plot")


def _hit_curves(cell_summary: Sequence[Mapping[str, Any]]) -> bytes:
    records = [record for record in cell_summary if record["regime_name"] == PRIMARY_REGIME]
    if len(records) != 4:
        raise ValueError("Hit@K plot requires exactly four controlled primary cells")
    width, height = 880, 520
    left, right, top, bottom = 72, 610, 58, 438

    def x(k: int) -> float:
        return left + KS.index(k) * (right - left) / (len(KS) - 1)

    body = [
        '<text x="24" y="30" class="title">Case-first Hit@K by representation and sampler</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value:.1f}</text>',
            ]
        )
    for k in KS:
        tick_x = x(k)
        body.append(
            f'<text x="{tick_x:.2f}" y="{bottom + 22}" text-anchor="middle" class="axis">{k}</text>'
        )
    for legend_position, record in enumerate(
        sorted(records, key=lambda item: (item["controlled_query_view"], item["sampler"]))
    ):
        key = (record["controlled_query_view"], record["sampler"])
        color = _COLORS[key]
        values = [float(record[f"hit_at_{k}"]) for k in KS]
        points = " ".join(
            f"{x(k):.2f},{bottom - value * (bottom - top):.2f}"
            for k, value in zip(KS, values)
        )
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for k, value in zip(KS, values):
            body.append(
                f'<circle cx="{x(k):.2f}" cy="{bottom - value * (bottom - top):.2f}" r="4" fill="{color}"/>'
            )
        label = f"{record['controlled_query_view']} / {record['sampler']}"
        legend_y = 90 + legend_position * 32
        body.extend(
            [
                f'<line x1="650" y1="{legend_y}" x2="682" y2="{legend_y}" stroke="{color}" stroke-width="4"/>',
                f'<text x="692" y="{legend_y + 4}" class="label">{html.escape(label)}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{(left + right) / 2}" y="{height - 28}" text-anchor="middle" class="axis">K</text>',
            f'<text x="18" y="{(top + bottom) / 2}" transform="rotate(-90 18 {(top + bottom) / 2})" text-anchor="middle" class="axis">Case-macro hit rate</text>',
            '<text x="650" y="235" class="note">Queries averaged within case;</text>',
            '<text x="650" y="251" class="note">seeds averaged within case;</text>',
            '<text x="650" y="267" class="note">then 42 cases averaged.</text>',
        ]
    )
    return _svg(width, height, body, title="Case-first Hit at K curves")


def _per_case_plot(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if len(rows) != 42:
        raise ValueError("Per-case plot requires exactly 42 cases")
    keys = ("flat_global_minus_local", "structured_global_minus_local")
    values = _finite(
        [float(row[key]) for row in rows for key in keys],
        name="per-case paired values",
    )
    bound = max(max(abs(value) for value in values) * 1.1, 0.05)
    width, height = 1020, 500
    left, right, top, bottom = 70, 970, 50, 420

    def x(position: int) -> float:
        return left + position * (right - left) / (len(rows) - 1)

    def y(value: float) -> float:
        return top + (bound - value) * (bottom - top) / (2 * bound)

    body = [
        '<text x="24" y="28" class="title">Per-case effect of global-uniform training on Hit@20</text>',
        f'<line x1="{left}" y1="{y(0):.2f}" x2="{right}" y2="{y(0):.2f}" stroke="#64748b" stroke-dasharray="4 4"/>',
    ]
    series = (
        (keys[0], "Flat", "#2563eb"),
        (keys[1], "Structured", "#dc2626"),
    )
    for key, label, color in series:
        points = " ".join(
            f"{x(position):.2f},{y(float(row[key])):.2f}"
            for position, row in enumerate(rows)
        )
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.75"/>')
        for position, row in enumerate(rows):
            body.append(
                f'<circle cx="{x(position):.2f}" cy="{y(float(row[key])):.2f}" r="2.8" fill="{color}"/>'
            )
        legend_x = 745 if label == "Flat" else 850
        body.extend(
            [
                f'<circle cx="{legend_x}" cy="25" r="4" fill="{color}"/>',
                f'<text x="{legend_x + 9}" y="29" class="label">{label}</text>',
            ]
        )
    for tick in range(5):
        value = -bound + tick * (2 * bound) / 4
        tick_y = y(value)
        body.append(
            f'<text x="{left - 8}" y="{tick_y + 4:.2f}" text-anchor="end" class="axis">{value:.2f}</text>'
        )
    body.extend(
        [
            f'<text x="{(left + right) / 2}" y="{height - 24}" text-anchor="middle" class="axis">Held-out cases (lexicographic ID order)</text>',
            f'<text x="18" y="{(top + bottom) / 2}" transform="rotate(-90 18 {(top + bottom) / 2})" text-anchor="middle" class="axis">Global-uniform minus local-unique</text>',
        ]
    )
    return _svg(width, height, body, title="Per-case paired sampler effects")


def _fold_load_table(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if len(rows) != 5:
        raise ValueError("Fold-load table requires exactly five folds")
    width, height = 660, 270
    columns = ("Fold", "Cases", "Queries", "Passages", "Ranking rows")
    x_positions = (60, 180, 285, 400, 540)
    body = ['<text x="24" y="28" class="title">Held-out fold workload</text>']
    for column, x_position in zip(columns, x_positions):
        body.append(
            f'<text x="{x_position}" y="65" text-anchor="middle" class="label">{column}</text>'
        )
    body.append('<line x1="25" y1="75" x2="635" y2="75" stroke="#94a3b8"/>')
    for position, row in enumerate(rows):
        y = 105 + position * 31
        values = (
            row["outer_fold"],
            row["case_count"],
            row["query_count"],
            row["passage_count"],
            row["ranking_row_count"],
        )
        if position % 2:
            body.append(f'<rect x="25" y="{y - 19}" width="610" height="27" fill="#f8fafc"/>')
        for value, x_position in zip(values, x_positions):
            body.append(
                f'<text x="{x_position}" y="{y}" text-anchor="middle" class="label">{value}</text>'
            )
    return _svg(width, height, body, title="Held-out fold workload table")


def _report_markdown(bundle: AnalysisBundle) -> bytes:
    summary = bundle.summary
    lines = [
        "# Retrieval cross-validation analysis",
        "",
        "This report is generated from five version-bound complete-ranking bundles. "
        "Every stored query metric was independently recomputed from the raw ranking "
        "and its multi-positive gold set before aggregation.",
        "",
        "## Locked analysis",
        "",
        "Queries are averaged within each held-out case, matched seeds 17/29/43 are "
        "then averaged within that case, and the 42 case values are finally averaged. "
        "The primary endpoint is case-macro Hit@20 under the fold-global candidate "
        "regime. Intervals use 10,000 paired case resamples with analysis seed 17; "
        "they are conditional on the three trained seeds. The hierarchical case/seed "
        "interval is reported only as a sensitivity analysis.",
        "",
        "## Primary cells",
        "",
        "| Representation | Sampler | Hit@20 | Seed SD |",
        "|---|---:|---:|---:|",
    ]
    for record in summary["primary_endpoint"]["cells"]:
        lines.append(
            f"| {record['query_view']} | {record['sampler']} | "
            f"{float(record['estimate']):.6f} | {float(record['seed_sd']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified contrasts",
            "",
            "| Contrast | Estimate | 95% paired-case CI | Hierarchical sensitivity CI | Seed 17 | Seed 29 | Seed 43 | Seed SD | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in bundle.contrasts:
        lines.append(
            f"| {record['label']} | {float(record['estimate']):.6f} | "
            f"[{float(record['case_bootstrap_lower']):.6f}, "
            f"{float(record['case_bootstrap_upper']):.6f}] | "
            f"[{float(record['hierarchical_lower']):.6f}, "
            f"{float(record['hierarchical_upper']):.6f}] | "
            f"{float(record['seed_17_estimate']):.6f} | "
            f"{float(record['seed_29_estimate']):.6f} | "
            f"{float(record['seed_43_estimate']):.6f} | "
            f"{float(record['seed_sd']):.6f} | {record['claim_status']} |"
        )
    lines.extend(
        [
            "",
            "A positive paper-facing claim is permitted only when the point estimate "
            "is positive and its paired case-bootstrap interval is wholly above zero. "
            "Intervals crossing zero are described as uncertain.",
            "",
            "## Context-excluded robustness",
            "",
            "| Representation | Sampler | Fold-global | Context-excluded | Difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in summary["context_excluded_sensitivity"]:
        lines.append(
            f"| {record['query_view']} | {record['sampler']} | "
            f"{float(record['fold_global']):.6f} | "
            f"{float(record['fold_global_context_excluded']):.6f} | "
            f"{float(record['difference']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The context-excluded regime removes visible non-gold passages from the "
            "complete fold-global ranking without rescoring and never removes a gold.",
            "",
            "## Sampler definitions",
            "",
            "`local_unique` uses 40 unique same-case and 20 unique other-case "
            "negatives. `global_uniform` samples 60 unique negatives passage-uniformly "
            "from all eligible passages in the outer training folds. Both exclude the "
            "current query's positives, use no replacement, and share matched positive "
            "selection across representation and sampler cells.",
            "",
            "## Study boundary and correction",
            "",
            "The controlled analysis uses the corrected 490-query dataset, in which "
            "all 42 cases yield queries. The frozen March 471-query dataset is retained "
            "only for legacy-configuration comparison and is not mixed into this "
            "aggregate. The parser correction includes the left-directed supporting "
            "edge into case 42's final Conclusion. The corrected controlled runs use "
            "24–26 training cases per outer fold, so absolute values are not presented "
            "as a bitwise replication of the March 34-case configuration.",
            "",
            "## Evidence",
            "",
            "`rankings_manifest.json` records the exact S3 key, VersionId, byte size, "
            "and SHA-256 for each raw ranking input; the multi-gigabyte ranking files "
            "are not duplicated in this compact report. `jobs.json` binds every fold "
            "to its completed SageMaker receipt. The exact experiment and fold manifests, "
            "the complete corrected dataset input files with their manifest, and all "
            "terminal/acquisition receipts are copied under `input/`. "
            "All derived tables and SVG inputs are "
            "listed in `analysis_manifest.json`.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _rename_directory_to_absent(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace analysis publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, f"Refusing to replace analysis output: {target}")
    raise OSError(error_number, f"Atomic analysis publication failed: {source} -> {target}")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _payloads(bundle: AnalysisBundle) -> dict[str, bytes]:
    for name, value, raw in (
        ("experiment", bundle.experiment_config, bundle.experiment_config_bytes),
        ("fold", bundle.fold_manifest, bundle.fold_manifest_bytes),
        ("dataset", bundle.dataset_manifest, bundle.dataset_manifest_bytes),
    ):
        if type(raw) is not bytes or not raw:
            raise ValueError(f"{name} input evidence must be exact non-empty bytes")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} input evidence is not valid JSON") from error
        if parsed != value:
            raise ValueError(f"{name} input evidence bytes disagree with parsed value")
    payloads = {
        "input/experiment.json": bundle.experiment_config_bytes,
        "input/folds.json": bundle.fold_manifest_bytes,
        "input/dataset_manifest.json": bundle.dataset_manifest_bytes,
        "evaluation_index.json": _canonical_bytes(bundle.evaluation_index),
        "jobs.json": _canonical_bytes(
            {"schema_version": REPORT_SCHEMA_VERSION, "jobs": bundle.jobs}
        ),
        "rankings_manifest.json": _canonical_bytes(
            {"schema_version": REPORT_SCHEMA_VERSION, "rankings": bundle.rankings}
        ),
        "fold_load.csv": _csv_bytes(bundle.fold_load),
        "query_metrics.csv": _csv_bytes(bundle.query_metrics),
        "case_metrics.csv": _csv_bytes(bundle.case_metrics),
        "system_summary.csv": _csv_bytes(bundle.system_summary),
        "cell_case_metrics.csv": _csv_bytes(bundle.cell_case_metrics),
        "cell_summary.csv": _csv_bytes(bundle.cell_summary),
        "seed_summary.csv": _csv_bytes(bundle.seed_summary),
        "contrasts.csv": _csv_bytes(bundle.contrasts),
        "per_case_primary.csv": _csv_bytes(bundle.per_case_primary),
        "summary.json": _canonical_bytes(bundle.summary),
        "report.md": _report_markdown(bundle),
        "figures/primary_contrasts_forest.svg": _forest_plot(bundle.contrasts),
        "figures/hit_at_k_curves.svg": _hit_curves(bundle.cell_summary),
        "figures/per_case_sampler_effect.svg": _per_case_plot(bundle.per_case_primary),
        "figures/fold_load_table.svg": _fold_load_table(bundle.fold_load),
    }
    if (
        tuple(name for name, _ in bundle.dataset_input_files)
        != EXPECTED_DATASET_INPUT_FILES
    ):
        raise ValueError("Corrected dataset evidence inventory changed")
    for relative_name, payload in bundle.dataset_input_files:
        if type(payload) is not bytes or not payload:
            raise ValueError(
                f"Corrected dataset evidence must be exact non-empty bytes: {relative_name}"
            )
        payloads[f"input/dataset/{relative_name}"] = payload
    if len(bundle.terminal_receipts) != 5 or len(bundle.acquisition_receipts) != 5:
        raise ValueError("Report evidence requires exactly five terminal/acquisition receipts")
    for fold, (terminal, acquisition) in enumerate(
        zip(bundle.terminal_receipts, bundle.acquisition_receipts)
    ):
        payloads[f"input/fold-{fold}-terminal-receipt.json"] = _pretty_canonical_bytes(
            terminal
        )
        payloads[f"input/fold-{fold}-acquisition-receipt.json"] = (
            _pretty_canonical_bytes(acquisition)
        )
    return dict(sorted(payloads.items()))


def publish_analysis_bundle(bundle: AnalysisBundle, *, output_dir: Path) -> Mapping[str, Any]:
    """Atomically publish compact analysis artifacts with a final commit marker."""

    if not isinstance(bundle, AnalysisBundle):
        raise TypeError("bundle must be AnalysisBundle")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Analysis output must be absent: {output_dir}")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"Analysis output parent must be a real directory: {parent}")
    incomplete = parent / f".{output_dir.name}.incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Stale incomplete analysis output exists: {incomplete}")
    incomplete.mkdir()
    payloads = _payloads(bundle)
    records: list[dict[str, Any]] = []
    for relative_name, payload in payloads.items():
        path = incomplete / relative_name
        _write_new(path, payload)
        records.append(
            {
                "path": relative_name,
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    figures_dir = incomplete / "figures"
    if figures_dir.is_dir():
        _fsync_directory(figures_dir)
    _fsync_directory(incomplete)
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": REPORT_PROTOCOL,
        "commit_marker": True,
        "files": records,
    }
    _write_new(incomplete / "analysis_manifest.json", _canonical_bytes(manifest))
    _fsync_directory(incomplete)
    _rename_directory_to_absent(incomplete, output_dir)
    _fsync_directory(parent)
    try:
        expected_names = {"analysis_manifest.json", *payloads}
        actual_names = {
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        if actual_names != expected_names:
            raise RuntimeError("Published analysis inventory changed")
        for record in records:
            path = output_dir / record["path"]
            if (
                path.is_symlink()
                or path.stat().st_size != record["size"]
                or _sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError(f"Published analysis readback failed: {record['path']}")
        manifest_sha256 = _sha256_file(output_dir / "analysis_manifest.json")
    except BaseException:
        manifest_path = output_dir / "analysis_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest_path.unlink()
            _fsync_directory(output_dir)
        if not incomplete.exists():
            _rename_directory_to_absent(output_dir, incomplete)
            _fsync_directory(parent)
        raise
    return {
        "output_name": output_dir.name,
        "analysis_manifest_sha256": manifest_sha256,
        "files": len(records),
    }
