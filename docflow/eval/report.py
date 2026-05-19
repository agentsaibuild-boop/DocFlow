"""Report writers: markdown, CSV, XLSX. JSON dump is handled by runner.

Produces one set of files per evaluation run:

  eval_report.md         human summary (provider comparison + critical-field table)
  eval_report.csv        long form: one row per (invoice, provider, field)
  eval_report.xlsx       multi-sheet: Summary, Per-invoice, Per-field, Providers
  eval_report.json       raw EvaluationResult dump (used by regression.py)
"""

import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

from docflow.eval.metrics import InvoiceComparison
from docflow.eval.runner import EvaluationResult, evaluation_result_to_dict
from docflow.eval.truth import CRITICAL_FIELDS, TRUTH_FIELDS


# ─── JSON ────────────────────────────────────────────────────────────────────

def write_json(result: EvaluationResult, out: Path) -> None:
    Path(out).write_text(
        json.dumps(evaluation_result_to_dict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── Markdown ────────────────────────────────────────────────────────────────

def write_markdown(result: EvaluationResult, out: Path) -> None:
    lines: list[str] = []
    lines.append(f"# DocFlow Evaluation Report")
    lines.append(f"")
    lines.append(f"- **Run at:** {result.started_at}")
    lines.append(f"- **Dataset:** `{result.dataset_dir}`")
    lines.append(f"- **Graded invoices:** {result.sample_count}")
    lines.append(f"- **Providers tested:** {', '.join(result.providers)}")
    if result.notes:
        lines.append(f"- **Notes:** {result.notes}")
    lines.append("")

    # Provider comparison summary.
    lines.append("## Provider comparison")
    lines.append("")
    lines.append("| Provider | n | Invoice acc | Critical acc | Halluc rate | Wrong-but-confident | Avg score | Avg coverage | Avg latency | Provider failures |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p, a in result.per_provider.items():
        lat = f"{a.avg_processing_time_s}s" if a.avg_processing_time_s is not None else "—"
        lines.append(
            f"| {p} | {a.invoice_count} | "
            f"{a.invoice_accuracy:.0%} | {a.critical_field_accuracy:.0%} | "
            f"{a.hallucination_rate:.0%} | {a.wrong_but_confident_rate:.0%} | "
            f"{a.avg_quality_score:.2f} | {a.avg_quality_coverage:.2f} | "
            f"{lat} | {a.provider_failure_count} |"
        )
    lines.append("")

    # Critical-field-per-provider accuracy.
    lines.append("## Critical-field accuracy (per provider)")
    lines.append("")
    header = "| Provider | " + " | ".join(CRITICAL_FIELDS) + " |"
    sep = "|---" * (len(CRITICAL_FIELDS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for p, a in result.per_provider.items():
        row = [p] + [f"{a.critical_field_accuracy_per_field.get(f, 0.0):.0%}" for f in CRITICAL_FIELDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Full per-field accuracy.
    lines.append("## All fields (per provider)")
    lines.append("")
    header = "| Provider | " + " | ".join(TRUTH_FIELDS) + " |"
    sep = "|---" * (len(TRUTH_FIELDS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for p, a in result.per_provider.items():
        row = [p] + [f"{a.field_accuracy.get(f, 0.0):.0%}" for f in TRUTH_FIELDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Notable mismatches — surface the cases that need attention.
    notable = _collect_notable_mismatches(result.invoices)
    if notable:
        lines.append("## Notable critical-field problems")
        lines.append("")
        lines.append("| Invoice | Provider | Field | Status | Truth | Extracted |")
        lines.append("|---|---|---|---|---|---|")
        for row in notable:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    Path(out).write_text("\n".join(lines), encoding="utf-8")


def _collect_notable_mismatches(invoices: list[InvoiceComparison]) -> list[list[str]]:
    """Critical-field problems: mismatch, missing, or hallucinated."""
    out: list[list[str]] = []
    for c in invoices:
        for f in c.fields:
            if not f.critical:
                continue
            if f.status not in ("mismatch", "missing", "hallucinated"):
                continue
            t = _short(f.truth_value)
            e = _short(f.extracted_value)
            out.append([c.invoice_file, c.provider, f.field, f.status, t, e])
    return out[:60]


def _short(v) -> str:
    if v is None:
        return "—"
    s = str(v)
    return s[:35]


# ─── CSV — long form, one row per (invoice, provider, field) ─────────────────

def write_csv(result: EvaluationResult, out: Path) -> None:
    cols = [
        "invoice_file", "provider", "field", "status",
        "critical", "derived", "confidence",
        "truth_value", "extracted_value",
        "extracted_quality_score", "extracted_quality_coverage",
        "validation_error_codes", "provider_error", "processing_time_s",
    ]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in result.invoices:
            for f in c.fields:
                w.writerow([
                    c.invoice_file, c.provider, f.field, f.status,
                    f.critical, f.derived, f.confidence,
                    _short(f.truth_value), _short(f.extracted_value),
                    c.extracted_quality_score, c.extracted_quality_coverage,
                    ";".join(c.validation_error_codes),
                    c.provider_error or "",
                    c.processing_time_s if c.processing_time_s is not None else "",
                ])


# ─── XLSX — multi-sheet ──────────────────────────────────────────────────────

def write_xlsx(result: EvaluationResult, out: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)

    # 1) Summary sheet
    ws = wb.create_sheet("Summary")
    ws.append(["Provider", "n", "Invoice acc", "Critical acc",
               "Halluc rate", "Wrong-but-confident", "Avg score",
               "Avg coverage", "Avg latency (s)", "Provider failures"])
    for cell in ws[1]: cell.font = bold
    for p, a in result.per_provider.items():
        ws.append([
            p, a.invoice_count,
            a.invoice_accuracy, a.critical_field_accuracy,
            a.hallucination_rate, a.wrong_but_confident_rate,
            a.avg_quality_score, a.avg_quality_coverage,
            a.avg_processing_time_s, a.provider_failure_count,
        ])

    # 2) Critical fields per provider
    ws = wb.create_sheet("Critical fields")
    header = ["Provider"] + list(CRITICAL_FIELDS)
    ws.append(header)
    for cell in ws[1]: cell.font = bold
    for p, a in result.per_provider.items():
        ws.append([p] + [a.critical_field_accuracy_per_field.get(f, 0.0) for f in CRITICAL_FIELDS])

    # 3) All fields per provider
    ws = wb.create_sheet("All fields")
    header = ["Provider"] + list(TRUTH_FIELDS)
    ws.append(header)
    for cell in ws[1]: cell.font = bold
    for p, a in result.per_provider.items():
        ws.append([p] + [a.field_accuracy.get(f, 0.0) for f in TRUTH_FIELDS])

    # 4) Per-invoice details
    ws = wb.create_sheet("Per-invoice")
    ws.append([
        "Invoice", "Provider", "Quality score", "Coverage",
        "Validation errors", "Provider error", "t (s)",
        "Matched / Graded", "Critical matched / total",
    ])
    for cell in ws[1]: cell.font = bold
    for c in result.invoices:
        graded = sum(1 for f in c.fields if f.status != "not_in_truth")
        matched = sum(1 for f in c.fields if f.status in ("exact_match", "normalized_match"))
        crit_g = sum(1 for f in c.fields if f.critical and f.status != "not_in_truth")
        crit_m = sum(1 for f in c.fields if f.critical and f.status in ("exact_match", "normalized_match"))
        ws.append([
            c.invoice_file, c.provider, c.extracted_quality_score, c.extracted_quality_coverage,
            ";".join(c.validation_error_codes),
            (c.provider_error or "")[:60], c.processing_time_s,
            f"{matched}/{graded}", f"{crit_m}/{crit_g}",
        ])

    # 5) Per-field long form
    ws = wb.create_sheet("Fields")
    ws.append(["Invoice", "Provider", "Field", "Status",
               "Critical", "Derived", "Confidence",
               "Truth", "Extracted"])
    for cell in ws[1]: cell.font = bold
    for c in result.invoices:
        for f in c.fields:
            ws.append([
                c.invoice_file, c.provider, f.field, f.status,
                f.critical, f.derived, f.confidence,
                _short(f.truth_value), _short(f.extracted_value),
            ])

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 30

    wb.save(out)


def write_all(result: EvaluationResult, output_dir: Path, basename: str = "eval_report") -> dict:
    """Write json + md + csv + xlsx. Returns paths produced."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{basename}.json",
        "md":   output_dir / f"{basename}.md",
        "csv":  output_dir / f"{basename}.csv",
        "xlsx": output_dir / f"{basename}.xlsx",
    }
    write_json(result, paths["json"])
    write_markdown(result, paths["md"])
    write_csv(result, paths["csv"])
    write_xlsx(result, paths["xlsx"])
    return paths
