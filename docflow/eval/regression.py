"""Compare two evaluation runs: surface per-provider, per-field deltas so a
regression is visible at a glance.

Usage:
    python -m docflow.eval --regression previous.json current.json

Or programmatically via `regression_report(prev_dict, curr_dict)`.
"""

import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def regression_report(prev: dict, curr: dict) -> str:
    """Return a markdown-formatted diff. Per-provider per-metric deltas."""
    lines: list[str] = []
    lines.append("# Evaluation regression diff")
    lines.append("")
    lines.append(f"- **Previous:** {prev.get('started_at', '?')} on `{prev.get('dataset_dir', '?')}`")
    lines.append(f"- **Current:**  {curr.get('started_at', '?')} on `{curr.get('dataset_dir', '?')}`")
    lines.append("")

    prev_pp = prev.get("per_provider", {})
    curr_pp = curr.get("per_provider", {})
    providers = sorted(set(prev_pp) | set(curr_pp))

    lines.append("## Top-line deltas (current - previous)")
    lines.append("")
    lines.append("| Provider | Δ Invoice acc | Δ Critical acc | Δ Halluc | Δ Wrong-but-conf | Δ Avg score |")
    lines.append("|---|---|---|---|---|---|")
    for p in providers:
        prv = prev_pp.get(p)
        cur = curr_pp.get(p)
        if prv is None:
            lines.append(f"| {p} | (new) | (new) | (new) | (new) | (new) |")
            continue
        if cur is None:
            lines.append(f"| {p} | (removed) | — | — | — | — |")
            continue
        lines.append(
            f"| {p} | "
            f"{_pp(cur['invoice_accuracy'] - prv['invoice_accuracy'])} | "
            f"{_pp(cur['critical_field_accuracy'] - prv['critical_field_accuracy'])} | "
            f"{_pp(cur['hallucination_rate'] - prv['hallucination_rate'])} | "
            f"{_pp(cur['wrong_but_confident_rate'] - prv['wrong_but_confident_rate'])} | "
            f"{_f(cur['avg_quality_score'] - prv['avg_quality_score'])} |"
        )
    lines.append("")

    # Per-field deltas — focus on critical fields first.
    from docflow.eval.truth import CRITICAL_FIELDS, TRUTH_FIELDS

    lines.append("## Per-field critical accuracy delta")
    lines.append("")
    header = "| Provider | " + " | ".join(CRITICAL_FIELDS) + " |"
    lines.append(header)
    lines.append("|---" * (len(CRITICAL_FIELDS) + 1) + "|")
    for p in providers:
        prv = prev_pp.get(p, {})
        cur = curr_pp.get(p, {})
        prv_f = prv.get("critical_field_accuracy_per_field", {})
        cur_f = cur.get("critical_field_accuracy_per_field", {})
        row = [p]
        for f in CRITICAL_FIELDS:
            if f in prv_f and f in cur_f:
                row.append(_pp(cur_f[f] - prv_f[f]))
            elif f in cur_f:
                row.append("(new)")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-file score changes for invoices that exist in both runs.
    prev_by_key = {(i["invoice_file"], i["provider"]): i for i in prev.get("invoices", [])}
    curr_by_key = {(i["invoice_file"], i["provider"]): i for i in curr.get("invoices", [])}
    common = sorted(set(prev_by_key) & set(curr_by_key))
    score_changes = []
    for k in common:
        d = curr_by_key[k]["extracted_quality_score"] - prev_by_key[k]["extracted_quality_score"]
        if abs(d) >= 0.05:
            score_changes.append((k[0], k[1], prev_by_key[k]["extracted_quality_score"],
                                 curr_by_key[k]["extracted_quality_score"], d))

    if score_changes:
        lines.append("## Per-file quality_score changes ≥ ±0.05")
        lines.append("")
        lines.append("| Invoice | Provider | Previous | Current | Δ |")
        lines.append("|---|---|---|---|---|")
        for f, p, prv_s, cur_s, d in sorted(score_changes, key=lambda x: abs(x[4]), reverse=True)[:60]:
            lines.append(f"| {f} | {p} | {prv_s:.2f} | {cur_s:.2f} | {_f(d)} |")
        lines.append("")
    else:
        lines.append("_No per-file score changes ≥ ±0.05._\n")

    return "\n".join(lines)


def compare_runs(prev_path: Path, curr_path: Path, out: Path | None = None) -> str:
    """Load two JSON dumps, produce markdown diff. Optionally write to `out`."""
    prev = _load(Path(prev_path))
    curr = _load(Path(curr_path))
    md = regression_report(prev, curr)
    if out is not None:
        Path(out).write_text(md, encoding="utf-8")
    return md


def _pp(x: float) -> str:
    """Format a fraction (0..1) delta as signed percentage points."""
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}pp"


def _f(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}"
