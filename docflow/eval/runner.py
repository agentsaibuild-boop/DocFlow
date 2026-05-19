"""Run the extraction → comparison → aggregation pipeline against a golden
dataset. Producer of the raw `EvaluationResult` data that report.py consumes.
"""

import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path

from docflow.eval.metrics import (
    AggregateMetrics, InvoiceComparison, aggregate, compare_invoice,
)
from docflow.eval.truth import InvoiceTruth, discover_dataset, load_truth
from docflow.pipeline import ProviderError, extract
from docflow.registry import SupplierRegistry
from docflow.validators import validate


@dataclass
class EvaluationResult:
    """Outcome of one full evaluation run (dataset × providers)."""
    started_at: str
    dataset_dir: str
    providers: list[str]
    invoices: list[InvoiceComparison]
    per_provider: dict[str, AggregateMetrics]
    sample_count: int  # graded invoices in dataset
    notes: str = ""


def run_evaluation(
    dataset_dir: Path,
    providers: list[str],
    *,
    allow_fallback: bool | None = False,
    verbose: bool = True,
) -> EvaluationResult:
    """For every (invoice, truth) pair in `dataset_dir`, run each provider on
    the invoice, validate, score, and compare to truth.

    allow_fallback defaults to False for benchmark integrity — when measuring
    provider X, a fallback to provider Y would taint X's row in the report.
    Pass True only if you explicitly want chained behavior.
    """
    pairs = discover_dataset(Path(dataset_dir))
    if not pairs:
        return EvaluationResult(
            started_at=datetime.now().isoformat(timespec="seconds"),
            dataset_dir=str(dataset_dir),
            providers=list(providers),
            invoices=[],
            per_provider={p: aggregate([], p) for p in providers},
            sample_count=0,
            notes=f"No (invoice, truth) pairs found in {dataset_dir}",
        )

    registry = SupplierRegistry()  # enrichment OK, no writes (human_confirmed=False)
    all_comparisons: list[InvoiceComparison] = []

    for invoice_path, truth_path in pairs:
        try:
            truth = load_truth(truth_path)
        except Exception as e:
            if verbose:
                print(f"  ! skip {invoice_path.name}: bad truth ({e})", flush=True)
            continue

        for provider in providers:
            comp = _one_run(invoice_path, truth, provider, registry, allow_fallback, verbose)
            all_comparisons.append(comp)

    per_provider = {p: aggregate(all_comparisons, p) for p in providers}

    return EvaluationResult(
        started_at=datetime.now().isoformat(timespec="seconds"),
        dataset_dir=str(dataset_dir),
        providers=list(providers),
        invoices=all_comparisons,
        per_provider=per_provider,
        sample_count=len(pairs),
    )


def _one_run(invoice_path, truth: InvoiceTruth, provider: str,
             registry: SupplierRegistry, allow_fallback, verbose) -> InvoiceComparison:
    t0 = time.time()
    doc = None
    provider_err: str | None = None
    validation_findings = []
    try:
        doc = extract(invoice_path, provider=provider, allow_fallback=allow_fallback)
        if doc.invoice:
            doc.invoice, _ = registry.enrich(doc.invoice)
        validation_findings = validate(doc)
        # Re-score with validation so capped numbers show up in the report.
        from docflow.quality import invoice_quality_coverage, invoice_quality_score
        doc.quality_score = invoice_quality_score(doc.invoice, validation_findings)
        doc.quality_coverage = invoice_quality_coverage(doc.invoice)
    except ProviderError as pe:
        provider_err = str(pe)
    except Exception as e:
        provider_err = f"{type(e).__name__}: {e}"

    elapsed = round(time.time() - t0, 2)
    comp = compare_invoice(
        doc, truth,
        invoice_file=invoice_path.name,
        provider=provider,
        validation_findings=validation_findings,
        provider_error=provider_err,
        processing_time_s=elapsed,
    )
    if verbose:
        matched = sum(1 for f in comp.fields if f.status in ("exact_match", "normalized_match"))
        graded = sum(1 for f in comp.fields if f.status != "not_in_truth")
        err = f" ✗ {provider_err[:40]}" if provider_err else ""
        print(f"  [{provider:25s}] {invoice_path.name}: matched {matched}/{graded} in {elapsed}s{err}",
              flush=True)
    return comp


def evaluation_result_to_dict(r: EvaluationResult) -> dict:
    """Serialize EvaluationResult to plain dicts for JSON / regression diff."""
    def _comp(c: InvoiceComparison) -> dict:
        return {
            "invoice_file": c.invoice_file,
            "provider": c.provider,
            "extracted_quality_score": c.extracted_quality_score,
            "extracted_quality_coverage": c.extracted_quality_coverage,
            "validation_error_codes": list(c.validation_error_codes),
            "provider_error": c.provider_error,
            "processing_time_s": c.processing_time_s,
            "fields": [
                {
                    "field": f.field,
                    "status": f.status,
                    "derived": f.derived,
                    "critical": f.critical,
                    "confidence": f.confidence,
                    "truth_value": _safe_serialize(f.truth_value),
                    "extracted_value": _safe_serialize(f.extracted_value),
                }
                for f in c.fields
            ],
        }

    def _agg(a: AggregateMetrics) -> dict:
        return {
            "provider": a.provider,
            "invoice_count": a.invoice_count,
            "invoice_accuracy": a.invoice_accuracy,
            "critical_field_accuracy": a.critical_field_accuracy,
            "field_accuracy": a.field_accuracy,
            "critical_field_accuracy_per_field": a.critical_field_accuracy_per_field,
            "hallucination_rate": a.hallucination_rate,
            "wrong_but_confident_rate": a.wrong_but_confident_rate,
            "validation_error_rate": a.validation_error_rate,
            "avg_quality_score": a.avg_quality_score,
            "avg_quality_coverage": a.avg_quality_coverage,
            "avg_processing_time_s": a.avg_processing_time_s,
            "provider_failure_count": a.provider_failure_count,
        }

    return {
        "started_at": r.started_at,
        "dataset_dir": r.dataset_dir,
        "providers": r.providers,
        "sample_count": r.sample_count,
        "notes": r.notes,
        "per_provider": {p: _agg(a) for p, a in r.per_provider.items()},
        "invoices": [_comp(c) for c in r.invoices],
    }


def _safe_serialize(v):
    """Best-effort JSON-compatible value (truth and extracted values can be
    Pydantic models or InvoiceData line items)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [_safe_serialize(x) for x in v]
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return str(v)
