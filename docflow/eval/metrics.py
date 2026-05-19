"""Field-by-field comparison of extracted InvoiceData vs InvoiceTruth.

Single-invoice comparison produces a list of FieldComparison records — one
per truth field. Aggregation rolls those up into AggregateMetrics across
multiple invoices and/or providers.

Match levels (per field):
    exact_match       — string/value equality (after both sides are str()'d
                        for strings; exact numeric equality for numbers)
    normalized_match  — equal after whitespace collapse + case-fold;
                        numerics: equal within tolerance
    mismatch          — both sides present, different value
    missing           — truth has a value, extracted is None
    hallucinated      — truth explicitly null (key present in JSON, value null),
                        extracted produced a value → the model invented data
    not_in_truth      — truth file omits this key entirely → not graded

A field is "matched" if exact_match OR normalized_match.

Each FieldComparison also carries:
    derived     — extracted value was filled by normalization (not extraction)
    confidence  — per-field weight from quality scoring (the signal that
                  covers this field; shared signals split the credit)
"""

import re
from dataclasses import dataclass, field as dc_field

from docflow.eval.truth import CRITICAL_FIELDS, InvoiceTruth, TRUTH_FIELDS


_NUMERIC_FIELDS = {"net_amount", "vat_amount", "total_to_pay"}
_NUMERIC_TOLERANCE = 0.01  # 1 cent / stotinka


@dataclass
class FieldComparison:
    field: str
    truth_value: object       # what the human said
    extracted_value: object   # what the model produced
    status: str               # exact_match | normalized_match | mismatch |
                              # missing | hallucinated | not_in_truth
    derived: bool = False     # was extracted value produced by normalization?
    critical: bool = False
    confidence: float = 0.0   # per-field signal weight from quality scoring


@dataclass
class InvoiceComparison:
    invoice_file: str         # filename for reporting
    provider: str             # which extractor produced the data
    fields: list[FieldComparison]
    extracted_quality_score: float = 0.0
    extracted_quality_coverage: float = 0.0
    validation_error_codes: list[str] = dc_field(default_factory=list)
    provider_error: str | None = None
    processing_time_s: float | None = None

    @property
    def matched_fields(self) -> list[FieldComparison]:
        return [f for f in self.fields if f.status in ("exact_match", "normalized_match")]

    @property
    def all_truth_fields_matched(self) -> bool:
        """True when every graded (truth-present) field is matched."""
        graded = [f for f in self.fields if f.status != "not_in_truth"]
        if not graded:
            return False
        return all(f.status in ("exact_match", "normalized_match") for f in graded)


@dataclass
class AggregateMetrics:
    """Roll-up across one or many invoices. Single-provider when `provider`
    is set; multi-provider comparisons live in AggregateMetrics per provider."""
    provider: str
    invoice_count: int
    invoice_accuracy: float                       # % of invoices where all graded fields matched
    critical_field_accuracy: float                # % of (invoice × critical field) cells that matched
    field_accuracy: dict[str, float]              # per-field accuracy %
    critical_field_accuracy_per_field: dict[str, float]  # per critical-field accuracy %
    hallucination_rate: float                     # mismatches / total graded cells
    wrong_but_confident_rate: float               # % of invoices where quality_score>=0.7 but not all-matched
    validation_error_rate: float                  # % of invoices with at least one validation error
    avg_quality_score: float
    avg_quality_coverage: float
    avg_processing_time_s: float | None
    provider_failure_count: int


# ─── Field-value normalization for comparison ────────────────────────────────

def _normalize_string(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _values_match(field: str, truth, extracted, *, truth_has_key: bool) -> str:
    """Return one of exact_match | normalized_match | mismatch | missing |
    hallucinated | not_in_truth.

    truth_has_key distinguishes 'key omitted from JSON' (not graded) from
    'key present, value null' (extractor must also produce None or we call
    it a hallucination)."""
    if not truth_has_key:
        return "not_in_truth"
    if truth is None:
        # Truth says: this field must be absent.
        if extracted is None or extracted == "":
            return "exact_match"
        return "hallucinated"
    if extracted is None or extracted == "":
        return "missing"

    if field in _NUMERIC_FIELDS:
        try:
            if abs(float(truth) - float(extracted)) <= _NUMERIC_TOLERANCE:
                return "exact_match" if float(truth) == float(extracted) else "normalized_match"
            return "mismatch"
        except (TypeError, ValueError):
            return "mismatch"

    if str(truth) == str(extracted):
        return "exact_match"
    if _normalize_string(truth) == _normalize_string(extracted):
        return "normalized_match"
    return "mismatch"


def _line_items_match(field: str, truth_items, extracted_items, *, truth_has_key: bool) -> str:
    """Line items match if counts match and each truth item finds a matching
    extracted item by description (normalized substring) and total.

    This is intentionally lenient — line_item order and exact descriptions
    vary by provider. Match is "good enough" not "perfect."
    """
    if not truth_has_key:
        return "not_in_truth"
    if truth_items is None:
        if not extracted_items:
            return "exact_match"
        return "hallucinated"
    if not extracted_items:
        return "missing"
    if len(truth_items) != len(extracted_items):
        return "mismatch"

    ex_remaining = list(extracted_items)
    matched_norm = True
    matched_exact = True
    for ti in truth_items:
        t_total = ti.total_without_vat
        t_desc_n = _normalize_string(ti.description)
        candidate = None
        for ei in ex_remaining:
            e_total = getattr(ei, "total_without_vat", None)
            totals_close = (
                t_total is not None and e_total is not None
                and abs(float(t_total) - float(e_total)) <= _NUMERIC_TOLERANCE
            )
            descs_close = (
                t_desc_n and _normalize_string(getattr(ei, "description", "")) and
                (t_desc_n in _normalize_string(getattr(ei, "description", ""))
                 or _normalize_string(getattr(ei, "description", "")) in t_desc_n)
            )
            if totals_close or descs_close:
                candidate = ei
                break
        if candidate is None:
            return "mismatch"
        ex_remaining.remove(candidate)
        if t_total is not None and getattr(candidate, "total_without_vat", None) is not None:
            if float(t_total) != float(candidate.total_without_vat):
                matched_exact = False
        if ti.description and getattr(candidate, "description", None):
            if str(ti.description) != str(candidate.description):
                matched_exact = False
    return "exact_match" if matched_exact else "normalized_match"


# ─── Extract a field value from InvoiceData by truth-field name ──────────────

def _extracted_value_for(field: str, inv) -> object:
    """Map truth-field-name → InvoiceData field. Same naming choices used by
    docflow.columns.get_row_value, but here we return the raw value (None when
    absent) — not a display-fallback string."""
    if inv is None:
        return None
    s = inv.supplier
    if field == "supplier_name":    return s.name if s else None
    if field == "supplier_eik":     return s.eik if s else None
    if field == "supplier_vat":     return s.vat_number if s else None
    if field == "iban":             return inv.iban
    if field == "invoice_number":   return inv.invoice_number
    if field == "issue_date":       return inv.issue_date
    if field == "net_amount":       return inv.net_amount
    if field == "vat_amount":
        if inv.vat_breakdown:
            return round(sum(v.vat_amount for v in inv.vat_breakdown), 2)
        return None
    if field == "total_to_pay":     return inv.total_to_pay
    if field == "currency":         return inv.currency
    if field == "payment_method":   return inv.payment_method
    if field == "line_items":       return inv.line_items
    return None


def _derived_paths_for(field: str) -> tuple[str, ...]:
    """Which InvoiceData.derived_fields paths indicate this truth field was derived?"""
    if field == "net_amount":     return ("net_amount",)
    if field == "vat_amount":     return ("vat_breakdown",)
    if field == "total_to_pay":   return ("total_to_pay",)
    return ()


# Map truth-field → quality signal that covers it. Shared signals (supplier_tax_id
# covers both eik and vat; any_total covers net_amount and total_to_pay) split
# their weight evenly across the truth fields that share them so per-field
# confidence sums correctly.
_TRUTH_FIELD_TO_SIGNAL: dict[str, tuple[str, float]] = {
    "supplier_name":   ("supplier_name",   1.0),
    "supplier_eik":    ("supplier_tax_id", 0.5),  # shared with supplier_vat
    "supplier_vat":    ("supplier_tax_id", 0.5),
    "iban":            ("iban",            1.0),
    "invoice_number":  ("invoice_number",  1.0),
    "issue_date":      ("issue_date",      1.0),
    "net_amount":      ("any_total",       0.5),  # shared with total_to_pay
    "vat_amount":      ("vat_info",        1.0),
    "total_to_pay":    ("any_total",       0.5),
    "currency":        ("currency",        1.0),
    "payment_method":  ("",                0.0),  # not in scoring
    "line_items":      ("line_items",      1.0),
}


def _field_confidence(field: str, breakdown: dict) -> float:
    """Effective per-field confidence: signal weight × share fraction. If the
    signal was discounted (derived), the breakdown already reflects that."""
    signal, share = _TRUTH_FIELD_TO_SIGNAL.get(field, ("", 0.0))
    if not signal:
        return 0.0
    # Breakdown may key the signal either plain or with '(derived)' suffix.
    weight = breakdown.get(signal, breakdown.get(f"{signal}(derived)", 0.0))
    if not isinstance(weight, (int, float)):
        return 0.0
    return round(weight * share, 4)


# ─── Public comparison entrypoint ────────────────────────────────────────────

def compare_invoice(
    extracted_doc,
    truth: InvoiceTruth,
    *,
    invoice_file: str,
    provider: str,
    validation_findings=None,
    provider_error: str | None = None,
    processing_time_s: float | None = None,
) -> InvoiceComparison:
    """Compare extracted InvoiceData vs truth, returning per-field results."""
    inv = extracted_doc.invoice if extracted_doc is not None else None
    derived_set = set(inv.derived_fields) if inv is not None else set()

    # Per-field confidence comes from the quality breakdown. Compute once.
    from docflow.quality import invoice_quality_breakdown
    _, breakdown = invoice_quality_breakdown(inv, validation_findings) if inv is not None else (0.0, {})

    fields: list[FieldComparison] = []
    truth_dict = truth.model_dump() if truth is not None else {}

    for fname in TRUTH_FIELDS:
        truth_has_key = truth.has_field(fname) if truth is not None else False
        truth_value = truth_dict.get(fname)
        extracted_value = _extracted_value_for(fname, inv)
        if fname == "line_items":
            from docflow.eval.truth import TruthLineItem
            truth_items = None
            if isinstance(truth_value, list):
                truth_items = [TruthLineItem(**i) if isinstance(i, dict) else i for i in truth_value]
            status = _line_items_match(fname, truth_items, extracted_value, truth_has_key=truth_has_key)
            display_truth = truth_items
        else:
            status = _values_match(fname, truth_value, extracted_value, truth_has_key=truth_has_key)
            display_truth = truth_value

        derived = any(p in derived_set for p in _derived_paths_for(fname))
        confidence = _field_confidence(fname, breakdown)

        fields.append(FieldComparison(
            field=fname,
            truth_value=display_truth,
            extracted_value=extracted_value,
            status=status,
            derived=derived,
            critical=fname in CRITICAL_FIELDS,
            confidence=confidence,
        ))

    return InvoiceComparison(
        invoice_file=invoice_file,
        provider=provider,
        fields=fields,
        extracted_quality_score=extracted_doc.quality_score if extracted_doc else 0.0,
        extracted_quality_coverage=getattr(extracted_doc, "quality_coverage", 0.0),
        validation_error_codes=sorted({
            f.code for f in (validation_findings or [])
            if getattr(f, "level", None) == "error"
        }),
        provider_error=provider_error,
        processing_time_s=processing_time_s,
    )


# ─── Aggregation ─────────────────────────────────────────────────────────────

_CONFIDENT_THRESHOLD = 0.7  # quality_score above which we consider the result "confident"


def aggregate(comparisons: list[InvoiceComparison], provider: str) -> AggregateMetrics:
    """Roll up per-invoice comparisons into a single AggregateMetrics record."""
    own = [c for c in comparisons if c.provider == provider]
    n = len(own)
    if n == 0:
        return AggregateMetrics(
            provider=provider, invoice_count=0, invoice_accuracy=0.0,
            critical_field_accuracy=0.0, field_accuracy={}, critical_field_accuracy_per_field={},
            hallucination_rate=0.0, wrong_but_confident_rate=0.0,
            validation_error_rate=0.0, avg_quality_score=0.0, avg_quality_coverage=0.0,
            avg_processing_time_s=None, provider_failure_count=0,
        )

    provider_failures = [c for c in own if c.provider_error is not None]
    successful = [c for c in own if c.provider_error is None]

    # Per-field accuracy: % matched out of graded.
    field_acc: dict[str, float] = {}
    for fname in TRUTH_FIELDS:
        graded = [c for c in successful for f in c.fields
                  if f.field == fname and f.status != "not_in_truth"]
        # graded is list of InvoiceComparison; each contributes once via the field check.
        # Re-compute properly:
        graded_count = 0
        matched_count = 0
        for c in successful:
            for f in c.fields:
                if f.field != fname:
                    continue
                if f.status == "not_in_truth":
                    continue
                graded_count += 1
                if f.status in ("exact_match", "normalized_match"):
                    matched_count += 1
        field_acc[fname] = round(matched_count / graded_count, 4) if graded_count else 0.0

    # Critical field accuracy (per field + aggregate).
    crit_per_field: dict[str, float] = {f: field_acc[f] for f in CRITICAL_FIELDS}
    crit_graded = 0
    crit_matched = 0
    for c in successful:
        for f in c.fields:
            if not f.critical or f.status == "not_in_truth":
                continue
            crit_graded += 1
            if f.status in ("exact_match", "normalized_match"):
                crit_matched += 1
    critical_acc = round(crit_matched / crit_graded, 4) if crit_graded else 0.0

    # Invoice accuracy: % of successful invoices where every graded field matched.
    full_matches = sum(1 for c in successful if c.all_truth_fields_matched)
    invoice_acc = round(full_matches / len(successful), 4) if successful else 0.0

    # Hallucination rate: (mismatch + hallucinated) / total graded cells.
    # 'mismatch' = extractor produced a wrong value where truth had one.
    # 'hallucinated' = extractor produced a value where truth said null.
    # Both are "the model invented data that disagrees with reality."
    total_graded = 0
    total_hallucinated = 0
    for c in successful:
        for f in c.fields:
            if f.status == "not_in_truth":
                continue
            total_graded += 1
            if f.status in ("mismatch", "hallucinated"):
                total_hallucinated += 1
    halluc_rate = round(total_hallucinated / total_graded, 4) if total_graded else 0.0

    # Wrong-but-confident: invoices with high quality_score but truth mismatches.
    confident = [c for c in successful if c.extracted_quality_score >= _CONFIDENT_THRESHOLD]
    confident_wrong = sum(1 for c in confident if not c.all_truth_fields_matched)
    wbc_rate = round(confident_wrong / len(confident), 4) if confident else 0.0

    validation_err_count = sum(1 for c in successful if c.validation_error_codes)
    validation_rate = round(validation_err_count / len(successful), 4) if successful else 0.0

    avg_score = round(sum(c.extracted_quality_score for c in successful) / len(successful), 4) if successful else 0.0
    avg_cov   = round(sum(c.extracted_quality_coverage for c in successful) / len(successful), 4) if successful else 0.0

    latencies = [c.processing_time_s for c in own if c.processing_time_s is not None]
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None

    return AggregateMetrics(
        provider=provider,
        invoice_count=n,
        invoice_accuracy=invoice_acc,
        critical_field_accuracy=critical_acc,
        field_accuracy=field_acc,
        critical_field_accuracy_per_field=crit_per_field,
        hallucination_rate=halluc_rate,
        wrong_but_confident_rate=wbc_rate,
        validation_error_rate=validation_rate,
        avg_quality_score=avg_score,
        avg_quality_coverage=avg_cov,
        avg_processing_time_s=avg_lat,
        provider_failure_count=len(provider_failures),
    )
