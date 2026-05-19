"""Invoice quality scoring: coverage, confidence, breakdown, validation caps.

Two distinct concepts:

  coverage   — did we get the field at all? (extracted OR derived counts)
  confidence — do we trust the value? (extraction full weight; derivation
               discounted by _DERIVATION_DISCOUNT; then capped by hard
               validation errors)

Public entry points:

  invoice_quality_score(inv, validation_findings=None) -> float
      Returns *confidence*. This is the number routing / display use.

  invoice_quality_breakdown(inv, validation_findings=None) -> tuple[float, dict]
      Returns (confidence, breakdown). Breakdown contains per-signal entries
      (with `(derived)` suffix when discounted), plus _coverage, _confidence,
      _capped_to / _cap_reason when applicable.

Validation caps (applied to confidence, lowest wins):
  • math_* error             → 0.5
  • iban_checksum error      → 0.6
  • iban_missing_for_bank_payment → 0.6
"""

from docflow.schema import InvoiceData


# Per-signal weights, summing to 1.0.
_WEIGHTS = {
    "supplier_name":     0.15,
    "supplier_tax_id":   0.10,
    "invoice_number":    0.10,
    "issue_date":        0.10,
    "any_total":         0.15,
    "line_items":        0.10,
    "customer_name":     0.10,
    "vat_info":          0.10,
    "iban":              0.05,
    "currency":          0.05,
}

# Derivation discount: how much weight a derived signal contributes vs extracted.
# 0.5 means a normalisation-produced value counts as half a real one.
_DERIVATION_DISCOUNT = 0.5

# Validation-error → score cap. Lowest applicable cap wins.
_VALIDATION_CAPS: dict[str, float] = {
    "math_*":                          0.5,
    "iban_checksum":                   0.6,
    "iban_missing_for_bank_payment":   0.6,
}


def _present(value) -> bool:
    return value is not None and value != ""


def _signal_was_derived(signal_key: str, inv: InvoiceData) -> bool:
    """Was the signal's underlying data produced by normalization rather than
    extraction? Only signals with derivable substrate need answering — others
    return False unconditionally.

    For composite signals (any_total — could come from net_amount OR total_to_pay),
    we treat it as derived only when EVERY present source field is derived.
    A single extracted source grounds the signal."""
    derived = inv.derived_fields  # tuple of dotted paths
    if signal_key == "any_total":
        # Signal fires when at least one of these is present.
        if inv.net_amount is not None and "net_amount" not in derived:
            return False
        if inv.total_to_pay is not None and "total_to_pay" not in derived:
            return False
        return True  # every present value was derived
    if signal_key == "vat_info":
        return "vat_breakdown" in derived
    return False


def _compute_signals(inv: InvoiceData) -> dict[str, tuple[float, bool]]:
    """Return present signals as {key: (full_weight, was_derived)}."""
    signals: dict[str, tuple[float, bool]] = {}

    if inv.supplier and _present(inv.supplier.name):
        signals["supplier_name"] = (_WEIGHTS["supplier_name"], False)

    if inv.supplier and (_present(inv.supplier.eik) or _present(inv.supplier.vat_number)):
        signals["supplier_tax_id"] = (_WEIGHTS["supplier_tax_id"], False)

    if _present(inv.invoice_number):
        signals["invoice_number"] = (_WEIGHTS["invoice_number"], False)

    if _present(inv.issue_date):
        signals["issue_date"] = (_WEIGHTS["issue_date"], False)

    if _present(inv.total_to_pay) or _present(inv.net_amount):
        signals["any_total"] = (_WEIGHTS["any_total"], _signal_was_derived("any_total", inv))

    if inv.line_items:
        signals["line_items"] = (_WEIGHTS["line_items"], False)

    if inv.customer and _present(inv.customer.name):
        signals["customer_name"] = (_WEIGHTS["customer_name"], False)

    has_vat_breakdown = bool(inv.vat_breakdown)
    derivable = bool(
        inv.line_items
        and any(li.vat_percent is not None and li.total_without_vat is not None
                for li in inv.line_items)
    )
    if has_vat_breakdown or derivable:
        signals["vat_info"] = (_WEIGHTS["vat_info"], _signal_was_derived("vat_info", inv))

    if _present(inv.iban):
        signals["iban"] = (_WEIGHTS["iban"], False)

    if _present(inv.currency):
        signals["currency"] = (_WEIGHTS["currency"], False)

    return signals


def _validation_cap(findings) -> tuple[float | None, str | None]:
    cap: float | None = None
    cap_code: str | None = None
    for f in findings or []:
        if getattr(f, "level", None) != "error":
            continue
        code = getattr(f, "code", "") or ""
        candidate: float | None = None
        if code.startswith("math_"):
            candidate = _VALIDATION_CAPS["math_*"]
        elif code in _VALIDATION_CAPS:
            candidate = _VALIDATION_CAPS[code]
        if candidate is None:
            continue
        if cap is None or candidate < cap:
            cap = candidate
            cap_code = code
    return cap, cap_code


def invoice_quality_breakdown(
    inv: InvoiceData | None,
    validation_findings=None,
) -> tuple[float, dict]:
    """Return (confidence, breakdown).

    Breakdown contains:
      • per-signal contributions, key has '(derived)' suffix when discounted,
        value is the EFFECTIVE weight (already × discount if applicable)
      • '_coverage'      — sum of full weights (no discount, no cap)
      • '_confidence'    — sum after discount, before cap
      • '_capped_to'     — present only when a cap fired
      • '_cap_reason'    — present only when a cap fired
    """
    if inv is None:
        return 0.0, {}

    signals = _compute_signals(inv)
    coverage = round(sum(w for w, _ in signals.values()), 4)
    confidence_raw = round(
        sum(w * (_DERIVATION_DISCOUNT if d else 1.0) for w, d in signals.values()),
        4,
    )

    breakdown: dict = {}
    for key, (w, derived) in signals.items():
        effective = round(w * (_DERIVATION_DISCOUNT if derived else 1.0), 4)
        breakdown[f"{key}(derived)" if derived else key] = effective
    breakdown["_coverage"] = coverage
    breakdown["_confidence"] = confidence_raw

    cap, cap_code = _validation_cap(validation_findings)
    if cap is not None and cap < confidence_raw:
        breakdown["_capped_to"] = cap
        breakdown["_cap_reason"] = cap_code
        return round(cap, 4), breakdown

    return confidence_raw, breakdown


def invoice_quality_score(
    inv: InvoiceData | None,
    validation_findings=None,
) -> float:
    """Confidence score in [0, 1]. The number you display / route on."""
    score, _ = invoice_quality_breakdown(inv, validation_findings)
    return score


def invoice_quality_coverage(
    inv: InvoiceData | None,
) -> float:
    """Coverage score in [0, 1] — how many signals are present. No discount, no cap."""
    if inv is None:
        return 0.0
    _, bd = invoice_quality_breakdown(inv, validation_findings=None)
    return bd.get("_coverage", 0.0)


def format_quality_breakdown(breakdown: dict) -> str:
    """Render breakdown for display."""
    if not breakdown:
        return ""
    parts = [
        f"{k}={v:.2f}"
        for k, v in breakdown.items()
        if not k.startswith("_") and isinstance(v, (int, float))
    ]
    text = ",".join(parts)
    cov = breakdown.get("_coverage")
    conf = breakdown.get("_confidence")
    if cov is not None and conf is not None:
        text += f" | cov={cov:.2f} conf={conf:.2f}"
    if "_capped_to" in breakdown:
        text += f" | capped→{breakdown['_capped_to']:.2f}({breakdown.get('_cap_reason','')})"
    return text
