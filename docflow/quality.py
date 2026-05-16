"""Invoice quality scoring.

Pipeline uses this to decide whether the current extractor produced something
good enough to stop early, or whether to try the next provider. The score is
intentionally coarse — the goal is "is this useful enough to ship?" not
fine-grained ranking.
"""

from docflow.schema import InvoiceData


_WEIGHTS = {
    "supplier_name":     0.15,
    "supplier_tax_id":   0.10,  # ЕИК or VAT
    "invoice_number":    0.10,
    "issue_date":        0.10,
    "any_total":         0.15,  # total_to_pay OR net_amount
    "line_items":        0.10,
    "customer_name":     0.10,
    "vat_info":          0.10,  # vat_breakdown OR derivable from lines
    "iban":              0.05,
    "currency":          0.05,
}
# Weights sum to 1.0.


def _present(value) -> bool:
    return value is not None and value != ""


def invoice_quality_score(inv: InvoiceData | None) -> float:
    """Return a 0..1 quality score for an invoice. None → 0.0."""
    if inv is None:
        return 0.0

    score = 0.0

    if inv.supplier and _present(inv.supplier.name):
        score += _WEIGHTS["supplier_name"]

    if inv.supplier and (_present(inv.supplier.eik) or _present(inv.supplier.vat_number)):
        score += _WEIGHTS["supplier_tax_id"]

    if _present(inv.invoice_number):
        score += _WEIGHTS["invoice_number"]

    if _present(inv.issue_date):
        score += _WEIGHTS["issue_date"]

    if _present(inv.total_to_pay) or _present(inv.net_amount):
        score += _WEIGHTS["any_total"]

    if inv.line_items:
        score += _WEIGHTS["line_items"]

    if inv.customer and _present(inv.customer.name):
        score += _WEIGHTS["customer_name"]

    has_vat_breakdown = bool(inv.vat_breakdown)
    derivable = bool(
        inv.line_items
        and any(li.vat_percent is not None and li.total_without_vat is not None
                for li in inv.line_items)
    )
    if has_vat_breakdown or derivable:
        score += _WEIGHTS["vat_info"]

    if _present(inv.iban):
        score += _WEIGHTS["iban"]

    # Currency: schema default is None. We award credit only when an extractor
    # explicitly populated it (EUR/€/евро/BGN/USD/...). The display layer shows
    # "EUR" as fallback for None — that's a presentation choice, not a signal
    # that the document actually carried a currency.
    if _present(inv.currency):
        score += _WEIGHTS["currency"]

    return round(score, 4)
