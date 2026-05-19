"""Shared column catalog + row-value resolver.

Single source of truth for the columns exposed in the UI export and used by
status-engine completeness checks. Both the Streamlit app and the batch path
import from here so display labels and the "what is empty?" notion stay
consistent.
"""

# (Колона за Excel, Етикет, Категория, По подразбиране)
COLUMN_CATALOG = [
    ("supplier_name",  "Доставчик",          "Основни",  True),
    ("supplier_vat",   "Номер по ДДС",       "Основни",  True),
    ("supplier_iban",  "IBAN доставчик",     "Основни",  True),
    ("invoice_number", "Номер фактура",      "Основни",  True),
    ("invoice_date",   "Дата фактура",       "Основни",  True),
    ("net_amount",     "Цена без ДДС",       "Основни",  True),
    ("vat_total",      "ДДС",                "Основни",  True),
    ("total_to_pay",   "Крайна сума",        "Основни",  True),

    ("supplier_eik",     "ЕИК доставчик",      "Доставчик",  False),
    ("supplier_address", "Адрес доставчик",    "Доставчик",  False),
    ("supplier_mol",     "МОЛ доставчик",      "Доставчик",  False),
    ("supplier_phone",   "Телефон доставчик",  "Доставчик",  False),
    ("supplier_email",   "Email доставчик",    "Доставчик",  False),
    ("supplier_bank",    "Банка доставчик",    "Доставчик",  False),
    ("supplier_bic",     "BIC доставчик",      "Доставчик",  False),

    ("customer_name",    "Получател",          "Получател",  False),
    ("customer_eik",     "ЕИК получател",      "Получател",  False),
    ("customer_vat",     "Номер по ДДС получ.","Получател",  False),
    ("customer_address", "Адрес получател",    "Получател",  False),
    ("customer_mol",     "МОЛ получател",      "Получател",  False),

    ("delivery_date",    "Дата доставка",      "Дати",       False),
    ("due_date",         "Срок плащане",       "Дати",       False),

    ("currency",         "Валута",             "Плащане",    True),
    ("subtotal",         "Сума без отстъпка",  "Плащане",    False),
    ("discount",         "Отстъпка",           "Плащане",    False),
    ("payment_method",   "Метод плащане",      "Плащане",    False),
    ("paid",             "Платени",            "Плащане",    False),
    ("remaining",        "Остава",             "Плащане",    False),

    ("quality_score",     "Quality score",      "Диагностика", True),
    ("quality_breakdown", "Quality breakdown",  "Диагностика", False),
    ("validation_errors", "Validation errors",  "Диагностика", True),
    ("registry_status",   "Registry status",    "Диагностика", True),
    ("derived_fields",    "Производни полета",  "Диагностика", True),
]


def get_row_value(field, doc, *, validation_findings=None, registry_findings=None):
    """Map a column key to the actual value from ExtractedDocument.

    Diagnostic columns (quality_score / validation_errors / registry_status /
    is_derived) need additional context, passed as keyword arguments. When
    those are omitted, diagnostic keys fall back to a sensible empty value.
    """
    if field == "quality_score":
        return round(doc.quality_score, 2) if doc is not None else ""
    if field == "quality_breakdown":
        if doc is None or doc.invoice is None:
            return ""
        from docflow.quality import format_quality_breakdown, invoice_quality_breakdown
        _, breakdown = invoice_quality_breakdown(doc.invoice, validation_findings)
        return format_quality_breakdown(breakdown)
    if field == "validation_errors":
        if not validation_findings:
            return ""
        codes = sorted({f.code for f in validation_findings if getattr(f, "level", None) == "error"})
        return ", ".join(codes)
    if field == "registry_status":
        if not registry_findings:
            return "—"
        codes = [getattr(f, "code", "") for f in registry_findings]
        if any(c == "registry_skipped" for c in codes):
            return "не потвърден"
        if any(c.startswith("supplier_") for c in codes):
            return "записан"
        return "—"
    if field == "derived_fields":
        if doc is None or doc.invoice is None:
            return ""
        return ", ".join(doc.invoice.derived_fields)

    if doc is None or doc.invoice is None:
        return ""
    inv = doc.invoice
    s = inv.supplier
    c = inv.customer
    if inv.vat_breakdown:
        vat_total = sum(v.vat_amount for v in inv.vat_breakdown)
    elif inv.total_to_pay is not None and inv.net_amount is not None:
        vat_total = round(inv.total_to_pay - inv.net_amount, 2)
    else:
        vat_total = None
    return {
        "supplier_name":    s.name if s else "",
        "supplier_eik":     s.eik if s else "",
        "supplier_vat":     s.vat_number if s else "",
        "supplier_address": s.address if s else "",
        "supplier_mol":     s.mol if s else "",
        "supplier_phone":   s.phone if s else "",
        "supplier_email":   s.email if s else "",
        "supplier_iban":    inv.iban,
        "supplier_bank":    inv.bank,
        "supplier_bic":     inv.bic,
        "customer_name":    c.name if c else "",
        "customer_eik":     c.eik if c else "",
        "customer_vat":     c.vat_number if c else "",
        "customer_address": c.address if c else "",
        "customer_mol":     c.mol if c else "",
        "invoice_number":   inv.invoice_number,
        "invoice_date":     inv.issue_date,
        "delivery_date":    inv.delivery_date,
        "due_date":         inv.payment_due_date,
        "currency":         inv.currency or "EUR",
        "subtotal":         inv.subtotal,
        "discount":         inv.discount,
        "net_amount":       inv.net_amount,
        "vat_total":        vat_total,
        "total_to_pay":     inv.total_to_pay,
        "payment_method":   inv.payment_method,
        "paid":             inv.paid,
        "remaining":        inv.remaining,
    }.get(field, "")
