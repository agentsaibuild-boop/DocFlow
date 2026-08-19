"""Line-item (артикули) export helpers shared by the UI and batch Excel writers."""

from docflow.columns import COLUMN_CATALOG, get_row_value
from docflow.schema import InvoiceData

# Fields shown in the single-invoice review pane (original | extracted).
REVIEW_FIELD_KEYS = [
    "supplier_name",
    "supplier_eik",
    "supplier_vat",
    "supplier_iban",
    "customer_name",
    "invoice_number",
    "invoice_date",
    "net_amount",
    "vat_total",
    "total_to_pay",
    "currency",
    "line_item_count",
]

LINE_ITEM_HEADERS = [
    "Файл", "Доставчик", "Номер фактура",
    "№", "Описание", "К-во", "ME",
    "Ед. цена", "Отстъпка %", "Цена с отст.",
    "ДДС %", "ДДС", "Общо без ДДС",
]

VAT_HEADERS = ["Файл", "Доставчик", "Номер фактура", "Ставка %", "Основа", "ДДС"]


def _invoice_of(result) -> InvoiceData | None:
    if getattr(result, "error", None) or getattr(result, "provider_error", None):
        return None
    doc = getattr(result, "doc", None)
    if doc is None:
        return None
    return doc.invoice


def _source_name(result) -> str:
    source = getattr(result, "source", None)
    if source is None:
        return ""
    return getattr(source, "name", str(source))


def line_item_count(inv: InvoiceData | None) -> int:
    if inv is None:
        return 0
    return len(inv.line_items or [])


def line_item_summary(inv: InvoiceData | None, *, max_items: int = 6) -> str:
    """Compact description list for the header table. Empty → em dash (not blank),
    so a missing article list does not mark the invoice row INCOMPLETE."""
    if inv is None:
        return "—"
    items = [li for li in (inv.line_items or []) if li.description]
    if not items:
        return "—"
    names = [li.description.strip() for li in items if li.description]
    extra = len(names) - max_items
    shown = names[:max_items]
    text = "; ".join(shown)
    if extra > 0:
        text += f" (+{extra})"
    return text


def iter_line_item_rows(results) -> list[list]:
    rows: list[list] = []
    for result in results:
        inv = _invoice_of(result)
        if inv is None:
            continue
        supplier = inv.supplier.name if inv.supplier else None
        for li in inv.line_items or []:
            rows.append([
                _source_name(result),
                supplier,
                inv.invoice_number,
                li.number,
                li.description,
                li.quantity,
                li.unit,
                li.unit_price,
                li.discount_percent,
                li.price_after_discount,
                li.vat_percent,
                li.vat_amount,
                li.total_without_vat,
            ])
    return rows


def invoice_review_pairs(doc, *, validation_findings=None, registry_findings=None) -> list[tuple[str, object]]:
    """Label/value pairs for the extracted-fields pane."""
    label_for = {k: lbl for k, lbl, _, _ in COLUMN_CATALOG}
    pairs: list[tuple[str, object]] = []
    for key in REVIEW_FIELD_KEYS:
        pairs.append((
            label_for.get(key, key),
            get_row_value(
                key, doc,
                validation_findings=validation_findings,
                registry_findings=registry_findings,
            ),
        ))
    return pairs


def detail_line_item_rows(result) -> tuple[list[str], list[list]]:
    """Headers and rows for one invoice — drop the batch Файл column."""
    headers = LINE_ITEM_HEADERS[3:]
    rows = [row[3:] for row in iter_line_item_rows([result])]
    return headers, rows


def detail_vat_rows(result) -> tuple[list[str], list[list]]:
    headers = VAT_HEADERS[3:]
    rows = [row[3:] for row in iter_vat_rows([result])]
    return headers, rows


def iter_vat_rows(results) -> list[list]:
    rows: list[list] = []
    for result in results:
        inv = _invoice_of(result)
        if inv is None:
            continue
        supplier = inv.supplier.name if inv.supplier else None
        for vat in inv.vat_breakdown or []:
            rows.append([
                _source_name(result),
                supplier,
                inv.invoice_number,
                vat.rate_percent,
                vat.base_amount,
                vat.vat_amount,
            ])
    return rows
