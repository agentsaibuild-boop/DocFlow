"""Lightweight regex-based field extractor for fallback extractors that don't produce InvoiceData.

Conservative by design: only extracts fields that are unambiguous from raw text
(IBAN with checksum, labeled invoice number). Does NOT populate supplier/customer
fields — distinguishing 'Изпълнител' vs 'Получател' from text alone is unreliable
across templates and OCR quality. Use Gemini path for trustworthy party data.
"""

import re

from docflow.schema import InvoiceData

INVOICE_NUMBER_RE = re.compile(
    r"(?:фактура|инвойс|invoice)[\s№#:]*([A-Z0-9][A-Z0-9/\-]{2,20})",
    re.IGNORECASE,
)
IBAN_RE = re.compile(r"\b(BG\d{2}[A-Z]{4}[A-Z0-9]{14})\b")


def parse_text_to_invoice(text: str) -> InvoiceData | None:
    """Returns minimal InvoiceData from raw text. Only sets safely-derivable fields."""
    if not text or not text.strip():
        return None

    inv = InvoiceData()
    found_anything = False

    m = INVOICE_NUMBER_RE.search(text)
    if m:
        inv.invoice_number = m.group(1).strip()
        found_anything = True

    m = IBAN_RE.search(text)
    if m:
        inv.iban = m.group(1)
        found_anything = True

    return inv if found_anything else None
