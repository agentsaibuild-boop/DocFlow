import re
from dataclasses import dataclass

from docflow.schema import ExtractedDocument, InvoiceData

EIK_LABELED_RE = re.compile(
    r"(?:ЕИК|ПИК|BG)[\s:/№#.\-]*(\d{13}|\d{10}|\d{9})\b",
    re.IGNORECASE,
)
IBAN_BG_PREFIX_RE = re.compile(r"\bBG\d{2}[A-Z]{4}[A-Z0-9]{4,22}\b")
VALID_EIK_LENGTHS = (9, 10, 13)
MATH_TOLERANCE = 0.05


@dataclass
class ValidationFinding:
    level: str  # "ok" | "warning" | "error"
    code: str
    message: str
    value: str | None = None


def _iban_checksum_ok(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def _check_eik(eik: str, role: str) -> ValidationFinding:
    cleaned = eik.strip()
    if cleaned.upper().startswith("BG"):
        cleaned = cleaned[2:].strip()
    if cleaned.isdigit() and len(cleaned) in VALID_EIK_LENGTHS:
        return ValidationFinding("ok", "eik_format", f"{role} ЕИК format valid", cleaned)
    return ValidationFinding("warning", "eik_format", f"{role} ЕИК wrong format", eik)


def validate_eiks_in_text(text: str) -> list[ValidationFinding]:
    """Fallback: only labeled ЕИК-like numbers (e.g. 'ЕИК: 207260294' or 'BG207260294')."""
    findings = []
    seen: set[str] = set()
    for match in EIK_LABELED_RE.finditer(text):
        eik = match.group(1)
        if eik in seen:
            continue
        seen.add(eik)
        if len(eik) in VALID_EIK_LENGTHS:
            findings.append(ValidationFinding("ok", "eik_format", "ЕИК format valid", eik))
    return findings


def validate_iban(iban: str) -> ValidationFinding:
    iban = iban.replace(" ", "").upper()
    if not iban.startswith("BG"):
        return ValidationFinding("warning", "iban_prefix", "IBAN does not start with BG", iban)
    if len(iban) != 22:
        return ValidationFinding(
            "error", "iban_length",
            f"IBAN is {len(iban)} chars (expected 22) — likely OCR/extraction error",
            iban,
        )
    if not _iban_checksum_ok(iban):
        return ValidationFinding(
            "error", "iban_checksum",
            "IBAN checksum INVALID — likely OCR/extraction error",
            iban,
        )
    return ValidationFinding("ok", "iban_checksum", "IBAN checksum valid", iban)


def validate_ibans_in_text(text: str) -> list[ValidationFinding]:
    findings = []
    seen: set[str] = set()
    for match in IBAN_BG_PREFIX_RE.finditer(text.replace(" ", "")):
        iban = match.group(0)
        if iban in seen:
            continue
        seen.add(iban)
        findings.append(validate_iban(iban))
    return findings


def _approx_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= MATH_TOLERANCE


def validate_invoice_math(inv: InvoiceData) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    if inv.line_items:
        line_sum = sum(li.total_without_vat or 0 for li in inv.line_items)
        if inv.net_amount is not None:
            if _approx_equal(line_sum, inv.net_amount):
                findings.append(ValidationFinding(
                    "ok", "math_lines_eq_net",
                    f"Σ редове = обща нетна сума ({line_sum:.2f})",
                ))
            else:
                findings.append(ValidationFinding(
                    "error", "math_lines_eq_net",
                    f"Σ редове ({line_sum:.2f}) ≠ обща нетна ({inv.net_amount:.2f}), разлика {line_sum - inv.net_amount:+.2f}",
                ))

    if inv.subtotal is not None and inv.discount is not None and inv.net_amount is not None:
        expected = inv.subtotal - abs(inv.discount)
        if _approx_equal(expected, inv.net_amount):
            findings.append(ValidationFinding(
                "ok", "math_subtotal_minus_discount",
                f"Сума без отстъпка − отстъпка = обща нетна ({expected:.2f})",
            ))
        else:
            findings.append(ValidationFinding(
                "error", "math_subtotal_minus_discount",
                f"Сума − отстъпка ({expected:.2f}) ≠ нетна ({inv.net_amount:.2f})",
            ))

    if inv.net_amount is not None and inv.vat_breakdown and inv.total_to_pay is not None:
        vat_total = sum(v.vat_amount for v in inv.vat_breakdown)
        expected = inv.net_amount + vat_total
        if _approx_equal(expected, inv.total_to_pay):
            findings.append(ValidationFinding(
                "ok", "math_net_plus_vat",
                f"Нетна + ДДС = за плащане ({expected:.2f})",
            ))
        else:
            findings.append(ValidationFinding(
                "error", "math_net_plus_vat",
                f"Нетна + ДДС ({expected:.2f}) ≠ за плащане ({inv.total_to_pay:.2f})",
            ))

    if inv.total_to_pay is not None and inv.paid is not None and inv.remaining is not None:
        expected = inv.total_to_pay - inv.paid
        if _approx_equal(expected, inv.remaining):
            findings.append(ValidationFinding(
                "ok", "math_remaining",
                f"За плащане − платени = остава ({expected:.2f})",
            ))
        else:
            findings.append(ValidationFinding(
                "error", "math_remaining",
                f"За плащане − платени ({expected:.2f}) ≠ остава ({inv.remaining:.2f})",
            ))

    return findings


def _validate_bank_payment_iban(inv: InvoiceData) -> list[ValidationFinding]:
    """Document-level fact: if the invoice explicitly says bank payment but no
    IBAN was extracted, that's an error regardless of column selection.

    This intentionally fires only on explicit bank payment hints — implicit/
    unknown payment methods do not punish missing IBAN.
    """
    from docflow.status import requires_iban  # local import to avoid cycle

    if not requires_iban(inv):
        return []
    if inv.iban:
        return []
    return [ValidationFinding(
        level="error",
        code="iban_missing_for_bank_payment",
        message=f"Платежният метод '{inv.payment_method}' изисква IBAN, но няма извлечен",
    )]


def validate(doc: ExtractedDocument) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    text = doc.full_text + "\n" + "\n".join(
        " ".join(cell for cell in row) for table in doc.tables for row in table.rows
    )

    if doc.invoice:
        if doc.invoice.supplier and doc.invoice.supplier.eik:
            findings.append(_check_eik(doc.invoice.supplier.eik, "Изпълнител"))
        if doc.invoice.customer and doc.invoice.customer.eik:
            findings.append(_check_eik(doc.invoice.customer.eik, "Получател"))
        if doc.invoice.iban:
            findings.append(validate_iban(doc.invoice.iban))
        else:
            findings.extend(validate_ibans_in_text(text))
        findings.extend(validate_invoice_math(doc.invoice))
        findings.extend(_validate_bank_payment_iban(doc.invoice))
    else:
        findings.extend(validate_eiks_in_text(text))
        findings.extend(validate_ibans_in_text(text))

    return findings
