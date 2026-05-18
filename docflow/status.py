"""Unified status engine.

A single function decides the per-row status used by both the Streamlit UI and
the batch xlsx writer. Status is supplier- and payment-method-aware: a missing
IBAN on a US card-paid invoice is not the same kind of "incomplete" as on a
Bulgarian bank-transfer invoice. Diagnostic columns (quality_score,
validation_errors, registry_status, is_derived) are not part of the
completeness contract — selecting them must not affect status.
"""

from dataclasses import dataclass, field
from enum import Enum

from docflow.columns import get_row_value

# Columns whose values are computed from pipeline metadata rather than
# extracted from the document. They are display-only and must never gate
# the status decision.
DIAGNOSTIC_KEYS = {
    "quality_score",
    "validation_errors",
    "registry_status",
    "is_derived",
}


@dataclass
class StatusResult:
    code: str
    label: str
    empty_columns: list[str] = field(default_factory=list)
    notes: str = ""


class SupplierOrigin(str, Enum):
    """Ternary supplier provenance.

    Binary foreign/BG was unsafe: a BG supplier billing in USD or a foreign
    invoice without an explicit country code could be misclassified. UNKNOWN
    is the honest answer when there is no proof either way; downstream rules
    must treat UNKNOWN conservatively (don't punish the row).
    """
    BG = "bg"
    FOREIGN = "foreign"
    UNKNOWN = "unknown"


def _vat_prefix(inv) -> str:
    if inv.supplier is None or not inv.supplier.vat_number:
        return ""
    return inv.supplier.vat_number.strip().upper()[:2]


def supplier_origin(inv) -> SupplierOrigin:
    """Classify the supplier as BG / FOREIGN / UNKNOWN with proof, not hints.

    Proof:
      • VAT prefix is a two-letter alpha country code: BG → BG, anything else → FOREIGN.
      • No VAT, but BG EIK present (9/10/13 BG digits) → BG.
      • Otherwise → UNKNOWN. (Currency alone is not proof — a BG supplier may bill in EUR/USD.)
    """
    if inv is None or inv.supplier is None:
        return SupplierOrigin.UNKNOWN
    prefix = _vat_prefix(inv)
    if prefix == "BG":
        return SupplierOrigin.BG
    if len(prefix) == 2 and prefix.isalpha():
        return SupplierOrigin.FOREIGN
    if inv.supplier.eik:
        return SupplierOrigin.BG
    return SupplierOrigin.UNKNOWN


_CARD_HINTS = (
    "карт", "card", "кредитна", "credit", "debit",
    "subscription", "абонамент", "stripe", "paypal",
    "в брой", "cash",
)
_BANK_HINTS = (
    "банк", "bank", "по сметка", "transfer", "wire",
    "по платеж", "платежно нареж", "по нар",
)


def requires_iban(inv) -> bool:
    """Conservative IBAN-requirement rule.

    IBAN is treated as required ONLY when there is clear evidence the invoice
    is bank-paid — an explicit bank-transfer payment method. Everything else
    (card, cash, subscription, no payment method stated, unknown supplier
    origin) → IBAN not required. This avoids penalising cash/card invoices
    that didn't explicitly state their method.
    """
    if inv is None:
        return False
    pm = (inv.payment_method or "").lower()
    if any(sig in pm for sig in _CARD_HINTS):
        return False
    if any(sig in pm for sig in _BANK_HINTS):
        return True
    return False


# Conservative list of clearly-foreign SaaS providers. Used only when the
# supplier's origin can't be proven from VAT/EIK and the name itself is the
# strongest evidence we have. Match is case-insensitive substring on
# supplier.name. Never used to fabricate VAT numbers — only to decide that a
# missing BG VAT is not a problem.
_FOREIGN_SAAS_PROVIDERS = (
    "github", "canva", "openai", "anthropic", "google",
    "microsoft", "stripe",
)


def is_known_foreign_saas(inv) -> bool:
    if inv is None or inv.supplier is None or not inv.supplier.name:
        return False
    name = inv.supplier.name.lower()
    return any(p in name for p in _FOREIGN_SAAS_PROVIDERS)


def _vat_required(inv) -> bool:
    """Is a BG VAT number meaningfully required for this invoice?

    Foreign suppliers obviously don't have BG VAT. Unknown-origin invoices
    from clearly-foreign SaaS providers (matched by name) also don't.
    Otherwise → required.
    """
    origin = supplier_origin(inv)
    if origin == SupplierOrigin.FOREIGN:
        return False
    if origin == SupplierOrigin.UNKNOWN and is_known_foreign_saas(inv):
        return False
    return True


def is_column_required_for(key: str, inv) -> bool:
    """Whether the column applies to this invoice's profile.

    Diagnostic keys are never status-relevant. Domain exemptions live here.
    """
    if key in DIAGNOSTIC_KEYS:
        return False
    if key == "supplier_iban":
        return requires_iban(inv)
    if key == "supplier_vat":
        return _vat_required(inv)
    return True


def compute_status(
    doc,
    validation_findings,
    required_column_keys: list[str] | None = None,
    extraction_error: str | None = None,
    provider_error: str | None = None,
    provider_error_kind: str | None = None,
) -> StatusResult:
    """Decide a row status, with a WHY explanation in notes.

    Precedence (first match wins):
      1. provider_error → PROVIDER_ERROR (separate from document quality)
      2. extraction_error → ERROR (non-provider exception)
      3. invoice missing / supplier.name empty → NO_DATA
      4. any validation finding with level=='error' → VALIDATION_ERROR
         (math mismatches surface here and always win over completeness)
      5. required_column_keys provided and some empty, after
         supplier/payment-method exemptions → INCOMPLETE
      6. otherwise → OK
    """
    if provider_error:
        kind = provider_error_kind or "other"
        kind_label = {
            "quota": "квота/503",
            "auth": "автентикация",
            "network": "мрежа/timeout",
            "other": "грешка",
        }.get(kind, kind)
        return StatusResult(
            code="PROVIDER_ERROR",
            label=f"🔌 Provider ({kind_label})",
            notes=f"Provider не върна резултат ({kind}). Документът не е тестван: {provider_error[:120]}",
        )

    if extraction_error:
        return StatusResult(
            code="ERROR",
            label="❌ ERROR",
            notes=f"Извличането се провали: {extraction_error[:120]}",
        )

    inv = doc.invoice if doc is not None else None
    if inv is None or not inv.supplier or not inv.supplier.name:
        return StatusResult(
            code="NO_DATA",
            label="⚠️ Без данни",
            notes="Не е разпознат доставчик в документа",
        )

    errors = [f for f in (validation_findings or []) if getattr(f, "level", None) == "error"]
    if errors:
        codes = ", ".join(sorted({f.code for f in errors}))
        return StatusResult(
            code="VALIDATION_ERROR",
            label="⚠️ Има грешки",
            notes=f"Validation грешки: {codes}",
        )

    if required_column_keys:
        empty: list[str] = []
        for key in required_column_keys:
            if not is_column_required_for(key, inv):
                continue
            value = get_row_value(key, doc)
            if value is None or value == "":
                empty.append(key)
        if empty:
            return StatusResult(
                code="INCOMPLETE",
                label=f"⚠️ Непълни ({len(empty)})",
                empty_columns=empty,
                notes="Празни задължителни полета: " + ", ".join(empty),
            )

    origin = supplier_origin(inv)
    origin_label = {
        SupplierOrigin.BG: "БГ",
        SupplierOrigin.FOREIGN: "чуждестранен",
        SupplierOrigin.UNKNOWN: "произход неясен",
    }[origin]
    iban_note = "IBAN изискван" if requires_iban(inv) else "IBAN неприложим"
    return StatusResult(
        code="OK",
        label="✅ OK",
        notes=f"Всички задължителни полета попълнени ({origin_label} доставчик, {iban_note})",
    )
