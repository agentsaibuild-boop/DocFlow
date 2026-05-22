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
    "quality_breakdown",
    "validation_errors",
    "registry_status",
    "derived_fields",
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
      • No VAT/EIK, but supplier name matches a known foreign SaaS vendor
        (VENDOR_PROFILES) → FOREIGN, deterministically.
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
    profile = vendor_profile(inv)
    if profile is not None:
        return {
            VendorOrigin.BG: SupplierOrigin.BG,
            VendorOrigin.FOREIGN: SupplierOrigin.FOREIGN,
            VendorOrigin.UNKNOWN: SupplierOrigin.UNKNOWN,
        }[profile.origin]
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


from docflow.vendor_registry import VENDORS, VendorOrigin, VendorProfile, lookup_vendor

# Re-export for callers that imported VENDOR_PROFILES before the refactor.
VENDOR_PROFILES = {v.key: v for v in VENDORS}


def vendor_profile(inv) -> VendorProfile | None:
    """Return the matched VendorProfile, or None. Wraps lookup_vendor with the
    invoice signature the rest of the module expects."""
    if inv is None or inv.supplier is None:
        return None
    return lookup_vendor(inv.supplier.name)


def is_known_foreign_saas(inv) -> bool:
    """Backward-compat boolean for callers that don't need the full profile."""
    p = vendor_profile(inv)
    return p is not None and p.origin == VendorOrigin.FOREIGN


def _vat_required(inv) -> bool:
    """BG VAT is meaningful only when the supplier is definitively Bulgarian.

    FOREIGN and UNKNOWN origins → not required. supplier_origin() already
    routes known foreign SaaS vendors to FOREIGN, so the check collapses to
    a single comparison.
    """
    return supplier_origin(inv) == SupplierOrigin.BG


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
        # User-facing labels: plain BG, actionable, no raw error codes in the badge.
        label_by_kind = {
            "quota":   "⏳ Моделът е претоварен",
            "auth":    "🔑 Грешен или липсващ API ключ",
            "network": "🌐 Проблем с връзката",
            "other":   "⚠️ Моделът не отговори",
        }
        notes_by_kind = {
            "quota":   "Услугата временно е претоварена. Опитай отново след минута или избери друг модел.",
            "auth":    "Провери API ключа в .env (или Streamlit secrets). Документът не е обработен.",
            "network": "Връзката с услугата прекъсна или таймаутна. Опитай отново.",
            "other":   "Моделът върна неочакван отговор. Опитай отново или избери друг модел.",
        }
        return StatusResult(
            code="PROVIDER_ERROR",
            label=label_by_kind.get(kind, "⚠️ Моделът не отговори"),
            notes=notes_by_kind.get(kind, f"Документът не е обработен: {provider_error[:120]}"),
        )

    if extraction_error:
        # extraction_error already arrives sanitised to an exception class name
        # (see app._process_single). We don't surface that to the user — only
        # the calm sentence below.
        return StatusResult(
            code="ERROR",
            label="❌ Грешка",
            notes="Файлът не беше обработен. Опитай отново или избери друг режим.",
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
    # IBAN note: never claim "неприложим" when the IBAN is actually present.
    # That would contradict the row's own data.
    if inv.iban:
        iban_note = "IBAN присъства"
    elif requires_iban(inv):
        iban_note = "IBAN изискван"  # status branch would normally INCOMPLETE; safety net.
    else:
        iban_note = "IBAN неприложим"
    return StatusResult(
        code="OK",
        label="✅ OK",
        notes=f"Всички задължителни полета попълнени ({origin_label} доставчик, {iban_note})",
    )
