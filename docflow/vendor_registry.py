"""Structured vendor registry for deterministic supplier classification.

Replaces the inline `VENDOR_PROFILES` dict that mixed substring keys with
profile metadata. Each vendor is a dataclass with explicit `name_patterns`
(substrings to match against supplier.name, case-insensitive). Looking up by
name returns the full VendorProfile or None.

To add a vendor, append to VENDORS with a fresh `key`, the canonical
`display` name, the patterns you want to match, and the origin/payment hints.
Do not match too broadly — short substrings like "google" will catch
"Google EOOD" too, which is BG. That's an accepted trade-off; downstream
status logic still uses VAT/EIK proof first and falls back to the vendor
registry only when neither is present.
"""

from dataclasses import dataclass
from enum import Enum


class VendorOrigin(str, Enum):
    """Mirrors docflow.status.SupplierOrigin without importing it (avoid cycle).
    The status module maps these onto its own enum at lookup time."""
    BG = "bg"
    FOREIGN = "foreign"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VendorProfile:
    key: str                       # stable identifier, also the lookup tag
    display: str                   # human-readable canonical name
    name_patterns: tuple[str, ...] # case-insensitive substrings matched against supplier.name
    origin: VendorOrigin
    expected_payment: str          # "card" | "card_or_invoice" | "subscription" | "bank_transfer" | ...
    notes: str = ""


# Order matters: first matching profile wins. Put more specific patterns first.
VENDORS: tuple[VendorProfile, ...] = (
    VendorProfile(
        key="github",
        display="GitHub",
        name_patterns=("github",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card_or_invoice",
    ),
    VendorProfile(
        key="canva",
        display="Canva",
        name_patterns=("canva",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="subscription",
    ),
    VendorProfile(
        key="openai",
        display="OpenAI",
        name_patterns=("openai",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card",
    ),
    VendorProfile(
        key="anthropic",
        display="Anthropic",
        name_patterns=("anthropic",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card",
    ),
    VendorProfile(
        key="google",
        display="Google",
        name_patterns=("google",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card_or_invoice",
        notes="May collide with BG-incorporated Google EOOD; status logic checks BG VAT/EIK first.",
    ),
    VendorProfile(
        key="microsoft",
        display="Microsoft",
        name_patterns=("microsoft",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card_or_invoice",
    ),
    VendorProfile(
        key="stripe",
        display="Stripe",
        name_patterns=("stripe",),
        origin=VendorOrigin.FOREIGN,
        expected_payment="card",
    ),
)


def lookup_vendor(supplier_name: str | None) -> VendorProfile | None:
    """Return the first VENDORS entry whose patterns match, or None."""
    if not supplier_name:
        return None
    name = supplier_name.lower()
    for v in VENDORS:
        if any(pattern in name for pattern in v.name_patterns):
            return v
    return None
