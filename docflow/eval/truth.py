"""Golden-dataset truth schema.

Each invoice in the evaluation dataset is paired with a sidecar truth file:

    invoice_001.pdf
    invoice_001.truth.json

The truth file is the human-verified ground-truth for a subset of InvoiceData
fields. It is intentionally NARROWER than InvoiceData — fields here are
exactly the ones we want to evaluate extraction accuracy against. Fields
omitted from the truth file are ignored during scoring (treated as "not
graded"), not as null.

Truth schema is a Pydantic model so the JSON is validated on load and the
evaluator can rely on types.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr

# Suffix appended to the invoice filename to find its truth sidecar.
TRUTH_SUFFIX = ".truth.json"


class TruthLineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_without_vat: Optional[float] = None


class InvoiceTruth(BaseModel):
    """Human-verified facts about one invoice.

    All fields optional — only graded if present in the truth JSON. Critical
    fields (EIK, VAT, IBAN, total_to_pay, net_amount) are usually present;
    secondary fields (payment_method, line_items) often partial.
    """
    supplier_name:   Optional[str]   = None
    supplier_eik:    Optional[str]   = None
    supplier_vat:    Optional[str]   = None
    iban:            Optional[str]   = None
    invoice_number:  Optional[str]   = None
    issue_date:      Optional[str]   = Field(default=None, description="YYYY-MM-DD")
    net_amount:      Optional[float] = None
    vat_amount:      Optional[float] = None
    total_to_pay:    Optional[float] = None
    currency:        Optional[str]   = None
    payment_method:  Optional[str]   = None
    line_items:      Optional[list[TruthLineItem]] = None
    notes:           Optional[str]   = Field(default=None, description="Free-text human note; not graded")

    # Which keys were actually present in the source JSON. Lets the evaluator
    # distinguish "field omitted by the human" (not_in_truth, not graded) from
    # "field explicitly null" (must equal null → hallucination if extractor
    # produced a value).
    _present_keys: set[str] = PrivateAttr(default_factory=set)

    def has_field(self, name: str) -> bool:
        """True if the source data explicitly carried this field.

        Priority:
          1. _present_keys set by load_truth from the source JSON dict
             (the authoritative answer for files on disk).
          2. Otherwise model_fields_set — Pydantic's record of which fields
             were explicitly passed at construction (vs. left to default).
             Lets programmatic construction work intuitively in tests.
        """
        if self._present_keys:
            return name in self._present_keys
        return name in self.model_fields_set


# Fields the truth schema covers. Order is the canonical display order.
TRUTH_FIELDS: tuple[str, ...] = (
    "supplier_name", "supplier_eik", "supplier_vat", "iban",
    "invoice_number", "issue_date",
    "net_amount", "vat_amount", "total_to_pay",
    "currency", "payment_method", "line_items",
)

# Critical fields — reported separately in aggregate metrics.
CRITICAL_FIELDS: tuple[str, ...] = (
    "supplier_eik", "supplier_vat", "iban", "total_to_pay", "net_amount",
)


def truth_path_for(invoice_path: Path) -> Path:
    """Given invoice_001.pdf return invoice_001.truth.json (sibling)."""
    return invoice_path.with_suffix("").with_name(invoice_path.stem + TRUTH_SUFFIX)


def load_truth(truth_path: Path) -> InvoiceTruth:
    """Read and validate a truth JSON. Raises ValidationError on bad shape.

    Tracks which top-level keys were actually present in the source JSON so
    downstream evaluation can tell 'omitted' from 'explicitly null'.
    """
    data = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    inst = InvoiceTruth.model_validate(data)
    inst._present_keys = set(data.keys()) if isinstance(data, dict) else set()
    return inst


def discover_dataset(folder: Path) -> list[tuple[Path, Path]]:
    """Return [(invoice_path, truth_path), ...] for every invoice in `folder`
    that has a matching .truth.json sibling. Invoices without truth are
    silently excluded — only the graded subset is returned.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    supported = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in supported:
            continue
        tp = truth_path_for(f)
        if tp.exists():
            pairs.append((f, tp))
    return pairs
