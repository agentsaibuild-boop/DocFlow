"""End-to-end test suite for DocFlow.

Run from project root:
    .venv/bin/python -m tests.test_full_system

Each test reports pass/fail/skip. Final summary tallies results.
Tests that require Gemini are skipped if API key not set.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docflow.env import load_env_file
load_env_file(PROJECT_ROOT / ".env")


PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []  # (name, status, detail)


def run(name: str, fn: Callable[[], str | None]) -> None:
    try:
        detail = fn()
        results.append((name, PASS, detail or ""))
        print(f"  ✓ {name}")
        if detail:
            print(f"      {detail}")
    except AssertionError as e:
        results.append((name, FAIL, str(e)))
        print(f"  ✗ {name}: {e}")
    except SkipTest as e:
        results.append((name, SKIP, str(e)))
        print(f"  ⊘ {name}: skipped — {e}")
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


class SkipTest(Exception):
    pass


# ─── helpers ────────────────────────────────────────────────────────────────

def clean_registry() -> None:
    db = PROJECT_ROOT / "registry.db"
    if db.exists():
        db.unlink()


def has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


SAMPLE_JPG = PROJECT_ROOT / "samples" / "eurofaktura_sample.jpg"
SAMPLE_BORN_DIGITAL_PDF = PROJECT_ROOT / "samples" / "test_invoice.pdf"
SAMPLE_SCANNED_PDF = PROJECT_ROOT / "samples" / "eurofaktura_sample_scanned.pdf"


# ─── Tests ──────────────────────────────────────────────────────────────────

# 1. Schema sanity
def test_schema_imports():
    from docflow.schema import (
        ExtractedDocument, ExtractedTable, InvoiceData, LineItem,
        Party, VatBreakdown, is_useful_invoice,
    )
    assert is_useful_invoice(InvoiceData(invoice_number="X"))
    assert not is_useful_invoice(InvoiceData())
    assert not is_useful_invoice(None)


def test_pipeline_registers_extractors():
    from docflow.pipeline import AVAILABLE_PROVIDERS, EXTRACTORS, PROVIDER_ALIASES
    names = [e.name for e in EXTRACTORS]
    # Active provider list is intentionally short and focused for the public
    # evaluation deployment — only the two benchmarked options.
    assert set(PROVIDER_ALIASES) == {
        "gemini-3.1-flash-lite", "qwen-3-vl-235b",
    }, PROVIDER_ALIASES
    # Removed providers must not leak back via EXTRACTORS or aliases.
    for removed in ("pdfplumber", "claude", "azure", "gemini_2.5_pro", "mistral_ocr"):
        assert removed not in names, f"removed extractor present in EXTRACTORS: {removed}"
        assert removed not in PROVIDER_ALIASES, f"removed alias still registered: {removed}"
    # 'auto' is hidden from the UI surface.
    assert "auto" not in AVAILABLE_PROVIDERS, AVAILABLE_PROVIDERS
    return f"order: {names}"


# 2. Validators
def test_eik_validator_accepts_9_10_13_digits():
    from docflow.validators import _check_eik
    for eik in ["123456789", "1234567890", "1234567890123"]:
        f = _check_eik(eik, "test")
        assert f.level == "ok", f"{eik}: {f.message}"
    f = _check_eik("12345", "test")
    assert f.level == "warning"
    return "9, 10, 13 digits accepted; 5 digits rejected"


def test_eik_validator_strips_BG_prefix():
    from docflow.validators import _check_eik
    f = _check_eik("BG123456789", "test")
    assert f.level == "ok"
    assert f.value == "123456789"


def test_iban_validator():
    from docflow.validators import validate_iban
    assert validate_iban("BG80BNBG96611020345678").level == "ok"
    too_long = validate_iban("BG80BNBGBNG96611020345678")
    assert too_long.level == "error" and "25 chars" in too_long.message
    bad_checksum = validate_iban("BG99BNBG96611020345678")
    assert bad_checksum.level == "error" and "checksum" in bad_checksum.message.lower()


def test_invoice_math_validator():
    from docflow.schema import InvoiceData, LineItem, VatBreakdown
    from docflow.validators import validate_invoice_math

    inv = InvoiceData(
        line_items=[LineItem(total_without_vat=100), LineItem(total_without_vat=200)],
        net_amount=300,
        subtotal=350, discount=50,
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=300, vat_amount=60)],
        total_to_pay=360,
        paid=100, remaining=260,
    )
    findings = validate_invoice_math(inv)
    errors = [f for f in findings if f.level == "error"]
    assert not errors, f"unexpected errors: {[f.message for f in errors]}"
    return f"4 math checks all OK"


def test_invoice_math_catches_wrong_remaining():
    from docflow.schema import InvoiceData
    from docflow.validators import validate_invoice_math
    inv = InvoiceData(total_to_pay=1000, paid=100, remaining=999)  # wrong: should be 900
    errors = [f for f in validate_invoice_math(inv) if f.level == "error"]
    assert len(errors) == 1 and errors[0].code == "math_remaining"


def test_validate_eiks_in_text_only_labeled():
    from docflow.validators import validate_eiks_in_text
    text = "Фактура № 0000000321 от ЕИК 207260294, ДДС BG9601270035"
    findings = validate_eiks_in_text(text)
    values = sorted(f.value for f in findings)
    assert "0000000321" not in values
    assert "207260294" in values
    assert "9601270035" in values
    return f"only labeled ЕИК-и: {values}"


# 3. Text parser
def test_text_parser_extracts_iban_and_invoice_number():
    from docflow.text_parser import parse_text_to_invoice
    text = "Фактура № 2026/0001\nIBAN: BG80BNBG96611020345678"
    inv = parse_text_to_invoice(text)
    assert inv is not None
    assert inv.invoice_number == "2026/0001"
    assert inv.iban == "BG80BNBG96611020345678"


def test_text_parser_does_not_set_party_fields():
    from docflow.text_parser import parse_text_to_invoice
    text = "Изпълнител: ООД Тест\nЕИК 123456789"
    inv = parse_text_to_invoice(text)
    if inv is not None:
        assert inv.supplier.name is None
        assert inv.supplier.eik is None


def test_text_parser_returns_none_for_empty_text():
    from docflow.text_parser import parse_text_to_invoice
    assert parse_text_to_invoice("") is None
    assert parse_text_to_invoice(None) is None  # type: ignore[arg-type]
    assert parse_text_to_invoice("just plain text without any matches") is None


# 4. Registry behavior
def test_registry_new_supplier():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    inv = InvoiceData(supplier=Party(name="ООД Тест", vat_number="BG123456789"))
    findings = r.record(inv, human_confirmed=True)
    assert len(findings) == 1 and findings[0].code == "supplier_new"

    suppliers = r.all()
    assert len(suppliers) == 1
    assert suppliers[0]["eik"] == "123456789"  # derived from VAT


def test_registry_does_not_mutate_input():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    inv = InvoiceData(supplier=Party(name="X", vat_number="BG123456789"))
    eik_before = inv.supplier.eik
    r.record(inv, human_confirmed=True)
    assert inv.supplier.eik == eik_before, f"record mutated input.eik to {inv.supplier.eik}"

    inv2 = InvoiceData(supplier=Party(name="X", vat_number="BG123456789"))
    eik_before2 = inv2.supplier.eik
    enriched, _ = r.enrich(inv2)
    assert inv2.supplier.eik == eik_before2, f"enrich mutated input.eik to {inv2.supplier.eik}"
    assert enriched is not inv2


def test_registry_auto_fills_missing_fields():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    r.record(InvoiceData(supplier=Party(name="X", vat_number="BG123456789")), human_confirmed=True)
    r.record(InvoiceData(
        supplier=Party(name="X", vat_number="BG123456789", address="ул. Тест 1"),
        iban="BG80BNBG96611020345678",
    ), human_confirmed=True)
    s = r.all()[0]
    assert s["address"] == "ул. Тест 1"
    assert s["iban"] == "BG80BNBG96611020345678"
    return "address+iban auto-filled on 2nd invoice"


def test_registry_enrich_overrides_mismatch():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    r.correct_or_insert = None  # noop placeholder
    # record initial good state
    r.record(InvoiceData(
        supplier=Party(name="X", vat_number="BG123456789"),
        iban="BG80BNBG96611020345678",
    ), human_confirmed=True)
    # simulate hallucinated IBAN coming from extractor
    bad_inv = InvoiceData(
        supplier=Party(name="X", vat_number="BG123456789"),
        iban="BG80BNBGBNG96611020345678",  # 25 chars — Gemini hallucination
    )
    enriched, findings = r.enrich(bad_inv)
    overrides = [f for f in findings if f.code == "registry_override"]
    assert overrides, f"expected registry_override, got: {[f.code for f in findings]}"
    assert enriched.iban == "BG80BNBG96611020345678"
    return f"hallucinated IBAN replaced from registry"


# 4a. Provenance metadata (PrivateAttr — must not leak into LLM schemas)
def test_derived_fields_not_in_json_schema():
    """Regression guard: provenance metadata is internal-only. If it leaks
    into InvoiceData.model_json_schema(), Gemini/Claude/Mistral break with
    schema validation errors (uniqueItems extra_forbidden)."""
    from docflow.schema import InvoiceData
    schema = InvoiceData.model_json_schema()
    props = schema.get("properties", {})
    assert "derived_fields" not in props, (
        f"derived_fields leaked into JSON schema. Properties: {sorted(props)}"
    )
    assert "_derived_fields" not in props
    return f"schema properties: {len(props)} fields, no provenance leak"


def test_derived_fields_provenance_still_records():
    """Pipeline normalization must still record what it filled, after the
    PrivateAttr refactor."""
    from docflow.pipeline import _derive_missing_totals
    from docflow.schema import InvoiceData, LineItem, VatBreakdown

    inv = InvoiceData(
        line_items=[LineItem(number=1, description="X", total_without_vat=2500)],
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=2500, vat_amount=500)],
    )
    assert inv.net_amount is None and inv.total_to_pay is None
    _derive_missing_totals(inv)

    assert inv.net_amount == 2500
    assert inv.total_to_pay == 3000
    assert inv.is_derived("net_amount"), "net_amount should be marked derived"
    assert inv.is_derived("total_to_pay"), "total_to_pay should be marked derived"
    assert inv.is_derived("line_items[0].vat_amount"), "line vat should be marked derived"

    paths = inv.derived_fields  # public property
    assert isinstance(paths, tuple) and paths == tuple(sorted(paths))
    return f"derived: {paths}"


# 4ab. Supplier-aware status rules
def _make_doc(inv):
    from docflow.schema import ExtractedDocument
    return ExtractedDocument(
        source_path="x.pdf", extraction_method="test", page_count=1, invoice=inv,
    )


def test_foreign_supplier_no_iban_not_incomplete():
    """GitHub/Canva/etc. — no BG identifiers, USD currency, no IBAN. Status
    must NOT be INCOMPLETE because of the missing IBAN."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import (
        SupplierOrigin, compute_status, requires_iban, supplier_origin,
    )

    inv = InvoiceData(
        invoice_number="INV-001",
        issue_date="2026-01-01",
        supplier=Party(name="GitHub, Inc."),
        customer=Party(name="ПЕТЪР КОЗАРЕВ ЕООД", eik="201730367"),
        net_amount=10.0,
        total_to_pay=10.0,
        currency="USD",
    )
    # Name matches VENDOR_PROFILES → deterministic FOREIGN (not UNKNOWN).
    assert supplier_origin(inv) == SupplierOrigin.FOREIGN
    assert not requires_iban(inv), "foreign + no payment_method → IBAN not required"

    required = [
        "supplier_name", "supplier_iban", "invoice_number", "invoice_date",
        "net_amount", "total_to_pay", "currency",
    ]
    result = compute_status(_make_doc(inv), [], required_column_keys=required)
    assert result.code == "OK", f"expected OK, got {result.code}: {result.notes}"
    assert "supplier_iban" not in result.empty_columns

    # Now with explicit foreign VAT prefix — same conclusion via a different path.
    inv_foreign = inv.model_copy(update={"supplier": Party(name="Canva Pty Ltd", vat_number="GB123456789")})
    assert supplier_origin(inv_foreign) == SupplierOrigin.FOREIGN
    result_foreign = compute_status(_make_doc(inv_foreign), [], required_column_keys=required)
    assert result_foreign.code == "OK"
    return result.notes


def test_bg_supplier_no_iban_incomplete_with_bank_payment():
    """Bulgarian supplier paying by bank transfer must carry an IBAN; status
    is INCOMPLETE if IBAN is missing. Without an explicit bank payment
    method, the conservative rule does NOT require IBAN — cash/card invoices
    that omit payment_method must not be punished."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import (
        SupplierOrigin, compute_status, requires_iban, supplier_origin,
    )

    inv = InvoiceData(
        invoice_number="100",
        issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="X"),
        net_amount=100.0,
        total_to_pay=120.0,
        payment_method="ПЛАЩАНЕ ПО БАНКА",
        currency="BGN",
    )
    assert supplier_origin(inv) == SupplierOrigin.BG
    assert requires_iban(inv), "BG supplier + explicit bank payment → IBAN required"

    required = ["supplier_name", "supplier_iban", "invoice_number", "invoice_date", "net_amount", "total_to_pay"]
    result = compute_status(_make_doc(inv), [], required_column_keys=required)
    assert result.code == "INCOMPLETE", f"expected INCOMPLETE, got {result.code}"
    assert "supplier_iban" in result.empty_columns, result.empty_columns

    # Same supplier without explicit payment_method — under the conservative
    # rule, IBAN is NOT required (cash/card-paid BG invoices are common).
    inv_default = inv.model_copy(update={"payment_method": None})
    assert not requires_iban(inv_default), "no payment_method → no IBAN requirement"
    result2 = compute_status(_make_doc(inv_default), [], required_column_keys=required)
    assert result2.code == "OK", f"expected OK (no PM stated), got {result2.code}"

    # Card-paid — IBAN exempt, status OK.
    inv_card = inv.model_copy(update={"payment_method": "Кредитна карта"})
    assert not requires_iban(inv_card)
    result3 = compute_status(_make_doc(inv_card), [], required_column_keys=required)
    assert result3.code == "OK"

    return f"bank→INCOMPLETE, default→OK, card→OK"


def test_math_mismatch_always_produces_error():
    """A math validation error must surface as VALIDATION_ERROR regardless
    of completeness or other signals."""
    from docflow.schema import InvoiceData, LineItem, Party, VatBreakdown
    from docflow.status import compute_status
    from docflow.validators import validate_invoice_math

    inv = InvoiceData(
        invoice_number="X",
        issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Y"),
        iban="BG80BNBG96611020345678",
        line_items=[LineItem(total_without_vat=100)],
        net_amount=100.0,
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        total_to_pay=200.0,  # WRONG: should be 120
        currency="BGN",
        payment_method="По банка",
    )
    findings = validate_invoice_math(inv)
    error_findings = [f for f in findings if f.level == "error"]
    assert error_findings, "math validator should flag total_to_pay mismatch"

    # Even though completeness is satisfied (IBAN present, BG bank payment),
    # math error must dominate.
    required = ["supplier_name", "supplier_iban", "invoice_number", "invoice_date", "net_amount", "total_to_pay"]
    result = compute_status(_make_doc(inv), findings, required_column_keys=required)
    assert result.code == "VALIDATION_ERROR", f"got {result.code}"
    assert "math" in result.notes.lower(), result.notes
    return f"math error → {result.code}: {result.notes}"


def test_supplier_vat_exempt_for_foreign_and_known_saas():
    """BG VAT must not gate status for foreign / known-foreign-SaaS suppliers.
    GitHub Inc. (no VAT at all, no IBAN, USD) → OK.
    Canva (non-BG VAT, no IBAN) → OK.
    BG supplier with missing VAT and the column selected → INCOMPLETE."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import (
        SupplierOrigin, compute_status, is_known_foreign_saas, supplier_origin,
    )

    required = [
        "supplier_name", "supplier_vat", "supplier_iban",
        "invoice_number", "invoice_date", "net_amount", "total_to_pay",
    ]

    # 1. GitHub: deterministic FOREIGN via VENDOR_PROFILES.
    inv_gh = InvoiceData(
        invoice_number="GH-1", issue_date="2026-05-01",
        supplier=Party(name="GitHub, Inc."),
        customer=Party(name="Купувач"),
        net_amount=10.0, total_to_pay=10.0, currency="USD",
    )
    assert supplier_origin(inv_gh) == SupplierOrigin.FOREIGN
    assert is_known_foreign_saas(inv_gh)
    r1 = compute_status(_make_doc(inv_gh), [], required_column_keys=required)
    assert r1.code == "OK", f"GitHub expected OK, got {r1.code}: {r1.notes} (empty={r1.empty_columns})"

    # 2. Canva: explicit non-BG VAT → FOREIGN.
    inv_canva = InvoiceData(
        invoice_number="CV-1", issue_date="2026-05-01",
        supplier=Party(name="Canva Pty Ltd", vat_number="IE3722896KH"),
        customer=Party(name="Купувач"),
        net_amount=12.99, total_to_pay=12.99, currency="EUR",
    )
    assert supplier_origin(inv_canva) == SupplierOrigin.FOREIGN
    r2 = compute_status(_make_doc(inv_canva), [], required_column_keys=required)
    assert r2.code == "OK", f"Canva expected OK, got {r2.code}: {r2.notes} (empty={r2.empty_columns})"

    # 3. BG supplier, missing VAT, VAT column selected → INCOMPLETE on supplier_vat.
    inv_bg = InvoiceData(
        invoice_number="BG-1", issue_date="2026-05-01",
        supplier=Party(name="БГ ООД", eik="123456789"),  # no vat_number
        customer=Party(name="X"),
        iban="BG80BNBG96611020345678",
        net_amount=100.0, total_to_pay=120.0, currency="BGN",
    )
    assert supplier_origin(inv_bg) == SupplierOrigin.BG
    r3 = compute_status(_make_doc(inv_bg), [], required_column_keys=required)
    assert r3.code == "INCOMPLETE", f"BG without VAT expected INCOMPLETE, got {r3.code}"
    assert "supplier_vat" in r3.empty_columns
    return f"GitHub→OK, Canva→OK, BG-no-VAT→INCOMPLETE"


def test_known_saas_does_not_fabricate_vat():
    """The known-SaaS exemption must affect column-required logic only —
    it must never write a value into supplier.vat_number."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import compute_status, is_known_foreign_saas

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-05-01",
        supplier=Party(name="OpenAI, LLC"),
        customer=Party(name="Y"),
        net_amount=20.0, total_to_pay=20.0, currency="USD",
    )
    assert is_known_foreign_saas(inv)
    assert inv.supplier.vat_number is None, "precondition"
    compute_status(_make_doc(inv), [], required_column_keys=["supplier_vat"])
    assert inv.supplier.vat_number is None, "status must not invent VAT"
    return "vat_number stays None after status evaluation"


# 4d. Provider error semantics
class _FakeExtractor:
    """Minimal extractor stub for testing provider-error pathways."""
    def __init__(self, name, *, can=True, raises=None, doc_factory=None):
        self.name = name
        self._can = can
        self._raises = raises
        self._doc_factory = doc_factory

    def can_handle(self, path):
        return self._can

    def extract(self, path):
        if self._raises is not None:
            raise self._raises
        return self._doc_factory()


def _trivial_doc(method="fake", score=None):
    """Build a minimal extracted document. Score=None lets the pipeline compute it."""
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    inv = InvoiceData(
        invoice_number="F-1", issue_date="2026-05-01",
        supplier=Party(name="Foo ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Купувач"),
        net_amount=100.0, total_to_pay=120.0, currency="BGN",
    )
    doc = ExtractedDocument(
        source_path="x.pdf", extraction_method=method, page_count=1, invoice=inv,
    )
    if score is not None:
        doc.quality_score = score
    return doc


def test_explicit_provider_does_not_fallback_by_default(tmp_path=None):
    """Explicit provider + quota 503 → ProviderError, not silent fallback."""
    import docflow.pipeline as pipeline
    from docflow.pipeline import ProviderError

    quota_exc = RuntimeError("503 Service Unavailable: quota exceeded")
    original = pipeline.EXTRACTORS
    original_aliases = pipeline.PROVIDER_ALIASES
    try:
        pipeline.EXTRACTORS = [
            _FakeExtractor("primary", raises=quota_exc),
            _FakeExtractor("backup", doc_factory=lambda: _trivial_doc("backup")),
        ]
        pipeline.PROVIDER_ALIASES = {"primary": "primary", "backup": "backup"}
        from pathlib import Path as _P
        raised = None
        try:
            pipeline.extract(_P("/tmp/whatever.pdf"), provider="primary")
        except ProviderError as e:
            raised = e
        assert raised is not None, "explicit provider must raise ProviderError, not return"
        assert raised.provider == "primary"
        assert raised.kind == "quota", f"expected 'quota', got {raised.kind!r}"
    finally:
        pipeline.EXTRACTORS = original
        pipeline.PROVIDER_ALIASES = original_aliases
    return "explicit primary 503 → ProviderError(quota), no fallback"


def test_explicit_provider_with_allow_fallback_uses_backup():
    """Explicit provider + allow_fallback=True + quota → fallback to next extractor."""
    import docflow.pipeline as pipeline

    quota_exc = RuntimeError("429 too many requests")
    original = pipeline.EXTRACTORS
    original_aliases = pipeline.PROVIDER_ALIASES
    try:
        pipeline.EXTRACTORS = [
            _FakeExtractor("primary", raises=quota_exc),
            _FakeExtractor("backup", doc_factory=lambda: _trivial_doc("backup", score=0.9)),
        ]
        pipeline.PROVIDER_ALIASES = {"primary": "primary", "backup": "backup"}
        from pathlib import Path as _P
        doc = pipeline.extract(_P("/tmp/whatever.pdf"), provider="primary", allow_fallback=True)
        assert doc.extraction_method == "backup"
    finally:
        pipeline.EXTRACTORS = original
        pipeline.PROVIDER_ALIASES = original_aliases
    return "allow_fallback=True → backup ran"


def test_auto_provider_falls_back_on_quota():
    """provider='auto' falls back automatically on quota."""
    import docflow.pipeline as pipeline

    original = pipeline.EXTRACTORS
    try:
        pipeline.EXTRACTORS = [
            _FakeExtractor("first", raises=RuntimeError("503 server overloaded")),
            _FakeExtractor("second", doc_factory=lambda: _trivial_doc("second", score=0.9)),
        ]
        from pathlib import Path as _P
        doc = pipeline.extract(_P("/tmp/whatever.pdf"), provider="auto")
        assert doc.extraction_method == "second"
    finally:
        pipeline.EXTRACTORS = original
    return "auto fell back to second after 503"


def test_provider_error_does_not_poison_quality_metrics():
    """If a sub-threshold result was already obtained, a later provider quota
    error must NOT raise — pipeline returns best_doc with score preserved."""
    import docflow.pipeline as pipeline
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    original = pipeline.EXTRACTORS
    try:
        # Sparse invoice → score below the 0.7 threshold so the pipeline keeps
        # trying the next extractor. Second 503s. Expected: return the sparse
        # doc, score recomputed (~0.15), no exception.
        def sparse_doc():
            # Has invoice_number → is_useful_invoice; ~0.25 score → below 0.7 threshold.
            inv = InvoiceData(invoice_number="S-1", supplier=Party(name="Sparse ООД"))
            return ExtractedDocument(
                source_path="x.pdf", extraction_method="first", page_count=1, invoice=inv,
            )

        pipeline.EXTRACTORS = [
            _FakeExtractor("first", doc_factory=sparse_doc),
            _FakeExtractor("second", raises=RuntimeError("503 quota exceeded")),
        ]
        from pathlib import Path as _P
        doc = pipeline.extract(_P("/tmp/whatever.pdf"), provider="auto")
        assert doc.extraction_method == "first", f"got {doc.extraction_method}"
        # The score must reflect the sparse doc (name only ≈ 0.15), not be
        # zeroed/None by the second extractor's quota event.
        assert 0.0 < doc.quality_score < 0.7, f"unexpected score: {doc.quality_score}"
    finally:
        pipeline.EXTRACTORS = original
    return "best_doc preserved from first extractor despite second's quota"


def test_provider_error_classification():
    """Error message → kind detection."""
    from docflow.pipeline import _classify_provider_error
    cases = [
        (RuntimeError("HTTP 429 Too Many Requests"), "quota"),
        (RuntimeError("503 server unavailable"), "quota"),
        (RuntimeError("RESOURCE_EXHAUSTED"), "quota"),
        (RuntimeError("401 Unauthorized"), "auth"),
        (RuntimeError("Invalid API key"), "auth"),
        (RuntimeError("Connection timed out"), "network"),
        (RuntimeError("DNS resolution failed"), "network"),
        (RuntimeError("Something else went wrong"), "other"),
    ]
    for exc, expected in cases:
        perr = _classify_provider_error("test", exc)
        assert perr.kind == expected, f"{exc!r} → expected {expected}, got {perr.kind}"
    return f"{len(cases)} classifications correct"


def test_summarize_batch_counts():
    """summarize_batch keeps provider failures separate from document failures."""
    from docflow.batch import BatchResult, summarize_batch
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    from docflow.validators import ValidationFinding
    from pathlib import Path as _P

    ok_inv = InvoiceData(
        invoice_number="OK-1", issue_date="2026-01-01",
        supplier=Party(name="ОК ООД", eik="123456789"),
        net_amount=100.0, total_to_pay=120.0,
    )
    ok_doc = ExtractedDocument(source_path="ok.pdf", extraction_method="m", page_count=1, invoice=ok_inv)

    bad_inv = ok_inv.model_copy(update={"invoice_number": "BAD-1"})
    bad_doc = ExtractedDocument(source_path="bad.pdf", extraction_method="m", page_count=1, invoice=bad_inv)
    err_finding = ValidationFinding(level="error", code="math_x", message="x")

    no_data_doc = ExtractedDocument(source_path="empty.pdf", extraction_method="m", page_count=1, invoice=None)

    results = [
        BatchResult(_P("ok.pdf"), None, ok_doc, None, [], [], []),
        BatchResult(_P("bad.pdf"), None, bad_doc, None, [], [err_finding], []),
        BatchResult(_P("empty.pdf"), None, no_data_doc, None, [], [], []),
        BatchResult(_P("q1.pdf"), None, None, None, [], [], [],
                    provider_error="[gemini/quota] 503", provider_error_kind="quota"),
        BatchResult(_P("q2.pdf"), None, None, None, [], [], [],
                    provider_error="[gemini/quota] 503", provider_error_kind="quota"),
        BatchResult(_P("crash.pdf"), None, None, "ValueError: bad data", [], [], []),
    ]
    s = summarize_batch(results)
    counts = {k: s[k] for k in ("processed", "extracted_ok", "validation_errors",
                                 "no_data", "provider_failures", "unknown_errors")}
    assert counts == {
        "processed": 6, "extracted_ok": 1, "validation_errors": 1,
        "no_data": 1, "provider_failures": 2, "unknown_errors": 1,
    }, counts
    return str(counts)


def test_quality_score_capped_by_math_error():
    """A math validation error must cap quality_score at 0.5, even if every
    field is populated."""
    from docflow.quality import invoice_quality_breakdown, invoice_quality_score
    from docflow.schema import InvoiceData, LineItem, Party, VatBreakdown
    from docflow.validators import validate_invoice_math

    # Fully populated invoice → would score ~1.0 without validation.
    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Y"),
        iban="BG80BNBG96611020345678",
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        net_amount=100.0,
        total_to_pay=999.0,  # WRONG: should be 120
        currency="BGN",
    )

    score_no_v = invoice_quality_score(inv, validation_findings=None)
    assert score_no_v >= 0.95, f"unvalidated score should be high, got {score_no_v}"

    findings = validate_invoice_math(inv)
    score_with_v = invoice_quality_score(inv, findings)
    assert score_with_v <= 0.5, f"math-error score must cap at 0.5, got {score_with_v}"

    _, breakdown = invoice_quality_breakdown(inv, findings)
    assert breakdown.get("_capped_to") == 0.5
    assert breakdown.get("_cap_reason", "").startswith("math_")
    return f"raw={score_no_v}, capped={score_with_v}, reason={breakdown['_cap_reason']}"


def test_quality_score_capped_by_iban_checksum():
    """Invalid IBAN checksum must cap quality_score at 0.6."""
    from docflow.quality import invoice_quality_score
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    from docflow.validators import validate

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Y"),
        iban="BG99BNBG96611020345678",  # bad checksum
        line_items=[],
        net_amount=100.0, total_to_pay=100.0,
        currency="BGN",
    )
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    findings = validate(doc)
    assert any(f.code == "iban_checksum" and f.level == "error" for f in findings), \
        [f.code for f in findings]
    score = invoice_quality_score(inv, findings)
    assert score <= 0.6, f"bad-IBAN score must cap at 0.6, got {score}"
    return f"capped to {score}"


def test_quality_breakdown_lists_signals():
    """Breakdown should list which signals contributed."""
    from docflow.quality import invoice_quality_breakdown, format_quality_breakdown
    from docflow.schema import InvoiceData, Party

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789"),
        currency="BGN",
    )
    score, bd = invoice_quality_breakdown(inv)
    assert "supplier_name" in bd
    assert "supplier_tax_id" in bd
    assert "invoice_number" in bd
    assert "issue_date" in bd
    assert "currency" in bd
    assert "_coverage" in bd and "_confidence" in bd
    rendered = format_quality_breakdown(bd)
    for token in ("supplier_name=0.15", "issue_date=0.10", "currency=0.05"):
        assert token in rendered, f"missing {token!r} in: {rendered}"
    return f"score={score}, signals={len(bd)-1}, rendered={rendered}"


def test_vendor_profiles_deterministic_foreign():
    """All listed vendors must be classified as FOREIGN deterministically,
    regardless of currency or other heuristics."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import SupplierOrigin, VENDOR_PROFILES, supplier_origin, vendor_profile

    expected = {"github", "canva", "openai", "anthropic", "google", "microsoft", "stripe"}
    assert expected.issubset(VENDOR_PROFILES.keys()), \
        f"missing vendors: {expected - set(VENDOR_PROFILES)}"

    test_cases = [
        ("GitHub, Inc.", "github"),
        ("Canva Pty Ltd", "canva"),
        ("OpenAI, LLC", "openai"),
        ("Anthropic PBC", "anthropic"),
        ("Google LLC", "google"),
        ("Microsoft Ireland", "microsoft"),
        ("Stripe Payments Europe", "stripe"),
    ]
    for name, expected_key in test_cases:
        inv = InvoiceData(supplier=Party(name=name))
        profile = vendor_profile(inv)
        assert profile is not None, f"{name}: no profile"
        assert profile.key == expected_key
        assert supplier_origin(inv) == SupplierOrigin.FOREIGN, f"{name}: origin not FOREIGN"
    return f"{len(test_cases)} vendors → deterministic FOREIGN"


def test_ok_note_does_not_lie_about_present_iban():
    """When the IBAN is in the data, the OK note must NOT say
    'IBAN неприложим' — even if the supplier is BG and payment method is card."""
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    from docflow.status import compute_status

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Y"),
        iban="BG80BNBG96611020345678",  # IBAN IS present
        net_amount=100.0, total_to_pay=120.0,
        currency="BGN",
        payment_method="Кредитна карта",  # card → requires_iban False
    )
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    result = compute_status(doc, [])
    assert result.code == "OK"
    assert "IBAN неприложим" not in result.notes, \
        f"OK note contradicted the data: {result.notes!r}"
    assert "IBAN присъства" in result.notes, result.notes
    return result.notes


def test_derived_fields_column_returns_list_not_boolean():
    """The diagnostic column 'derived_fields' must list paths, not just да/не."""
    from docflow.columns import get_row_value
    from docflow.pipeline import _derive_missing_totals
    from docflow.schema import ExtractedDocument, InvoiceData, LineItem, VatBreakdown

    inv = InvoiceData(
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
    )
    _derive_missing_totals(inv)
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    value = get_row_value("derived_fields", doc)
    assert value not in ("да", "не"), f"got boolean string: {value!r}"
    # Should contain dotted paths.
    assert "net_amount" in value, f"missing net_amount in: {value!r}"
    return f"derived: {value}"


def test_derived_signals_get_discounted():
    """A signal whose substrate is in derived_fields must contribute less than
    the same signal grounded in extraction. Coverage stays full; confidence
    drops."""
    from docflow.pipeline import _derive_missing_totals
    from docflow.quality import (
        _DERIVATION_DISCOUNT, invoice_quality_breakdown, invoice_quality_coverage,
    )
    from docflow.schema import InvoiceData, LineItem, Party, VatBreakdown

    # Path 1: line_items + net_amount + total_to_pay + vat_breakdown all EXTRACTED.
    inv_extracted = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="X", eik="123456789"),
        customer=Party(name="Y"),
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        net_amount=100.0,
        total_to_pay=120.0,
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        currency="BGN",
    )
    score_ex, bd_ex = invoice_quality_breakdown(inv_extracted)
    cov_ex = invoice_quality_coverage(inv_extracted)

    # Path 2: same signal set but totals + vat_breakdown DERIVED from line_items.
    inv_derived = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="X", eik="123456789"),
        customer=Party(name="Y"),
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        currency="BGN",
    )
    _derive_missing_totals(inv_derived)
    # After derivation: net_amount, total_to_pay, vat_breakdown all in derived_fields.
    assert "net_amount" in inv_derived.derived_fields
    assert "total_to_pay" in inv_derived.derived_fields
    assert "vat_info" not in inv_derived.derived_fields  # vat_breakdown stored under that key
    assert "vat_breakdown" in inv_derived.derived_fields

    score_d, bd_d = invoice_quality_breakdown(inv_derived)
    cov_d = invoice_quality_coverage(inv_derived)

    assert score_d < score_ex, f"derived score {score_d} should be < extracted {score_ex}"
    assert cov_d == cov_ex, f"coverage should be equal: extracted={cov_ex}, derived={cov_d}"
    assert "any_total(derived)" in bd_d
    assert "vat_info(derived)" in bd_d
    assert bd_d["any_total(derived)"] == round(0.15 * _DERIVATION_DISCOUNT, 4)
    return f"extracted_conf={score_ex}, derived_conf={score_d}, coverage_both={cov_ex}"


def test_coverage_and_confidence_are_different_concepts():
    """When all signals are derived from a single line item, coverage should
    still be high (we know everything) but confidence should be lower."""
    from docflow.pipeline import _derive_missing_totals
    from docflow.quality import invoice_quality_breakdown
    from docflow.schema import InvoiceData, LineItem, Party

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="X", eik="123456789"),
        customer=Party(name="Y"),
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        currency="BGN",
    )
    _derive_missing_totals(inv)
    score, bd = invoice_quality_breakdown(inv)
    coverage = bd["_coverage"]
    confidence = bd["_confidence"]
    assert coverage > confidence, f"coverage={coverage} must exceed confidence={confidence}"
    return f"coverage={coverage}, confidence={confidence}"


def test_vendor_registry_is_structured_not_inline():
    """vendor_registry exposes a real dataclass with explicit name_patterns,
    not a substring-keyed dict."""
    from docflow.vendor_registry import VENDORS, VendorOrigin, VendorProfile, lookup_vendor

    assert isinstance(VENDORS, tuple)
    for v in VENDORS:
        assert isinstance(v, VendorProfile)
        assert isinstance(v.name_patterns, tuple) and len(v.name_patterns) >= 1
        assert v.origin in (VendorOrigin.FOREIGN, VendorOrigin.BG, VendorOrigin.UNKNOWN)

    # Lookup works case-insensitively and substring-matched.
    assert lookup_vendor("GitHub, Inc.").key == "github"
    assert lookup_vendor("microsoft ireland operations").key == "microsoft"
    assert lookup_vendor("Райкомерс ЕООД") is None
    assert lookup_vendor("") is None
    assert lookup_vendor(None) is None
    return f"{len(VENDORS)} structured vendor profiles"


def test_summarize_batch_per_provider_stats():
    """summarize_batch returns per-provider averages including latency."""
    from docflow.batch import BatchResult, summarize_batch
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    from pathlib import Path as _P

    def _result(name, method, score, coverage, lat):
        inv = InvoiceData(
            invoice_number=name, issue_date="2026-01-01",
            supplier=Party(name="X", eik="123456789"),
        )
        doc = ExtractedDocument(
            source_path=name, extraction_method=method, page_count=1, invoice=inv,
            quality_score=score, quality_coverage=coverage,
        )
        return BatchResult(_P(name), None, doc, None, [], [], [], processing_time_s=lat)

    results = [
        _result("f1.pdf", "gemini:gemini-3.1-flash-lite", 0.90, 0.95, 7.0),
        _result("f2.pdf", "gemini:gemini-3.1-flash-lite", 0.80, 0.90, 8.0),
        _result("f3.pdf", "qwen_3_vl_235b:qwen3-vl-235b-a22b-instruct", 1.00, 1.00, 25.0),
        BatchResult(_P("q.pdf"), None, None, None, [], [], [],
                    provider_error="[gemini/quota] x", provider_error_kind="quota",
                    processing_time_s=2.0),
    ]
    s = summarize_batch(results)
    assert s["processed"] == 4
    assert s["provider_failures"] == 1
    assert s["extracted_ok"] == 3
    assert "per_provider" in s
    gp = s["per_provider"]["gemini"]
    assert gp["files"] == 2
    assert gp["avg_score"] == 0.85
    assert gp["avg_latency_s"] == 7.5
    qp = s["per_provider"]["qwen_3_vl_235b"]
    assert qp["files"] == 1 and qp["avg_score"] == 1.0 and qp["avg_latency_s"] == 25.0
    return f"providers in summary: {sorted(s['per_provider'])}"


# 4e. Evaluation framework (truth-based accuracy measurement)
def test_eval_truth_schema_and_discovery():
    from docflow.eval.truth import (
        CRITICAL_FIELDS, InvoiceTruth, TRUTH_FIELDS, discover_dataset,
        truth_path_for,
    )

    truth = InvoiceTruth(
        supplier_name="X", supplier_eik="123456789",
        iban="BG80BNBG96611020345678", total_to_pay=120.0,
    )
    assert truth.supplier_name == "X"
    assert "supplier_eik" in TRUTH_FIELDS
    assert "supplier_eik" in CRITICAL_FIELDS

    # truth_path_for must produce a sibling .truth.json
    p = Path("/x/invoice_001.pdf")
    assert truth_path_for(p).name == "invoice_001.truth.json"

    # discover_dataset returns only files with matching sidecars
    import tempfile as _t
    with _t.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.pdf").write_bytes(b"")
        (td / "a.truth.json").write_text('{"supplier_name": "X"}')
        (td / "b.pdf").write_bytes(b"")  # no truth
        pairs = discover_dataset(td)
        names = [p.name for p, _ in pairs]
        assert names == ["a.pdf"], names
    return f"{len(TRUTH_FIELDS)} truth fields, {len(CRITICAL_FIELDS)} critical"


def test_eval_compare_invoice_matches_and_mismatches():
    """compare_invoice produces correct per-field statuses."""
    from docflow.eval.metrics import compare_invoice
    from docflow.eval.truth import InvoiceTruth
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    truth = InvoiceTruth(
        supplier_name="ООД Тест",
        supplier_eik="123456789",
        iban="BG80BNBG96611020345678",
        invoice_number="2026/0001",
        net_amount=100.0,
        total_to_pay=120.0,
    )
    inv = InvoiceData(
        invoice_number="2026/0001",                 # exact match
        supplier=Party(name="ООД  Тест", eik="123456789"),  # normalized (extra space)
        iban="BG80BNBG96611020345678",              # exact match
        net_amount=100.0,                            # exact match
        total_to_pay=99.99,                          # mismatch (>tolerance)
        # supplier_vat missing in truth → not_in_truth
    )
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    comp = compare_invoice(doc, truth, invoice_file="x.pdf", provider="test")

    by_field = {f.field: f for f in comp.fields}
    assert by_field["invoice_number"].status == "exact_match"
    assert by_field["supplier_name"].status == "normalized_match"
    assert by_field["supplier_eik"].status == "exact_match"
    assert by_field["iban"].status == "exact_match"
    assert by_field["net_amount"].status == "exact_match"
    assert by_field["total_to_pay"].status == "mismatch", \
        f"99.99 vs 120.0 should mismatch, got {by_field['total_to_pay'].status}"
    assert by_field["supplier_vat"].status == "not_in_truth"

    # Critical-field flag is set for all 5 critical fields.
    critical = [f for f in comp.fields if f.critical]
    assert {f.field for f in critical} == {
        "supplier_eik", "supplier_vat", "iban", "total_to_pay", "net_amount",
    }
    return "exact/normalized/mismatch/not_in_truth all detected"


def test_provider_catalog_shows_real_model_names_with_short_descriptions():
    """Catalog shows the real model names — users care which AI they use and
    bring their own API keys. Each profile carries one badge, a short
    headline, plain-language description, best-for tags, tradeoff and cost
    note. No fake numbers leak through measured_summary."""
    from docflow.pipeline import PROVIDER_ALIASES
    from docflow.provider_catalog import (
        PROFILES, RECOMMENDED_PROVIDER, get_profile, measured_summary,
        options_in_display_order,
    )

    # Every runtime provider has a profile, and every profile maps to a runtime alias.
    profiles_by_id = {p.provider: p for p in PROFILES}
    assert set(profiles_by_id) == set(PROVIDER_ALIASES), \
        f"catalog/aliases drift: only-aliases={set(PROVIDER_ALIASES)-set(profiles_by_id)}, " \
        f"only-catalog={set(profiles_by_id)-set(PROVIDER_ALIASES)}"

    # Real-name expectations: each display must contain its model family.
    expected_names = {
        "gemini-3.1-flash-lite": "Gemini",
        "qwen-3-vl-235b":        "Qwen",
    }
    for prov, family in expected_names.items():
        prof = get_profile(prov)
        assert prof is not None, prov
        assert family in prof.display, f"{prov} display '{prof.display}' missing brand '{family}'"

    # Required fields populated and concise.
    for p in PROFILES:
        assert p.display and p.badge and p.headline and p.description, p.provider
        assert p.tradeoff and p.cost, p.provider
        assert p.best_for, f"{p.provider} missing best_for"
        assert len(p.headline) <= 120, f"{p.provider} headline too long"
        # Badge must carry an emoji + label (chip-friendly).
        assert any(c for c in p.badge if ord(c) > 0x2000), \
            f"{p.provider} badge lacks an emoji: {p.badge!r}"

    # Exactly one Recommended; it matches RECOMMENDED_PROVIDER.
    recommended = [p for p in PROFILES if "Препоръчан" in p.badge]
    assert len(recommended) == 1, [p.provider for p in recommended]
    assert recommended[0].provider == RECOMMENDED_PROVIDER

    # measured_summary stays prose — no leaked numbers/percentages.
    for p in PROFILES:
        s = measured_summary(p.provider)
        if s is not None:
            assert "0." not in s, f"{p.provider} leaks a number: {s}"
            assert "%" not in s, f"{p.provider} leaks a percentage: {s}"

    # Display order: recommended provider first.
    ordered = options_in_display_order(list(PROVIDER_ALIASES))
    assert ordered[0] == RECOMMENDED_PROVIDER, \
        f"recommended should be first, got {ordered[0]}"

    # Hard cap on the active provider count for now — focused list, not buffet.
    # Public evaluation deployment surfaces exactly the two benchmarked models.
    assert len(PROFILES) == 2, f"expected 2 providers, got {len(PROFILES)}"

    return f"{len(PROFILES)} models: " + " · ".join(p.display for p in PROFILES)


def test_benchmark_registry_only_lists_real_measurements():
    """The benchmark registry must contain only providers we've actually
    measured. Required fields populated; values in plausible ranges; only
    providers present in PROVIDER_ALIASES."""
    from docflow.eval.benchmarks import BENCHMARKS, benchmarks_as_rows, get_benchmark
    from docflow.pipeline import PROVIDER_ALIASES

    assert len(BENCHMARKS) > 0, "expected at least one benchmarked provider"

    for b in BENCHMARKS:
        assert b.provider in PROVIDER_ALIASES, \
            f"benchmark for unknown provider: {b.provider}"
        assert b.invoice_count > 0, b.provider
        assert 0.0 <= b.avg_quality_score <= 1.0, (b.provider, b.avg_quality_score)
        assert 0.0 <= b.supplier_accuracy <= 1.0, (b.provider, b.supplier_accuracy)
        assert 0.0 <= b.success_rate <= 1.0, (b.provider, b.success_rate)
        assert b.avg_latency_s >= 0, (b.provider, b.avg_latency_s)
        assert b.notes, f"{b.provider}: empty notes"
        assert b.measured_at, f"{b.provider}: empty measured_at"
        # Lookup round-trip
        assert get_benchmark(b.provider) is b

    # Untested providers must return None — not a stub or fabricated record.
    assert get_benchmark("definitely_not_a_real_provider") is None

    # benchmarks_as_rows must produce display-ready dicts.
    rows = benchmarks_as_rows()
    assert len(rows) == len(BENCHMARKS)
    for r in rows:
        assert "Provider" in r and "Quality score" in r and "Latency (s)" in r
    return f"{len(BENCHMARKS)} provider benchmarks; {len(PROVIDER_ALIASES) - len(BENCHMARKS)} untested"


def test_eval_distinguishes_hallucinated_from_not_in_truth():
    """Explicit null in truth + extracted value → 'hallucinated'.
    Missing key in truth → 'not_in_truth' (not graded)."""
    import tempfile as _t

    from docflow.eval.metrics import compare_invoice
    from docflow.eval.truth import load_truth
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    # Truth: supplier_vat explicitly null; supplier_eik present; iban omitted.
    truth_json = '{"supplier_eik": "123456789", "supplier_vat": null}'
    with _t.NamedTemporaryFile(suffix=".truth.json", mode="w", delete=False) as fh:
        fh.write(truth_json)
        path = Path(fh.name)
    try:
        truth = load_truth(path)
        assert truth.has_field("supplier_vat") is True   # explicitly null
        assert truth.has_field("supplier_eik") is True
        assert truth.has_field("iban") is False           # omitted
    finally:
        path.unlink()

    # Extracted produces a VAT (hallucinated against explicit null).
    inv = InvoiceData(supplier=Party(eik="123456789", vat_number="BG999999999"),
                      iban="BG80BNBG96611020345678")
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    comp = compare_invoice(doc, truth, invoice_file="x.pdf", provider="p")
    by_f = {f.field: f for f in comp.fields}
    assert by_f["supplier_eik"].status == "exact_match"
    assert by_f["supplier_vat"].status == "hallucinated", by_f["supplier_vat"].status
    assert by_f["iban"].status == "not_in_truth", by_f["iban"].status
    return f"vat→hallucinated, iban→not_in_truth"


def test_eval_per_field_confidence_populated():
    """Each FieldComparison carries a confidence from quality breakdown."""
    from docflow.eval.metrics import compare_invoice
    from docflow.eval.truth import InvoiceTruth
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    truth = InvoiceTruth(
        supplier_eik="123456789", supplier_vat="BG123456789",
        net_amount=100.0, total_to_pay=120.0, iban="BG80BNBG96611020345678",
    )
    truth._present_keys = {"supplier_eik", "supplier_vat", "net_amount",
                           "total_to_pay", "iban"}
    inv = InvoiceData(
        invoice_number="X",
        supplier=Party(name="X", eik="123456789", vat_number="BG123456789"),
        iban="BG80BNBG96611020345678",
        net_amount=100.0, total_to_pay=120.0, currency="BGN",
    )
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv)
    comp = compare_invoice(doc, truth, invoice_file="x.pdf", provider="p")
    by_f = {f.field: f for f in comp.fields}

    # supplier_eik and supplier_vat split supplier_tax_id (0.10 weight) evenly: 0.05 each.
    assert by_f["supplier_eik"].confidence == 0.05, by_f["supplier_eik"].confidence
    assert by_f["supplier_vat"].confidence == 0.05
    # net_amount and total_to_pay split any_total (0.15) → 0.075 each.
    assert by_f["net_amount"].confidence == 0.075
    assert by_f["total_to_pay"].confidence == 0.075
    # iban has its own signal (0.05) → full weight.
    assert by_f["iban"].confidence == 0.05
    return "per-field confidences from breakdown signals"


def test_eval_aggregate_metrics_critical_and_wbc():
    """Aggregate correctly computes critical_field_accuracy and
    wrong_but_confident_rate."""
    from docflow.eval.metrics import aggregate, compare_invoice
    from docflow.eval.truth import InvoiceTruth
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    truth = InvoiceTruth(supplier_eik="123456789", total_to_pay=100.0,
                         net_amount=80.0, iban="BG80BNBG96611020345678",
                         supplier_vat="BG123456789")

    # 1) Correct extraction, score 0.95 → matched, NOT wrong-but-confident.
    inv_ok = InvoiceData(
        supplier=Party(name="X", eik="123456789", vat_number="BG123456789"),
        iban="BG80BNBG96611020345678",
        net_amount=80.0, total_to_pay=100.0,
    )
    doc_ok = ExtractedDocument(source_path="ok", extraction_method="t",
                               page_count=1, invoice=inv_ok, quality_score=0.95)

    # 2) Wrong total but high score → wrong-but-confident.
    inv_wbc = InvoiceData(
        supplier=Party(name="X", eik="123456789", vat_number="BG123456789"),
        iban="BG80BNBG96611020345678",
        net_amount=80.0, total_to_pay=999.0,
    )
    doc_wbc = ExtractedDocument(source_path="wbc", extraction_method="t",
                                page_count=1, invoice=inv_wbc, quality_score=0.85)

    # 3) Low score, several misses → does NOT count toward wbc rate.
    inv_low = InvoiceData(supplier=Party(name="X"))
    doc_low = ExtractedDocument(source_path="low", extraction_method="t",
                                page_count=1, invoice=inv_low, quality_score=0.15)

    cs = [
        compare_invoice(doc_ok,  truth, invoice_file="a.pdf", provider="p"),
        compare_invoice(doc_wbc, truth, invoice_file="b.pdf", provider="p"),
        compare_invoice(doc_low, truth, invoice_file="c.pdf", provider="p"),
    ]
    a = aggregate(cs, "p")
    assert a.invoice_count == 3
    # Invoice acc: only doc_ok matches all graded fields. doc_low matches 0 graded.
    assert a.invoice_accuracy == round(1/3, 4), a.invoice_accuracy
    # Critical fields graded per invoice = 5 (eik, vat, iban, net_amount, total_to_pay).
    # doc_ok matches 5/5; doc_wbc matches 4/5 (total wrong); doc_low matches 0/5 (all missing).
    # 5+4+0 = 9 of 15 graded critical cells.
    expected_crit = round(9/15, 4)
    assert a.critical_field_accuracy == expected_crit, (a.critical_field_accuracy, expected_crit)
    # Wrong-but-confident: confident invoices are doc_ok and doc_wbc (≥0.7).
    # doc_ok matched all → not wrong. doc_wbc has mismatch → wrong-but-confident.
    assert a.wrong_but_confident_rate == 0.5, a.wrong_but_confident_rate
    return f"invoice_acc={a.invoice_accuracy}, crit_acc={a.critical_field_accuracy}, wbc={a.wrong_but_confident_rate}"


def test_eval_report_writes_all_four_files():
    """write_all produces json + md + csv + xlsx."""
    from openpyxl import load_workbook

    from docflow.eval.metrics import aggregate, compare_invoice
    from docflow.eval.report import write_all
    from docflow.eval.runner import EvaluationResult
    from docflow.eval.truth import InvoiceTruth
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    truth = InvoiceTruth(supplier_name="X", supplier_eik="123456789")
    inv = InvoiceData(supplier=Party(name="X", eik="123456789"))
    doc = ExtractedDocument(source_path="x", extraction_method="t", page_count=1, invoice=inv,
                            quality_score=0.6, quality_coverage=0.7)
    cs = [compare_invoice(doc, truth, invoice_file="x.pdf", provider="prov")]
    result = EvaluationResult(
        started_at="2026-05-18T00:00:00", dataset_dir="/x", providers=["prov"],
        invoices=cs, per_provider={"prov": aggregate(cs, "prov")}, sample_count=1,
    )
    with tempfile.TemporaryDirectory() as td:
        paths = write_all(result, Path(td), basename="test_eval")
        for kind, p in paths.items():
            assert p.exists() and p.stat().st_size > 0, f"empty: {kind}"
        wb = load_workbook(paths["xlsx"])
        assert set(wb.sheetnames) >= {"Summary", "Critical fields", "All fields", "Per-invoice", "Fields"}
        md_text = paths["md"].read_text(encoding="utf-8")
        assert "DocFlow Evaluation Report" in md_text
        assert "Critical-field accuracy" in md_text
    return "json + md + csv + xlsx written"


def test_eval_regression_diff_surfaces_deltas():
    from docflow.eval.regression import regression_report

    prev = {
        "started_at": "2026-05-17T10:00:00",
        "dataset_dir": "/x",
        "per_provider": {
            "prov": {
                "invoice_accuracy": 0.50, "critical_field_accuracy": 0.60,
                "hallucination_rate": 0.30, "wrong_but_confident_rate": 0.20,
                "avg_quality_score": 0.70,
                "critical_field_accuracy_per_field": {
                    "supplier_eik": 0.5, "supplier_vat": 0.5, "iban": 0.6,
                    "total_to_pay": 0.7, "net_amount": 0.7,
                },
            }
        },
        "invoices": [{"invoice_file": "a.pdf", "provider": "prov", "extracted_quality_score": 0.40}],
    }
    curr = {
        "started_at": "2026-05-18T10:00:00",
        "dataset_dir": "/x",
        "per_provider": {
            "prov": {
                "invoice_accuracy": 0.80, "critical_field_accuracy": 0.85,
                "hallucination_rate": 0.10, "wrong_but_confident_rate": 0.05,
                "avg_quality_score": 0.90,
                "critical_field_accuracy_per_field": {
                    "supplier_eik": 0.9, "supplier_vat": 0.9, "iban": 0.8,
                    "total_to_pay": 0.85, "net_amount": 0.85,
                },
            }
        },
        "invoices": [{"invoice_file": "a.pdf", "provider": "prov", "extracted_quality_score": 0.95}],
    }
    md = regression_report(prev, curr)
    assert "+30.0pp" in md  # invoice_accuracy delta
    assert "+25.0pp" in md  # critical_field_accuracy delta
    assert "-20.0pp" in md  # halluc rate dropped (good)
    assert "+0.55" in md    # per-file score change for a.pdf
    return "deltas surface in diff"


def test_custom_api_keys_session_only_no_persistence():
    """Session-scoped API-key overrides:
      - off by default → no env mutation
      - on + values entered → env temporarily overridden, then restored
      - whitespace-only values ignored (don't override)
    No keys are written to disk, env outside the with-block, or logs.
    """
    import os
    import app as _app

    # 1. Off → no overrides regardless of provided values.
    assert _app._resolve_key_overrides({"use_custom_keys": False,
                                         "custom_gemini_key": "abc"}) == {}

    # 2. On but no values → empty overrides (fallback to env stays).
    assert _app._resolve_key_overrides({"use_custom_keys": True}) == {}

    # 3. On with both keys present.
    overrides = _app._resolve_key_overrides({
        "use_custom_keys": True,
        "custom_gemini_key": "gem-xyz",
        "custom_openrouter_key": "sk-or-xyz",
    })
    assert overrides == {
        "GEMINI_API_KEY": "gem-xyz",
        "OPENROUTER_API_KEY": "sk-or-xyz",
    }

    # 4. Whitespace-only is treated as not provided.
    assert _app._resolve_key_overrides({
        "use_custom_keys": True,
        "custom_gemini_key": "   ",
        "custom_openrouter_key": "real",
    }) == {"OPENROUTER_API_KEY": "real"}

    # Save & restore real env so this test never leaks state to later tests
    # that rely on the real GEMINI_API_KEY / OPENROUTER_API_KEY from .env.
    real_g = os.environ.get("GEMINI_API_KEY")
    real_o = os.environ.get("OPENROUTER_API_KEY")
    try:
        # 5. _override_env temporarily sets and then restores exactly. Cover both
        # the "was set before" and the "was not set before" branches.
        os.environ["GEMINI_API_KEY"] = "PRE_EXISTING"
        os.environ.pop("OPENROUTER_API_KEY", None)
        with _app._override_env({"GEMINI_API_KEY": "TMP_G", "OPENROUTER_API_KEY": "TMP_O"}):
            assert os.environ["GEMINI_API_KEY"] == "TMP_G"
            assert os.environ["OPENROUTER_API_KEY"] == "TMP_O"
        # After exit: restored exactly.
        assert os.environ["GEMINI_API_KEY"] == "PRE_EXISTING"
        assert "OPENROUTER_API_KEY" not in os.environ, "must be removed if absent before"

        # 6. No-op when overrides dict is empty.
        os.environ["GEMINI_API_KEY"] = "STAY"
        with _app._override_env({}):
            assert os.environ["GEMINI_API_KEY"] == "STAY"
        assert os.environ["GEMINI_API_KEY"] == "STAY"
    finally:
        # Restore real env exactly.
        if real_g is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = real_g
        if real_o is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = real_o

    return "session-scoped overrides; env restored on exit"


def test_custom_api_keys_never_persisted_to_disk_or_logs():
    """Static regression guard: app.py must not write user keys anywhere
    persistent and must not print/log them."""
    src = Path("app.py").read_text(encoding="utf-8")

    # No writes referencing the custom key state-keys.
    for needle in (
        ".write_text", ".write_bytes", "open(", "Path(",
    ):
        # The first three are dangerous if combined with our key var names;
        # last is generic. We only care about lines that ALSO mention key state.
        pass  # checked below with combined assertion

    # The key var names must not co-occur with any write/log call on the same line.
    forbidden_co_occurrence = ("custom_gemini_key", "custom_openrouter_key")
    danger_verbs = ("print(", "logger.", "log.", ".write(", ".write_text", ".write_bytes",
                    "open(", "json.dump", "json.dumps", "yaml.dump", "pickle.dump")
    for line in src.splitlines():
        if any(name in line for name in forbidden_co_occurrence):
            for verb in danger_verbs:
                assert verb not in line, (
                    f"key state '{forbidden_co_occurrence}' co-occurs with persist verb "
                    f"'{verb}' on line: {line.strip()}"
                )

    return "no persistence path detected"


def test_upload_over_limit_blocks_and_does_not_truncate():
    """Batch-size guard must hard-block, not silently truncate. The helpers
    exposed by app.py are the same ones the UI uses."""
    import app as _app
    assert _app.MAX_FILES_PER_BATCH == 25, _app.MAX_FILES_PER_BATCH

    # Inside the limit → no block.
    assert _app._is_over_batch_limit(0) is False
    assert _app._is_over_batch_limit(1) is False
    assert _app._is_over_batch_limit(25) is False

    # Over the limit → block.
    assert _app._is_over_batch_limit(26) is True

    # Upload-mode message names the public-demo framing and the limit.
    msg = _app._over_batch_limit_message(30).lower()
    assert "публичната демо" in msg
    assert "максимум" in msg
    assert "25" in msg
    assert "качи" in msg

    # Folder variant references "папка" AND includes the actual count so the
    # user knows the source of the overflow.
    fmsg = _app._folder_over_limit_message(42)
    assert "42" in fmsg and "25" in fmsg
    assert "папка" in fmsg.lower()


def test_upload_flow_has_no_silent_truncation():
    """Defence against regression: source must not contain a slice that
    silently trims to MAX_FILES_PER_BATCH."""
    src = Path("app.py").read_text(encoding="utf-8")
    forbidden = (
        "uploaded_files[:MAX_FILES_PER_BATCH]",
        "files[:MAX_FILES_PER_BATCH]",
    )
    for needle in forbidden:
        assert needle not in src, f"silent truncation found: {needle}"


def test_reject_criteria_end_to_end_xlsx():
    """End-to-end acceptance: build the three reject-criteria scenarios as real
    InvoiceData, run them through write_consolidated → reload xlsx → assert
    the rejected combinations do NOT appear in the Фактури sheet.

    Reject criteria (from user spec):
      A. Math mismatch → must NOT be ✅ OK.
      B. Foreign/unknown supplier without IBAN → must NOT be ⚠️ Непълни (N) due to IBAN.
      C. BG supplier with explicit bank payment, no IBAN → must NOT be ✅ OK.

    Synthetic but exercises the real export path (status engine → write_consolidated → xlsx).
    """
    import tempfile as _tempfile
    from pathlib import Path as _Path

    from openpyxl import load_workbook

    from docflow.batch import BatchResult, write_consolidated
    from docflow.schema import (
        ExtractedDocument, InvoiceData, LineItem, Party, VatBreakdown,
    )
    from docflow.validators import validate

    # Scenario A: math mismatch (total ≠ net + vat).
    inv_math = InvoiceData(
        invoice_number="A-MATH-001",
        issue_date="2026-01-01",
        supplier=Party(name="БГ Доставчик ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Клиент"),
        iban="BG80BNBG96611020345678",
        line_items=[LineItem(total_without_vat=100)],
        net_amount=100.0,
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        total_to_pay=200.0,  # ✗ should be 120
        currency="BGN",
        payment_method="По банка",
    )

    # Scenario B: GitHub-style foreign invoice — no BG identifiers, no IBAN, USD.
    inv_github = InvoiceData(
        invoice_number="B-GH-001",
        issue_date="2026-02-01",
        supplier=Party(name="GitHub, Inc."),
        customer=Party(name="Клиент"),
        net_amount=10.0,
        total_to_pay=10.0,
        currency="USD",
    )

    # Scenario C: BG supplier, explicit bank payment, no IBAN.
    inv_bg_bank = InvoiceData(
        invoice_number="C-BG-BANK-001",
        issue_date="2026-03-01",
        supplier=Party(name="БГ Доставчик ООД", eik="987654321", vat_number="BG987654321"),
        customer=Party(name="Клиент"),
        net_amount=100.0,
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        total_to_pay=120.0,
        currency="BGN",
        payment_method="ПЛАЩАНЕ ПО БАНКА",
        # iban deliberately missing
    )

    results: list[BatchResult] = []
    for name, inv in [
        ("A_math.pdf", inv_math),
        ("B_github.pdf", inv_github),
        ("C_bg_bank_no_iban.pdf", inv_bg_bank),
    ]:
        doc = ExtractedDocument(
            source_path=name, extraction_method="fixture", page_count=1, invoice=inv,
        )
        v = validate(doc)
        results.append(BatchResult(_Path(name), None, doc, None, [], v, []))

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        out = _Path(tf.name)
    try:
        write_consolidated(results, out)
        wb = load_workbook(out)
        ws = wb["Фактури"]
        # Build {filename: {column: value}} from sheet.
        headers = [c.value for c in ws[1]]
        rows = {}
        for r in ws.iter_rows(min_row=2, values_only=True):
            row = dict(zip(headers, r))
            rows[row["Файл"]] = row

        # A: math mismatch → status must contain "грешки"/"error" (NOT plain OK).
        st_a = rows["A_math.pdf"]["Статус"]
        assert "OK" not in st_a or "грешки" in st_a, f"REJECT: math mismatch marked OK: {st_a!r}"
        assert "грешки" in st_a, f"REJECT: math mismatch did not surface as validation error: {st_a!r}"

        # B: GitHub-style → no IBAN must not produce INCOMPLETE.
        st_b = rows["B_github.pdf"]["Статус"]
        assert "Непълни" not in st_b, f"REJECT: foreign/unknown invoice flagged INCOMPLETE: {st_b!r}"

        # C: BG + bank + no IBAN → must NOT be OK.
        st_c = rows["C_bg_bank_no_iban.pdf"]["Статус"]
        assert "OK" not in st_c, f"REJECT: BG bank-paid invoice without IBAN marked OK: {st_c!r}"

        return f"A={st_a}, B={st_b}, C={st_c}"
    finally:
        out.unlink(missing_ok=True)


def test_diagnostic_keys_never_gate_status():
    """Selecting diagnostic columns (quality_score, validation_errors,
    registry_status, is_derived) must not cause INCOMPLETE — they aren't
    extracted data."""
    from docflow.schema import InvoiceData, Party
    from docflow.status import compute_status, DIAGNOSTIC_KEYS

    inv = InvoiceData(
        invoice_number="X", issue_date="2026-01-01",
        supplier=Party(name="БГ ООД", eik="123456789", vat_number="BG123456789"),
        customer=Party(name="Y"),
        iban="BG80BNBG96611020345678",
        net_amount=100.0, total_to_pay=120.0,
        currency="BGN", payment_method="По банка",
    )
    required = ["supplier_name"] + sorted(DIAGNOSTIC_KEYS)
    result = compute_status(_make_doc(inv), [], required_column_keys=required)
    assert result.code == "OK", f"diagnostic-only required keys should not affect status, got {result.code}"
    return f"diagnostic keys ignored: {sorted(DIAGNOSTIC_KEYS)}"


# 4b. Currency contract
def test_currency_default_is_none():
    from docflow.schema import InvoiceData
    inv = InvoiceData()
    assert inv.currency is None, f"expected None default, got {inv.currency!r}"


def test_currency_quality_not_awarded_for_default():
    from docflow.quality import invoice_quality_score
    from docflow.schema import InvoiceData, Party
    base = dict(supplier=Party(name="X"))
    s_no = invoice_quality_score(InvoiceData(**base))
    s_eur = invoice_quality_score(InvoiceData(**base, currency="EUR"))
    s_bgn = invoice_quality_score(InvoiceData(**base, currency="BGN"))
    assert s_eur > s_no, f"explicit EUR should beat None: {s_eur} vs {s_no}"
    assert s_bgn > s_no, f"explicit BGN should beat None: {s_bgn} vs {s_no}"
    assert abs(s_eur - s_bgn) < 1e-9, "EUR and BGN should score the same"
    return f"None={s_no}, EUR={s_eur}, BGN={s_bgn}"


def test_currency_display_falls_back_to_EUR():
    from docflow.columns import get_row_value
    from docflow.schema import ExtractedDocument, InvoiceData, Party

    inv_none = InvoiceData(supplier=Party(name="X"))
    doc_none = ExtractedDocument(
        source_path="x", extraction_method="t", page_count=1, invoice=inv_none,
    )
    assert get_row_value("currency", doc_none) == "EUR"

    inv_bgn = InvoiceData(supplier=Party(name="X"), currency="BGN")
    doc_bgn = ExtractedDocument(
        source_path="x", extraction_method="t", page_count=1, invoice=inv_bgn,
    )
    assert get_row_value("currency", doc_bgn) == "BGN"


def test_currency_none_is_never_mutated_to_EUR():
    """Invariant: no exporter, validator, registry, or quality call may
    promote a None currency to 'EUR' on the InvoiceData itself. 'EUR' is a
    pure display-layer fallback."""
    clean_registry()
    import tempfile as _tempfile
    from pathlib import Path as _Path

    from docflow.columns import get_row_value
    from docflow.output import write_excel
    from docflow.quality import invoice_quality_score
    from docflow.registry import SupplierRegistry
    from docflow.schema import (
        ExtractedDocument, InvoiceData, LineItem, Party, VatBreakdown,
    )
    from docflow.validators import validate

    inv = InvoiceData(
        invoice_number="N",
        issue_date="2026-01-01",
        supplier=Party(name="X", vat_number="BG123456789"),
        customer=Party(name="Y"),
        line_items=[LineItem(total_without_vat=100, vat_percent=20)],
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        net_amount=100,
        total_to_pay=120,
        iban="BG80BNBG96611020345678",
    )
    assert inv.currency is None, "precondition: created without currency"

    doc = ExtractedDocument(
        source_path="x.pdf", extraction_method="t", page_count=1, invoice=inv,
    )

    # 1. Quality scoring must not mutate.
    invoice_quality_score(inv)
    assert inv.currency is None, "quality.invoice_quality_score mutated currency"

    # 2. Validation must not mutate.
    validate(doc)
    assert inv.currency is None, "validators.validate mutated currency"

    # 3. Registry enrich + confirmed record must not mutate.
    r = SupplierRegistry()
    r.record(inv, human_confirmed=True)
    assert inv.currency is None, "registry.record mutated currency"
    enriched, _ = r.enrich(inv)
    assert inv.currency is None, "registry.enrich mutated source currency"
    # enriched is a deep copy — its currency may differ if registry had one,
    # but here we never stored a currency so it should still be None.
    assert enriched.currency is None, "registry.enrich invented a currency"

    # 4. Column resolver returns 'EUR' fallback without touching inv.
    row_val = get_row_value("currency", doc)
    assert row_val == "EUR", f"display fallback should be EUR, got {row_val!r}"
    assert inv.currency is None, "columns.get_row_value mutated currency"

    # 5. Excel writer must not mutate.
    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        out = _Path(tf.name)
    try:
        write_excel(doc, out)
    finally:
        out.unlink(missing_ok=True)
    assert inv.currency is None, "output.write_excel mutated currency"

    return "None survived quality, validate, registry record+enrich, columns, write_excel"


# 4c. Registry human-confirmation gate
def test_registry_no_auto_write_without_confirmation():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    inv = InvoiceData(supplier=Party(name="X", vat_number="BG123456789"))
    findings = r.record(inv)  # default human_confirmed=False
    assert r.all() == [], f"unexpected write: {r.all()}"
    assert len(findings) == 1 and findings[0].code == "registry_skipped"
    assert findings[0].level == "info"
    return "default call returns registry_skipped, DB untouched"


def test_registry_writes_only_after_confirmation():
    clean_registry()
    from docflow.registry import SupplierRegistry
    from docflow.schema import InvoiceData, Party

    r = SupplierRegistry()
    inv = InvoiceData(supplier=Party(name="X", vat_number="BG123456789"))

    # First call without confirmation → no write.
    r.record(inv)
    assert r.all() == [], "unconfirmed call must not persist"

    # Same data with confirmation → write.
    findings = r.record(inv, human_confirmed=True)
    assert any(f.code == "supplier_new" for f in findings), [f.code for f in findings]
    suppliers = r.all()
    assert len(suppliers) == 1 and suppliers[0]["eik"] == "123456789"
    return "unconfirmed→skipped, confirmed→persisted"


# 5. End-to-end pipeline tests
def test_pipeline_gemini_path():
    if not has_gemini_key():
        raise SkipTest("GEMINI_API_KEY not set")
    if not SAMPLE_JPG.exists():
        raise SkipTest(f"missing {SAMPLE_JPG.name}")

    from docflow.pipeline import extract
    try:
        doc = extract(SAMPLE_JPG)
    except RuntimeError as e:
        if "quota" in str(e).lower() or "503" in str(e):
            raise SkipTest(f"Gemini unavailable: {e}")
        raise
    if not doc.extraction_method.startswith("gemini"):
        raise SkipTest(f"Gemini did not run (fallback to {doc.extraction_method}, likely quota/503)")
    assert doc.invoice is not None
    inv = doc.invoice
    assert inv.invoice_number == "0000000321", f"got {inv.invoice_number}"
    assert inv.supplier.eik == "9601270035", f"got {inv.supplier.eik}"
    assert inv.customer.eik == "207260294", f"got {inv.customer.eik}"
    assert len(inv.line_items) == 5, f"got {len(inv.line_items)} items"
    assert inv.total_to_pay == 37824.66, f"got {inv.total_to_pay}"
    return f"all 5 line items, total=37824.66, EIKs correct"


# 6. CLI errors
def test_cli_missing_file():
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "docflow", "no_such_file.pdf"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert r.returncode == 2
    assert "not found" in r.stderr


def test_cli_unsupported_extension():
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tf.write(b"hello")
        tmp = tf.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "docflow", tmp],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert r.returncode == 1
        assert "No extractor can handle" in r.stderr
    finally:
        os.unlink(tmp)


# 7. Output
def test_excel_writer_produces_invoice_sheets():
    from docflow.output import write_excel
    from docflow.schema import (
        ExtractedDocument, InvoiceData, LineItem, Party, VatBreakdown,
    )
    from openpyxl import load_workbook

    inv = InvoiceData(
        invoice_number="TEST-1",
        supplier=Party(name="X", eik="123456789"),
        line_items=[LineItem(description="A", total_without_vat=100)],
        vat_breakdown=[VatBreakdown(rate_percent=20, base_amount=100, vat_amount=20)],
        total_to_pay=120,
    )
    doc = ExtractedDocument(
        source_path="test", extraction_method="test", page_count=1, invoice=inv,
    )
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        out = Path(tf.name)
    try:
        write_excel(doc, out)
        wb = load_workbook(out)
        assert "invoice" in wb.sheetnames
        assert "line_items" in wb.sheetnames
        assert "vat" in wb.sheetnames
        assert "_meta" in wb.sheetnames
        return f"sheets: {wb.sheetnames}"
    finally:
        out.unlink(missing_ok=True)


def test_batch_consolidated_structure():
    from docflow.batch import write_consolidated, BatchResult
    from docflow.schema import ExtractedDocument, InvoiceData, Party
    from openpyxl import load_workbook

    doc = ExtractedDocument(
        source_path="x.pdf", extraction_method="test", page_count=1,
        invoice=InvoiceData(invoice_number="1", supplier=Party(eik="123456789")),
    )
    results = [BatchResult(Path("x.pdf"), Path("x.xlsx"), doc, None, [], [], [])]
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        out = Path(tf.name)
    try:
        write_consolidated(results, out)
        wb = load_workbook(out)
        assert wb.sheetnames == ["Фактури", "Артикули", "ДДС", "Validation", "_meta"]
    finally:
        out.unlink(missing_ok=True)


# 8. Env loader
def test_env_loader_respects_existing_env():
    from docflow.env import load_env_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as tf:
        tf.write("DOCFLOW_TEST_VAR=from_file\n")
        tf.write("# comment ignored\n")
        tf.write("DOCFLOW_TEST_VAR2=from_file\n")
        tmp = Path(tf.name)
    try:
        os.environ["DOCFLOW_TEST_VAR"] = "from_shell"
        os.environ.pop("DOCFLOW_TEST_VAR2", None)
        load_env_file(tmp)
        assert os.environ["DOCFLOW_TEST_VAR"] == "from_shell"  # shell wins
        assert os.environ["DOCFLOW_TEST_VAR2"] == "from_file"  # file fills gap
    finally:
        tmp.unlink()
        os.environ.pop("DOCFLOW_TEST_VAR", None)
        os.environ.pop("DOCFLOW_TEST_VAR2", None)


# ─── Runner ─────────────────────────────────────────────────────────────────

TESTS: list[tuple[str, Callable]] = [
    ("schema imports + is_useful_invoice", test_schema_imports),
    ("pipeline excludes pdfplumber and auto; lists explicit providers", test_pipeline_registers_extractors),
    ("ЕИК validator accepts 9/10/13 digits, rejects 5", test_eik_validator_accepts_9_10_13_digits),
    ("ЕИК validator strips BG prefix", test_eik_validator_strips_BG_prefix),
    ("IBAN validator: valid / too long / bad checksum", test_iban_validator),
    ("invoice math validator: 4 cross-checks", test_invoice_math_validator),
    ("invoice math catches wrong remaining", test_invoice_math_catches_wrong_remaining),
    ("validate_eiks_in_text only labeled (not invoice numbers)", test_validate_eiks_in_text_only_labeled),
    ("text_parser extracts IBAN + invoice number", test_text_parser_extracts_iban_and_invoice_number),
    ("text_parser does NOT set party fields", test_text_parser_does_not_set_party_fields),
    ("text_parser returns None for empty input", test_text_parser_returns_none_for_empty_text),
    ("registry: new supplier insert + ЕИК from VAT", test_registry_new_supplier),
    ("registry: record/enrich do not mutate input", test_registry_does_not_mutate_input),
    ("registry: auto-fill missing fields on subsequent invoice", test_registry_auto_fills_missing_fields),
    ("registry: enrich replaces hallucinated IBAN", test_registry_enrich_overrides_mismatch),
    ("provenance: derived_fields not in JSON schema", test_derived_fields_not_in_json_schema),
    ("provenance: derivation still records via PrivateAttr", test_derived_fields_provenance_still_records),
    ("status: foreign supplier without IBAN is not INCOMPLETE", test_foreign_supplier_no_iban_not_incomplete),
    ("status: BG supplier with bank payment + no IBAN is INCOMPLETE", test_bg_supplier_no_iban_incomplete_with_bank_payment),
    ("status: math mismatch always produces VALIDATION_ERROR", test_math_mismatch_always_produces_error),
    ("status: supplier_vat exempt for FOREIGN and known-SaaS", test_supplier_vat_exempt_for_foreign_and_known_saas),
    ("status: known-SaaS exemption does not fabricate VAT", test_known_saas_does_not_fabricate_vat),
    ("provider: explicit selection does not fallback by default", test_explicit_provider_does_not_fallback_by_default),
    ("provider: explicit + allow_fallback=True falls back on quota", test_explicit_provider_with_allow_fallback_uses_backup),
    ("provider: auto falls back on quota", test_auto_provider_falls_back_on_quota),
    ("provider: quota error does not poison earlier quality result", test_provider_error_does_not_poison_quality_metrics),
    ("provider: error classification (quota/auth/network/other)", test_provider_error_classification),
    ("batch: summarize_batch separates provider failures from doc errors", test_summarize_batch_counts),
    ("quality: math error caps score at 0.5", test_quality_score_capped_by_math_error),
    ("quality: invalid IBAN checksum caps score at 0.6", test_quality_score_capped_by_iban_checksum),
    ("quality: breakdown lists signal contributions", test_quality_breakdown_lists_signals),
    ("vendors: GitHub/Canva/OpenAI/etc. deterministically FOREIGN", test_vendor_profiles_deterministic_foreign),
    ("status: OK note never contradicts present IBAN", test_ok_note_does_not_lie_about_present_iban),
    ("diagnostics: derived_fields column lists paths (not да/не)", test_derived_fields_column_returns_list_not_boolean),
    ("calibration: derived signals get half-weight discount", test_derived_signals_get_discounted),
    ("calibration: coverage > confidence when signals are derived", test_coverage_and_confidence_are_different_concepts),
    ("calibration: vendor registry is structured, not inline-spaghetti", test_vendor_registry_is_structured_not_inline),
    ("calibration: summarize_batch returns per-provider stats", test_summarize_batch_per_provider_stats),
    ("eval: truth schema + discover_dataset finds (invoice, truth) pairs", test_eval_truth_schema_and_discovery),
    ("eval: compare_invoice detects exact/normalized/mismatch/missing", test_eval_compare_invoice_matches_and_mismatches),
    ("provider catalog shows real model names with descriptions", test_provider_catalog_shows_real_model_names_with_short_descriptions),
    ("benchmark registry only lists real measurements", test_benchmark_registry_only_lists_real_measurements),
    ("eval: hallucinated status differs from not_in_truth", test_eval_distinguishes_hallucinated_from_not_in_truth),
    ("eval: FieldComparison carries per-field confidence", test_eval_per_field_confidence_populated),
    ("eval: aggregate computes critical-field and wrong-but-confident rates", test_eval_aggregate_metrics_critical_and_wbc),
    ("eval: write_all produces json + md + csv + xlsx", test_eval_report_writes_all_four_files),
    ("eval: regression diff surfaces top-line + per-file deltas", test_eval_regression_diff_surfaces_deltas),
    ("custom keys: session-scoped overrides resolved correctly", test_custom_api_keys_session_only_no_persistence),
    ("custom keys: no persistence of user-entered API keys", test_custom_api_keys_never_persisted_to_disk_or_logs),
    ("upload: over-limit blocks (no silent truncation)", test_upload_over_limit_blocks_and_does_not_truncate),
    ("upload: app.py contains no silent-truncation slice", test_upload_flow_has_no_silent_truncation),
    ("acceptance: 3 reject-criteria scenarios through real xlsx export", test_reject_criteria_end_to_end_xlsx),
    ("status: diagnostic columns never gate status", test_diagnostic_keys_never_gate_status),
    ("currency: schema default is None", test_currency_default_is_none),
    ("currency: quality score awards only for explicit currency", test_currency_quality_not_awarded_for_default),
    ("currency: display falls back to EUR when not extracted", test_currency_display_falls_back_to_EUR),
    ("currency: None is never mutated to EUR by any module", test_currency_none_is_never_mutated_to_EUR),
    ("registry: no auto-write without human_confirmed", test_registry_no_auto_write_without_confirmation),
    ("registry: writes only after human_confirmed=True", test_registry_writes_only_after_confirmation),
    ("pipeline: gemini path on JPG (real EuroFaktura)", test_pipeline_gemini_path),
    ("CLI: missing file → exit 2", test_cli_missing_file),
    ("CLI: unsupported extension → exit 1", test_cli_unsupported_extension),
    ("output.write_excel: invoice/line_items/vat/_meta sheets", test_excel_writer_produces_invoice_sheets),
    ("batch.write_consolidated: Фактури/Артикули/ДДС/Validation/_meta structure", test_batch_consolidated_structure),
    ("env loader: shell wins, file fills gaps", test_env_loader_respects_existing_env),
]


def main() -> int:
    print(f"Running {len(TESTS)} tests...\n")
    for name, fn in TESTS:
        run(name, fn)

    pass_count = sum(1 for _, s, _ in results if s == PASS)
    fail_count = sum(1 for _, s, _ in results if s == FAIL)
    skip_count = sum(1 for _, s, _ in results if s == SKIP)
    total = len(results)

    print(f"\n{'═' * 60}")
    print(f"  PASS: {pass_count}/{total}    FAIL: {fail_count}    SKIP: {skip_count}")
    print(f"{'═' * 60}")

    if fail_count:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  ✗ {name}\n      {detail}")

    if skip_count:
        print("\nSkipped tests:")
        for name, status, detail in results:
            if status == SKIP:
                print(f"  ⊘ {name}: {detail}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
