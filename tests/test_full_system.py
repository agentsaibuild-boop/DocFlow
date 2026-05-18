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
    from docflow.pipeline import EXTRACTORS
    names = [e.name for e in EXTRACTORS]
    assert names[0] == "pdfplumber", f"first should be pdfplumber, got {names[0]}"
    assert "claude" in names and "gemini" in names, names
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
    # No VAT, no EIK → origin is UNKNOWN (honest), not falsely "foreign".
    assert supplier_origin(inv) == SupplierOrigin.UNKNOWN
    assert not requires_iban(inv), "unknown-origin + no payment_method → IBAN not required"

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

    # 1. GitHub: unknown origin (no VAT, no EIK), name matches SaaS list.
    inv_gh = InvoiceData(
        invoice_number="GH-1", issue_date="2026-05-01",
        supplier=Party(name="GitHub, Inc."),
        customer=Party(name="Купувач"),
        net_amount=10.0, total_to_pay=10.0, currency="USD",
    )
    assert supplier_origin(inv_gh) == SupplierOrigin.UNKNOWN
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
    assert s == {
        "processed": 6,
        "extracted_ok": 1,
        "validation_errors": 1,
        "no_data": 1,
        "provider_failures": 2,
        "unknown_errors": 1,
    }, s
    return str(s)


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
def test_pipeline_pdfplumber_path():
    if not SAMPLE_BORN_DIGITAL_PDF.exists():
        raise SkipTest(f"missing {SAMPLE_BORN_DIGITAL_PDF.name}")

    # Force pdfplumber-only. With the best-of-quality pipeline ("auto" now
    # keeps trying past pdfplumber if its quality_score < threshold), pinning
    # the provider is how a test verifies a specific extractor end-to-end.
    saved_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        from docflow.pipeline import extract
        doc = extract(SAMPLE_BORN_DIGITAL_PDF, provider="pdfplumber")
        assert doc.extraction_method.startswith("pdfplumber"), doc.extraction_method
        assert doc.tables, "expected at least 1 table from pdfplumber"
        return f"method={doc.extraction_method}, tables={len(doc.tables)}, pages={doc.page_count}"
    finally:
        if saved_key:
            os.environ["GEMINI_API_KEY"] = saved_key


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
    ("pipeline registers extractors with pdfplumber first", test_pipeline_registers_extractors),
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
    ("acceptance: 3 reject-criteria scenarios through real xlsx export", test_reject_criteria_end_to_end_xlsx),
    ("status: diagnostic columns never gate status", test_diagnostic_keys_never_gate_status),
    ("currency: schema default is None", test_currency_default_is_none),
    ("currency: quality score awards only for explicit currency", test_currency_quality_not_awarded_for_default),
    ("currency: display falls back to EUR when not extracted", test_currency_display_falls_back_to_EUR),
    ("currency: None is never mutated to EUR by any module", test_currency_none_is_never_mutated_to_EUR),
    ("registry: no auto-write without human_confirmed", test_registry_no_auto_write_without_confirmation),
    ("registry: writes only after human_confirmed=True", test_registry_writes_only_after_confirmation),
    ("pipeline: pdfplumber path on born-digital PDF", test_pipeline_pdfplumber_path),
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
