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
    findings = r.record(inv)
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
    r.record(inv)
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
    r.record(InvoiceData(supplier=Party(name="X", vat_number="BG123456789")))
    r.record(InvoiceData(
        supplier=Party(name="X", vat_number="BG123456789", address="ул. Тест 1"),
        iban="BG80BNBG96611020345678",
    ))
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
    ))
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
    """Invariant: no exporter, validator, or quality call may promote a None
    currency to 'EUR' on the InvoiceData itself. 'EUR' is a pure display-layer
    fallback."""
    import tempfile as _tempfile
    from pathlib import Path as _Path

    from docflow.columns import get_row_value
    from docflow.output import write_excel
    from docflow.quality import invoice_quality_score
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

    invoice_quality_score(inv)
    assert inv.currency is None, "quality.invoice_quality_score mutated currency"

    validate(doc)
    assert inv.currency is None, "validators.validate mutated currency"

    row_val = get_row_value("currency", doc)
    assert row_val == "EUR", f"display fallback should be EUR, got {row_val!r}"
    assert inv.currency is None, "columns.get_row_value mutated currency"

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        out = _Path(tf.name)
    try:
        write_excel(doc, out)
    finally:
        out.unlink(missing_ok=True)
    assert inv.currency is None, "output.write_excel mutated currency"

    return "None survived quality, validate, columns, write_excel"


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
    ("currency: schema default is None", test_currency_default_is_none),
    ("currency: quality score awards only for explicit currency", test_currency_quality_not_awarded_for_default),
    ("currency: display falls back to EUR when not extracted", test_currency_display_falls_back_to_EUR),
    ("currency: None is never mutated to EUR by any module", test_currency_none_is_never_mutated_to_EUR),
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
