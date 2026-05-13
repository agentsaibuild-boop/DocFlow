# DocFlow

**Bulgarian Cyrillic invoice & protocol extraction system.** Reads PDF or image, returns structured data (Excel + canonical JSON), validates, and learns per-supplier from corrections.

Status: **v0.1 working prototype.** Single-document extraction proven end-to-end on a real Bulgarian invoice (Gemini 2.5 Flash path) and a born-digital PDF (pdfplumber path). Batch processing produces consolidated summary.

---

## What it does

Input: a Bulgarian fakтура or приемно-предавателен протокол, in any of these forms — PDF (born-digital or scanned), JPG, PNG, TIFF, BMP, WEBP.

Output:
- **Excel file** with structured sheets: invoice header, line items, VAT breakdown, audit metadata
- **Canonical JSON** matching the `InvoiceData` Pydantic schema (19 typed fields)
- **Validation report**: ЕИК/ПИК format, IBAN checksum, math cross-checks (line totals, VAT, payment math)
- **Registry update**: per-supplier database that auto-corrects known-good fields on subsequent invoices

What makes it different from generic OCR pipelines:

1. **Vision-LLM first**, not generic OCR. Gemini 2.5 Flash with Pydantic structured output reads the document as an invoice, not as pixels — so numbers are preserved exact, decimal separators don't drift, columns stay aligned.
2. **Per-client learning**, like Controlisy. Once a human corrects the IBAN of supplier X, every future invoice from X is auto-corrected silently. The OCR engine is replaceable; the registry is the moat.
3. **Math validation always runs**, regardless of which extractor produced the numbers. Σ(line items) = net amount; net + VAT = total; total − paid = remaining. Failures flag the document for human review automatically.
4. **Graceful fallback chain**: when Gemini is unavailable (503, free-tier quota), system falls through to pdfplumber (born-digital PDF) or EasyOCR+img2table (image). Output quality drops but the pipeline never crashes.

---

## Architecture

```
                    ┌─────────────┐
   input file  ─→   │  pipeline   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌─────────────┐    ┌─────────────────┐
  │  Gemini  │      │  pdfplumber │    │  EasyOCR +      │
  │  vision  │      │  (PDF text) │    │  img2table      │
  └────┬─────┘      └──────┬──────┘    └────────┬────────┘
       │                   │                    │
       ▼ InvoiceData       ▼ tables             ▼ tables
  ┌─────────────────────────────────────────────────────┐
  │  text_parser    (fills minimal InvoiceData          │
  │                  if extractor produced none)        │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  SupplierRegistry.enrich()                          │
  │  registry-known fields override extraction          │
  │  (name, address, vat, IBAN, bank, BIC)              │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  validators.validate()                              │
  │  • ЕИК format (9/10/13 digits)                      │
  │  • IBAN length + mod-97 checksum                    │
  │  • Math: Σlines = net, subtotal − disc = net,       │
  │    net + VAT = total, total − paid = remaining      │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  output.write_excel() — invoice/line_items/vat/_meta│
  │  SupplierRegistry.record() — fill missing, count++  │
  └─────────────────────────────────────────────────────┘
```

Trust order, highest to lowest:
1. **Registry** — facts verified by past human review
2. **Gemini typed extraction** — high-confidence structured output
3. **Regex on raw text** — only unambiguous patterns (IBAN with checksum, labeled invoice number); never party identification

---

## Quick start

```bash
# 1. Clone, create venv, install
python3 -m venv .venv
.venv/bin/pip install -e .

# 2. Configure Gemini API key (free tier from https://aistudio.google.com/apikey)
echo 'GEMINI_API_KEY=your_key_here' > .env
chmod 600 .env

# 3. Run on a single document
.venv/bin/python -m docflow path/to/invoice.pdf

# 4. Batch a folder
.venv/bin/python -m docflow path/to/invoices/

# Output: output/<name>.xlsx per document, output/_summary.xlsx for batches
```

`registry.db` (SQLite) is created automatically on first invoice with a recognized supplier.

To correct a known supplier's IBAN after a human review:

```python
from docflow.registry import SupplierRegistry
SupplierRegistry().correct("9601270035", iban="BG80BNBG96611020345678")
```

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Schema | Pydantic v2 | Native to Gemini structured output; runtime validation |
| Vision-LLM | Gemini 2.5 Flash | Cheapest top-tier on document benchmarks; native PDF/image; Cyrillic-strong |
| Born-digital PDF | pdfplumber | Most accurate text+table extraction from PDFs with text layer |
| Image OCR fallback | EasyOCR | Cyrillic supported; pure pip install |
| Table detection (image) | img2table | Auto-detect table structure from image, OCR each cell |
| Image preprocessing | OpenCV | Upscale + CLAHE + denoise before OCR |
| Excel output | openpyxl | Direct .xlsx write, Cyrillic-safe |
| Registry | SQLite | Zero-config, embedded, file-based |
| Env loading | hand-rolled (`docflow/env.py`) | No dotenv dependency |

Total runtime dependencies: 7 packages.

---

## What's working (v0.1)

- ✓ Single-document extraction: PDF, JPG, PNG, TIFF, BMP, WEBP
- ✓ Folder batch processing with `_summary.xlsx`
- ✓ Gemini structured extraction → 19-field canonical JSON
- ✓ Math validation (4 cross-checks)
- ✓ Format validation (ЕИК 9/10/13, IBAN mod-97)
- ✓ Per-supplier registry (auto-fill missing fields, override on registry mismatch)
- ✓ Graceful fallback (Gemini 503/quota → pdfplumber → EasyOCR)
- ✓ Retry with exponential backoff for Gemini server errors
- ✓ Conservative regex fallback parser (IBAN, invoice number only)
- ✓ Excel output with separate sheets: invoice / line_items / vat / _meta

## What's planned

- Scanned PDF rendering (PyMuPDF) when Gemini unavailable — currently scanned PDF without Gemini falls through to no extractor
- Чл. 117 protocols (self-charged VAT) — schema supports, validation rules pending
- Multi-page invoice testing on real samples
- CSV / XML export options alongside Excel
- Registry CLI (list, correct, export) instead of Python API
- Test suite (pytest)
- Logging framework (structured logs instead of print)
- Optional ensemble mode (Gemini + Claude vote on numbers; flag disagreements)

---

## File map

```
docflow/
├── __main__.py              CLI entry: python -m docflow <path>
├── cli.py                   Argument parsing, single-file vs batch routing
├── env.py                   Loads .env into os.environ (shell wins)
├── pipeline.py              Extractor selection + fallback orchestration
├── batch.py                 Folder traversal, per-file processing, _summary.xlsx
├── schema.py                Pydantic models: ExtractedDocument, InvoiceData, Party, LineItem, VatBreakdown
├── extractors/
│   ├── __init__.py          Extractor Protocol
│   ├── gemini_extractor.py  Vision-LLM with structured output
│   ├── pdfplumber_extractor.py   Born-digital PDF text + tables
│   └── easyocr_image_extractor.py   img2table + EasyOCR for images
├── preprocess.py            cv2 image cleanup before OCR
├── text_parser.py           Conservative regex fallback (IBAN + invoice number only)
├── registry.py              SupplierRegistry (SQLite) + enrich/record/correct
├── validators.py            ЕИК / IBAN / math cross-check validation
└── output.py                Excel writer (invoice / line_items / vat / _meta sheets)
```

External:
- `LESSONS_LEARNED.json` — captured lessons, design principles, bugs caught (machine-readable)
- `CLAUDE.md` — AI Engineer principles followed during development
- `pyproject.toml` — declarative deps and entry point
- `.env` — local secrets (gitignored)
- `registry.db` — SQLite per-supplier knowledge base (gitignored)
- `samples/` — input documents (gitignored except .gitkeep)
- `output/` — Excel results (gitignored except .gitkeep)

---

## Cost & quota notes

Gemini 2.5 Flash on free tier: **20 requests/day** per project. Consider:
- `GEMINI_MODEL=gemini-2.5-flash-lite` in `.env` for higher free quota during dev
- Paid tier for production (~$0.0003 per typical invoice; 1000 invoices ≈ $0.30)

Cost breakdown per invoice (Gemini 2.5 Flash, paid tier):
- Input: image (~1500 tokens) + prompt (~250 tokens) → ~$0.000165
- Output: structured JSON (~500 tokens) → ~$0.000150
- Total: ~$0.0003 per invoice

Other paths (pdfplumber, EasyOCR+img2table) are free, run locally, no API call.
