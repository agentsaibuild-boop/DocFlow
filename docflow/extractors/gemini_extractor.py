import os
from pathlib import Path

from docflow.extraction_prompt import LINE_ITEMS_SECTION
from docflow.schema import ExtractedDocument, ExtractedTable, InvoiceData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}

DEFAULT_MODEL = "gemini-2.5-flash"

EXTRACTION_PROMPT = """\
You are an expert invoice data extractor specialized in Bulgarian (български) invoices and acceptance protocols (приемно-предавателни протоколи).

Extract ALL structured data from this document into the provided schema.

═══ CRITICAL: BULGARIAN NUMBER FORMAT ═══

In Bulgaria, the conventions are OPPOSITE of English:
  • COMMA (,) is the DECIMAL separator
  • DOT (.) is the THOUSANDS separator
  • SPACE ( ) is also a thousands separator

Examples — read EACH carefully:
  • "1.999,00"  →  1999.00      (one thousand nine hundred ninety-nine, zero cents)
  • "1.000,00"  →  1000.00      (ONE THOUSAND, ABSOLUTELY NOT 1.0!)
  • "36.265,00" →  36265.00     (thirty-six thousand two hundred sixty-five)
  • "799,60"    →  799.60       (seven hundred ninety-nine and 60 cents)
  • "0,50"      →  0.50         (fifty cents)
  • "1 230,00"  →  1230.00      (space as thousands)

⚠ COMMON MISTAKE TO AVOID: NEVER interpret the dot as a decimal separator.
"1.000" in a Bulgarian invoice is NEVER 1.0 — it is always 1000.
"Платени: 1.000,00 лв" means PAID ONE THOUSAND LEVS, not 1.0 lev.

Sanity check yourself: invoice amounts are in lev (BGN). A line item priced "1.000,00" is 1000 levs, which is reasonable. "1.0" levs would be one lev, which is unusual for most goods/services.

═══ FIELD RULES (Bulgarian invoices) ═══

- Identify supplier (Изпълнител) vs customer (Получател) correctly — they are usually labeled.
- ЕИК/ПИК = Bulgarian identifier: 9 digits (legal entity), 10 digits (ЕГН — sole proprietor), or 13 digits (БУЛСТАТ). No prefix.
- ИН по ДДС / VAT number = OPTIONAL. Only present if VAT-registered. Format BG + digits. If no VAT line shown, return null — do NOT fabricate.
- IBAN = starts with BG, exactly 22 characters. Only present if shown.
- VAT breakdown rates in Bulgaria are typically 9% or 20%.
- For dates, return ISO format YYYY-MM-DD.

═══ NON-BULGARIAN / INTERNATIONAL INVOICES ═══

For invoices NOT in Bulgarian format (e.g. US, EU, UK), capture equivalents:
- `eik` field → put the supplier's local Tax ID / EIN / VAT number / Company number (without country prefix). E.g. Anthropic EIN goes here.
- `vat_number` field → put country-prefixed VAT/Tax number if present (e.g. "GB123456789", "DE123456789"). Otherwise null.
- `iban` field → put IBAN if shown (any country, not just BG). For ACH/wire-only US invoices, leave null.
- `currency` field → "USD", "EUR", "GBP", etc. as shown on invoice. Do NOT default to BGN for non-BG invoices.
- `vat_breakdown` → for US "sales tax" or EU VAT, capture rate and amount. If invoice shows "$0.00 tax", use rate_percent=0, vat_amount=0.
- Line items still get captured the same way.
- Numbers: use the invoice's locale convention. US: "1,000.00" = 1000.00. EU: "1.000,00" = 1000.00 (same value, different notation).

═══ COMMON ═══

- Return null only when field genuinely absent. DO NOT hallucinate.
""" + LINE_ITEMS_SECTION


class GeminiExtractor:
    api_key_env = "GEMINI_API_KEY"

    def __init__(self, model: str | None = None, name: str | None = None) -> None:
        self._client = None
        self._model_override = model
        self.name = name or "gemini"

    @property
    def _model(self) -> str:
        return self._model_override or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    def _get_client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} not set. Get a free key at https://aistudio.google.com/apikey"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return False
        return bool(os.environ.get(self.api_key_env))

    def extract_structured(self, path: Path, schema: type, prompt: str):
        """Same Gemini call as invoices: vision + JSON schema. Schema/prompt vary per project."""
        import time

        from google.genai import errors, types

        client = self._get_client()
        mime = self._mime_for(path)
        file_bytes = path.read_bytes()
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        )

        delays_str = os.environ.get("GEMINI_RETRY_DELAYS", "")
        delays = [int(d) for d in delays_str.split(",") if d.strip()] if delays_str else []
        max_attempts = len(delays) + 1
        last_err: Exception | None = None
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=[part, prompt],
                    config=config,
                )
                break
            except errors.ServerError as e:
                last_err = e
                if attempt <= len(delays):
                    delay = delays[attempt - 1]
                    print(f"  gemini busy (503), attempt {attempt}/{max_attempts} failed, retry in {delay}s", flush=True)
                    time.sleep(delay)
            except errors.ClientError as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    raise RuntimeError(
                        f"Gemini daily quota exhausted for model '{self._model}'. "
                        f"Free tier: 20 req/day for gemini-2.5-flash, more for gemini-2.5-flash-lite. "
                        f"Set GEMINI_MODEL=gemini-2.5-flash-lite in .env, wait until UTC midnight, or upgrade to paid tier."
                    ) from e
                raise
        if response is None:
            raise RuntimeError(f"Gemini unavailable after {max_attempts} attempt(s) (set GEMINI_RETRY_DELAYS to retry, e.g. '5,15')") from last_err

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise RuntimeError(
                f"Gemini returned no parsed {schema.__name__} — possible safety block, "
                "malformed JSON, or unsupported document"
            )
        full_text = response.text or ""
        page_count = self._count_pages(path) if path.suffix.lower() == ".pdf" else 1
        return parsed, full_text, page_count

    def extract(self, path: Path) -> ExtractedDocument:
        invoice, full_text, page_count = self.extract_structured(
            path, InvoiceData, EXTRACTION_PROMPT,
        )
        return ExtractedDocument(
            source_path=str(path),
            extraction_method=f"{self.name}:{self._model}",
            page_count=page_count,
            tables=self._invoice_to_tables(invoice),
            full_text=full_text,
            invoice=invoice,
        )

    @staticmethod
    def _count_pages(path: Path) -> int:
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return len(pdf.pages)
        except Exception:
            return 1

    @staticmethod
    def _mime_for(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }[ext]

    @staticmethod
    def _invoice_to_tables(inv: InvoiceData) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        if inv.line_items:
            header = ["№", "Описание", "К-во", "ME", "Ед. цена", "Отстъпка %",
                     "Цена с отст.", "ДДС %", "ДДС", "Общо без ДДС"]
            rows = [header]
            for li in inv.line_items:
                rows.append([
                    str(li.number) if li.number is not None else "",
                    li.description or "",
                    _fmt(li.quantity),
                    li.unit or "",
                    _fmt(li.unit_price),
                    _fmt(li.discount_percent),
                    _fmt(li.price_after_discount),
                    _fmt(li.vat_percent),
                    _fmt(li.vat_amount),
                    _fmt(li.total_without_vat),
                ])
            tables.append(ExtractedTable(page=1, rows=rows))
        return tables


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".replace(".", ",")
