"""Claude vision extractor for Bulgarian invoices.

Uses Anthropic Claude with tool-use to enforce InvoiceData schema.
Supports PDF (native), JPG, PNG, GIF, WEBP via vision input.
"""

import base64
import os
import time
from pathlib import Path

from docflow.schema import ExtractedDocument, ExtractedTable, InvoiceData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}

DEFAULT_MODEL = "claude-sonnet-4-6"

EXTRACTION_PROMPT = """\
You are an expert invoice data extractor specialized in Bulgarian (български) invoices and acceptance protocols (приемно-предавателни протоколи).

Call the extract_invoice tool with ALL structured data from this document.

═══ CRITICAL: BULGARIAN NUMBER FORMAT ═══
In Bulgaria, conventions are OPPOSITE of English:
  • COMMA (,) is the DECIMAL separator
  • DOT (.) is the THOUSANDS separator

Examples:
  • "1.999,00" → 1999.00
  • "1.000,00" → 1000.00 (ONE THOUSAND, NOT 1.0)
  • "36.265,00" → 36265.00
  • "799,60" → 799.60

⚠ NEVER interpret the dot as a decimal separator. "1.000" is ALWAYS 1000, never 1.0.

═══ FIELD RULES ═══
- Identify supplier (Изпълнител) vs customer (Получател) by labels.
- ЕИК = 9, 10 (ЕГН), or 13 (БУЛСТАТ) digits, no prefix.
- ИН по ДДС = OPTIONAL field. Only present if VAT-registered. Format BG + digits. Return null if no VAT line — do NOT fabricate.
- IBAN = BG + 2 digits + 4 letters + 14 alphanumeric, exactly 22 chars.
- VAT breakdown rates in Bulgaria: typically 9% or 20%.
- Capture every row in the goods/services table.
- Dates: parse from DD.MM.YYYY → YYYY-MM-DD.
- Return null for fields not present. DO NOT hallucinate.
- Currency defaults to BGN unless explicitly stated.
"""


class ClaudeExtractor:
    name = "claude"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self) -> None:
        self._client = None

    @property
    def _model(self) -> str:
        return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)

    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} not set. Get key at https://console.anthropic.com/settings/keys"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return False
        return bool(os.environ.get(self.api_key_env))

    def extract(self, path: Path) -> ExtractedDocument:
        import anthropic

        client = self._get_client()
        ext = path.suffix.lower()
        data_b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        if ext == ".pdf":
            content_block = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data_b64},
            }
        else:
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "gif": "image/gif", "webp": "image/webp"}[ext.lstrip(".")]
            content_block = {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data_b64},
            }

        schema = InvoiceData.model_json_schema()
        tools = [{
            "name": "extract_invoice",
            "description": "Submit the structured invoice data extracted from the document.",
            "input_schema": schema,
        }]

        delays_str = os.environ.get("CLAUDE_RETRY_DELAYS", "")
        delays = [int(d) for d in delays_str.split(",") if d.strip()] if delays_str else []
        last_err: Exception | None = None
        response = None

        for attempt in range(1, len(delays) + 2):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    tools=tools,
                    tool_choice={"type": "tool", "name": "extract_invoice"},
                    messages=[{
                        "role": "user",
                        "content": [content_block, {"type": "text", "text": EXTRACTION_PROMPT}],
                    }],
                )
                break
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    raise RuntimeError(
                        f"Claude rate limit hit for model '{self._model}'. "
                        f"Check usage at https://console.anthropic.com/settings/usage"
                    ) from e
                if attempt <= len(delays):
                    time.sleep(delays[attempt - 1])

        if response is None:
            raise RuntimeError(f"Claude unavailable after {len(delays) + 1} attempt(s)") from last_err

        tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError("Claude did not return tool_use block")

        invoice = InvoiceData.model_validate(tool_use.input)
        tables = self._invoice_to_tables(invoice)
        page_count = self._count_pages(path) if ext == ".pdf" else 1

        return ExtractedDocument(
            source_path=str(path),
            extraction_method=f"{self.name}:{self._model}",
            page_count=page_count,
            tables=tables,
            full_text="",
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
    def _invoice_to_tables(inv: InvoiceData) -> list[ExtractedTable]:
        tables = []
        if inv.line_items:
            header = ["№", "Описание", "К-во", "ME", "Ед. цена", "Отстъпка %",
                     "Цена с отст.", "ДДС %", "ДДС", "Общо без ДДС"]
            rows = [header]
            for li in inv.line_items:
                rows.append([
                    str(li.number) if li.number is not None else "",
                    li.description or "",
                    str(li.quantity) if li.quantity is not None else "",
                    li.unit or "",
                    str(li.unit_price) if li.unit_price is not None else "",
                    str(li.discount_percent) if li.discount_percent is not None else "",
                    str(li.price_after_discount) if li.price_after_discount is not None else "",
                    str(li.vat_percent) if li.vat_percent is not None else "",
                    str(li.vat_amount) if li.vat_amount is not None else "",
                    str(li.total_without_vat) if li.total_without_vat is not None else "",
                ])
            tables.append(ExtractedTable(page=1, rows=rows))
        return tables
