"""OpenRouter extractor — unified gateway to all major LLMs.

Single API key gives access to: Claude, Gemini, GPT, DeepSeek, Qwen, Llama, Mistral, etc.
User picks model via OPENROUTER_MODEL env var (e.g. "anthropic/claude-haiku-4-5",
"google/gemini-2.5-flash-lite", "deepseek/deepseek-chat", "qwen/qwen-2.5-vl-72b-instruct").

Uses OpenAI-compatible chat completions endpoint with tool-use for structured output.
For PDFs, renders pages to PNG via pypdfium2 (most OpenRouter models don't accept PDF directly).
"""

import base64
import io
import json
import os
import threading
from pathlib import Path

import httpx

from docflow.schema import ExtractedDocument, ExtractedTable, InvoiceData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
PDF_RENDER_DPI = 150
MAX_PAGES = 5

_PDFIUM_LOCK = threading.Lock()

EXTRACTION_PROMPT = """\
Extract structured data from this Bulgarian invoice. Call the extract_invoice tool.

CRITICAL Bulgarian number format:
- COMMA is decimal separator: "1,99" → 1.99
- DOT is thousands separator: "1.000" → 1000 (NEVER 1.0!)
- "1.000,00" → 1000.00

Field rules:
- Supplier (Изпълнител) vs Customer (Получател) — use labels.
- ЕИК = 9/10/13 digits, no prefix.
- VAT (ИН по ДДС) = BG+digits, OPTIONAL.
- IBAN = BG+22 chars, OPTIONAL.
- Dates: YYYY-MM-DD format.
- Return null for missing fields. Don't hallucinate.

For non-Bulgarian invoices (US, EU): adapt — use Tax ID for eik, USD/EUR currency, etc.
"""


class OpenRouterExtractor:
    api_key_env = "OPENROUTER_API_KEY"

    def __init__(self, model: str | None = None, name: str | None = None) -> None:
        self._timeout = httpx.Timeout(180.0)
        self._explicit_model = model
        self.name = name or "openrouter"

    @property
    def _model(self) -> str:
        if self._explicit_model:
            return self._explicit_model
        return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return False
        return bool(os.environ.get(self.api_key_env))

    def extract(self, path: Path) -> ExtractedDocument:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set. Get key at https://openrouter.ai/keys"
            )

        ext = path.suffix.lower()
        if ext == ".pdf":
            with _PDFIUM_LOCK:
                image_blocks, page_count = self._pdf_to_image_blocks(path)
        else:
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "gif": "image/gif"}[ext.lstrip(".")]
            b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
            image_blocks = [{"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}}]
            page_count = 1

        schema = InvoiceData.model_json_schema()
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_invoice",
                "description": "Submit extracted invoice data.",
                "parameters": schema,
            },
        }]

        content = [{"type": "text", "text": EXTRACTION_PROMPT}, *image_blocks]
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "extract_invoice"}},
            "temperature": 0.0,
            "max_tokens": 4096,
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/agentsaibuild-boop/DocFlow",
                    "X-Title": "DocFlow",
                },
                json=payload,
            )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raise RuntimeError(
                f"Model '{self._model}' did not call extract_invoice tool. "
                f"Response: {message.get('content', '')[:200]}"
            )
        args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
        invoice_dict = json.loads(args_str)
        invoice = InvoiceData.model_validate(invoice_dict)

        tables = self._invoice_to_tables(invoice)
        model_short = self._model.split("/")[-1]
        return ExtractedDocument(
            source_path=str(path),
            extraction_method=f"{self.name}:{model_short}",
            page_count=page_count,
            tables=tables,
            full_text="",
            invoice=invoice,
        )

    @staticmethod
    def _pdf_to_image_blocks(path: Path) -> tuple[list[dict], int]:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        total_pages = len(pdf)
        scale = PDF_RENDER_DPI / 72
        blocks: list[dict] = []
        for i, page in enumerate(pdf):
            if i >= MAX_PAGES:
                break
            pil_image = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG", optimize=True)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
            page.close()
        pdf.close()
        return blocks, total_pages

    @staticmethod
    def _invoice_to_tables(inv: InvoiceData) -> list[ExtractedTable]:
        if not inv.line_items:
            return []
        header = ["№", "Описание", "К-во", "ME", "Ед. цена", "Отстъпка %",
                  "Цена с отст.", "ДДС %", "ДДС", "Общо без ДДС"]
        rows = [header]
        for li in inv.line_items:
            rows.append([
                str(li.number) if li.number else "",
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
        return [ExtractedTable(page=1, rows=rows)]
