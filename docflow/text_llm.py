"""LLM-based extractor that takes already-extracted text and returns typed InvoiceData.

Used as post-processor when other extractors return text/tables but no typed invoice data.
Cheaper than vision LLM: ~1500 input tokens vs 15000 for an image.
"""

import os
import time
from typing import Final

from docflow.extraction_prompt import LINE_ITEMS_SECTION
from docflow.schema import InvoiceData

DEFAULT_MODEL: Final[str] = "gemini-2.5-flash-lite"
MIN_TEXT_LENGTH: Final[int] = 50

EXTRACTION_PROMPT_TEXT = """\
You are extracting structured data from a Bulgarian invoice or acceptance protocol.

The text below was extracted from the document by an OCR or PDF parser. Some fields may be misread, mis-aligned, or in awkward order. Reconstruct the structured data as best you can.

═══ CRITICAL: BULGARIAN NUMBER FORMAT ═══
  • COMMA (,) is the DECIMAL separator: "1,99" → 1.99
  • DOT (.) is the THOUSANDS separator: "1.000" → 1000
  • "1.000,00" → 1000.00 (one thousand, NEVER 1.0)
  • "36.265,00" → 36265.00

═══ FIELD RULES ═══
  • Identify supplier (Изпълнител) vs customer (Получател) by labels.
  • ЕИК = 9, 10 (ЕГН), or 13 (БУЛСТАТ) digits, no prefix.
  • ИН по ДДС = OPTIONAL, format BG + ЕИК digits. Return null if not present.
  • IBAN = BG + 2 digits + 4 letters + 14 alphanumeric, exactly 22 chars.
  • Dates: parse from DD.MM.YYYY → YYYY-MM-DD.
  • Return null for fields not present. DO NOT hallucinate.
""" + LINE_ITEMS_SECTION + """
═══ TEXT FROM DOCUMENT ═══
"""


class TextLLMExtractor:
    """Wraps Gemini text-mode for parsing extracted text into InvoiceData."""

    name = "gemini_text"
    api_key_env = "GEMINI_API_KEY"

    def __init__(self) -> None:
        self._client = None

    @property
    def _model(self) -> str:
        return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    def is_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def _get_client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def extract_from_text(self, text: str) -> InvoiceData | None:
        """Returns InvoiceData parsed from raw text, or None if input too short or LLM fails."""
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return None

        from google.genai import errors, types

        client = self._get_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InvoiceData,
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
                    contents=[EXTRACTION_PROMPT_TEXT + text[:20000]],
                    config=config,
                )
                break
            except errors.ServerError as e:
                last_err = e
                if attempt <= len(delays):
                    time.sleep(delays[attempt - 1])
            except errors.ClientError as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    return None
                raise

        if response is None:
            return None

        invoice: InvoiceData | None = getattr(response, "parsed", None)
        return invoice
