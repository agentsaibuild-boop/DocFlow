"""Mistral OCR extractor — specialized OCR API returning structured markdown.

Mistral OCR doesn't return typed invoice fields directly. It returns clean
markdown with preserved tables/structure. This is then passed to text_llm
post-processor for typed InvoiceData extraction.

Pricing: ~$1/1000 pages (mid-tier between Azure DI and Gemini).
"""

import base64
import os
from pathlib import Path

import httpx

from docflow.schema import ExtractedDocument

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}

API_URL = "https://api.mistral.ai/v1/ocr"
DEFAULT_MODEL = "mistral-ocr-latest"


class MistralExtractor:
    name = "mistral_ocr"
    api_key_env = "MISTRAL_API_KEY"

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(120.0)

    @property
    def _model(self) -> str:
        return os.environ.get("MISTRAL_OCR_MODEL", DEFAULT_MODEL)

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return False
        return bool(os.environ.get(self.api_key_env))

    def extract(self, path: Path) -> ExtractedDocument:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set. Get key at https://console.mistral.ai/api-keys"
            )

        ext = path.suffix.lower()
        b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        if ext == ".pdf":
            document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"}
        else:
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "tif": "image/tiff", "tiff": "image/tiff",
                    "bmp": "image/bmp"}[ext.lstrip(".")]
            document = {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"}

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": self._model, "document": document},
            )
        response.raise_for_status()
        data = response.json()

        pages = data.get("pages", [])
        markdown_parts = []
        for page in pages:
            md = page.get("markdown", "")
            if md:
                markdown_parts.append(md)
        full_text = "\n\n".join(markdown_parts)
        page_count = len(pages) if pages else 1

        return ExtractedDocument(
            source_path=str(path),
            extraction_method=f"{self.name}:{self._model}",
            page_count=page_count,
            tables=[],
            full_text=full_text,
            invoice=None,  # left empty — text_llm post-processor fills it
        )
