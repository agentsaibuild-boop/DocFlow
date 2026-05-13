from pathlib import Path

from docflow.extractors import Extractor
from docflow.extractors.azure_extractor import AzureExtractor
from docflow.extractors.claude_extractor import ClaudeExtractor
from docflow.extractors.gemini_extractor import GeminiExtractor
from docflow.extractors.mistral_extractor import MistralExtractor
from docflow.extractors.pdfplumber_extractor import PdfplumberExtractor
from docflow.schema import ExtractedDocument, is_useful_invoice
from docflow.text_llm import TextLLMExtractor
from docflow.text_parser import parse_text_to_invoice


_text_llm = TextLLMExtractor()

EXTRACTORS: list[Extractor] = [
    AzureExtractor(),       # 1st: invoice-specialized model, per-field confidence
    PdfplumberExtractor(),  # 2nd: born-digital PDF text → text_llm produces typed data
    MistralExtractor(),     # 3rd: Mistral OCR markdown → text_llm produces typed data
    ClaudeExtractor(),      # 4th: Claude vision (fallback for non-standard docs)
    GeminiExtractor(),      # 5th: Gemini default (model from GEMINI_MODEL env)
    GeminiExtractor(model="gemini-3.1-flash-lite", name="gemini_3.1_flash_lite"),
    GeminiExtractor(model="gemini-2.5-pro", name="gemini_2.5_pro"),
]

PROVIDER_ALIASES = {
    "azure": "azure_di_invoice",
    "pdfplumber": "pdfplumber",
    "mistral": "mistral_ocr",
    "claude": "claude",
    "gemini": "gemini",
    "gemini-3.1-flash-lite": "gemini_3.1_flash_lite",
    "gemini-2.5-pro": "gemini_2.5_pro",
}

AVAILABLE_PROVIDERS = ["auto"] + list(PROVIDER_ALIASES.keys())


def list_providers() -> list[tuple[str, bool, str]]:
    """Returns (alias, available, reason). available=True means API key/endpoint configured."""
    import os

    rows = []
    for alias, name in PROVIDER_ALIASES.items():
        extractor = next((e for e in EXTRACTORS if e.name == name), None)
        if extractor is None:
            rows.append((alias, False, "not registered"))
            continue
        if hasattr(extractor, "api_key_env"):
            if os.environ.get(extractor.api_key_env):
                rows.append((alias, True, "API key configured"))
            else:
                rows.append((alias, False, f"set {extractor.api_key_env} in .env"))
        elif hasattr(extractor, "endpoint_env"):
            ep = os.environ.get(extractor.endpoint_env)
            key = os.environ.get(extractor.key_env)
            if ep and key:
                rows.append((alias, True, "endpoint + key configured"))
            else:
                missing = []
                if not ep: missing.append(extractor.endpoint_env)
                if not key: missing.append(extractor.key_env)
                rows.append((alias, False, f"set {', '.join(missing)} in .env"))
        else:
            rows.append((alias, True, "no auth required"))
    return rows


class NoExtractorFound(Exception):
    pass


def extract(path: Path, provider: str = "auto") -> ExtractedDocument:
    if provider != "auto":
        if provider not in PROVIDER_ALIASES:
            raise ValueError(f"Unknown provider '{provider}'. Available: {AVAILABLE_PROVIDERS}")
        target_name = PROVIDER_ALIASES[provider]
        extractors = [e for e in EXTRACTORS if e.name == target_name]
        if not extractors:
            raise RuntimeError(f"Provider '{provider}' not registered in pipeline")
    else:
        extractors = EXTRACTORS

    last_error: Exception | None = None
    tried: list[str] = []
    for extractor in extractors:
        if not extractor.can_handle(path):
            continue
        tried.append(extractor.name)
        try:
            doc = extractor.extract(path)
            table_text = "\n".join(
                " ".join(cell for cell in row)
                for table in doc.tables for row in table.rows
            )
            full_text = doc.full_text + "\n" + table_text

            if not is_useful_invoice(doc.invoice):
                if _text_llm.is_available():
                    llm_invoice = _text_llm.extract_from_text(full_text)
                    if is_useful_invoice(llm_invoice):
                        doc.invoice = llm_invoice
                        doc.extraction_method = f"{extractor.name}+gemini_text"
                if not is_useful_invoice(doc.invoice):
                    parsed = parse_text_to_invoice(full_text)
                    if parsed is not None:
                        doc.invoice = parsed
                    elif doc.invoice is not None and not is_useful_invoice(doc.invoice):
                        doc.invoice = None

            if doc.invoice and doc.invoice.iban:
                _cross_check_iban(doc.invoice, full_text)
            return doc
        except Exception as e:
            last_error = e
            print(f"  {extractor.name} failed ({type(e).__name__}), falling back", flush=True)
    if tried:
        raise RuntimeError(
            f"All extractors failed for {path.name}: {', '.join(tried)}. Last error: {last_error}"
        ) from last_error
    raise NoExtractorFound(f"No extractor can handle {path.name}")


def _cross_check_iban(invoice, text: str) -> None:
    """If extracted IBAN fails checksum, try to recover a valid one from raw text."""
    from docflow.text_parser import IBAN_RE
    from docflow.validators import _iban_checksum_ok

    iban = invoice.iban.replace(" ", "").upper() if invoice.iban else ""
    if len(iban) == 22 and _iban_checksum_ok(iban):
        return

    for match in IBAN_RE.finditer(text.replace(" ", "")):
        candidate = match.group(1)
        if _iban_checksum_ok(candidate):
            invoice.iban = candidate
            return
