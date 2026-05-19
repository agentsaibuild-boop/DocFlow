from pathlib import Path

from docflow.extractors import Extractor
from docflow.extractors.gemini_extractor import GeminiExtractor
from docflow.extractors.mistral_extractor import MistralExtractor
from docflow.extractors.openrouter_extractor import OpenRouterExtractor
from docflow.quality import invoice_quality_coverage, invoice_quality_score
from docflow.schema import ExtractedDocument, is_useful_invoice
from docflow.text_llm import TextLLMExtractor
from docflow.text_parser import parse_text_to_invoice

QUALITY_OK_THRESHOLD = 0.7


_text_llm = TextLLMExtractor()

EXTRACTORS: list[Extractor] = [
    GeminiExtractor(model="gemini-3.1-flash-lite", name="gemini_3.1_flash_lite"),
    OpenRouterExtractor(
        model="qwen/qwen3-vl-235b-a22b-instruct",
        name="qwen_3_vl_235b",
    ),
    MistralExtractor(),
]

PROVIDER_ALIASES = {
    "gemini-3.1-flash-lite": "gemini_3.1_flash_lite",
    "qwen-3-vl-235b":        "qwen_3_vl_235b",
    "mistral":               "mistral_ocr",
}

# UI/CLI-visible provider list. 'auto' is intentionally NOT exposed — every
# extraction goes through an explicit, named provider so the resulting row's
# `extraction_method` is unambiguous.
AVAILABLE_PROVIDERS = list(PROVIDER_ALIASES.keys())


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
    """Raised when no registered extractor can handle the input file type."""


class ProviderError(Exception):
    """Provider/transport failure — separate from document-quality failure.

    A ProviderError means we couldn't get a response from the provider
    (quota exceeded, rate-limited, 5xx, auth missing, network down). It
    does NOT mean the document is bad or that extraction quality is low —
    the document was never given a fair attempt.

    Batch callers must surface ProviderError separately from extraction
    success/failure so quota events don't poison quality metrics.
    """

    def __init__(self, provider: str, kind: str, message: str, original: Exception | None = None):
        self.provider = provider
        self.kind = kind        # "quota" | "auth" | "network" | "other"
        self.original = original
        super().__init__(f"[{provider}/{kind}] {message}")


_QUOTA_HINTS = (
    "429", "503", "quota", "rate limit", "rate_limit",
    "resource_exhausted", "exhausted", "too many requests",
    "overloaded",
)
_AUTH_HINTS = (
    "401", "403", "unauthorized", "invalid api key", "authentication",
    "permission denied",
)
_NETWORK_HINTS = (
    "connection", "timeout", "timed out", "dns", "name resolution",
    "ssl", "tls handshake",
)


def _classify_provider_error(provider: str, exc: Exception) -> ProviderError:
    msg = str(exc).lower()
    if any(h in msg for h in _QUOTA_HINTS):
        kind = "quota"
    elif any(h in msg for h in _AUTH_HINTS):
        kind = "auth"
    elif any(h in msg for h in _NETWORK_HINTS):
        kind = "network"
    else:
        kind = "other"
    return ProviderError(provider, kind, str(exc), original=exc)


def extract(
    path: Path,
    provider: str = "auto",
    allow_fallback: bool | None = None,
) -> ExtractedDocument:
    """Extract one document.

    Behavior contract:
      • provider="auto"     → try the registered chain. allow_fallback defaults to True.
        On provider errors the next extractor is tried. If every extractor errored
        and none returned a useful result, raise ProviderError (last error's kind).
      • provider=<explicit> → use ONLY that extractor. allow_fallback defaults to False.
        On any error from that provider, raise ProviderError. Caller stays in control
        of which model produced the result. To opt into "use this first, but fall
        back to the rest on quota/transport", pass allow_fallback=True explicitly.

    Why this design: explicit provider selection is a deliberate user choice. Silently
    falling back would lie about which model produced the row. The override exists
    for batch scenarios where you'd rather accept any model than fail mid-run on a
    transient 503.

    Raises:
      ValueError         — unknown provider alias
      NoExtractorFound   — no registered extractor can handle this file type
      ProviderError      — provider/transport failure (quota/auth/network/other)
    """
    explicit = provider != "auto"
    if allow_fallback is None:
        allow_fallback = not explicit

    if explicit:
        if provider not in PROVIDER_ALIASES:
            raise ValueError(f"Unknown provider '{provider}'. Available: {AVAILABLE_PROVIDERS}")
        target_name = PROVIDER_ALIASES[provider]
        primary = [e for e in EXTRACTORS if e.name == target_name]
        if not primary:
            raise RuntimeError(f"Provider '{provider}' not registered in pipeline")
        if allow_fallback:
            rest = [e for e in EXTRACTORS if e.name != target_name]
            extractors = primary + rest
        else:
            extractors = primary
    else:
        extractors = EXTRACTORS

    last_provider_error: ProviderError | None = None
    tried: list[str] = []
    best_doc: ExtractedDocument | None = None
    best_score: float = -1.0

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
            if doc.invoice:
                _derive_missing_totals(doc.invoice)

            doc.quality_score = invoice_quality_score(doc.invoice)
            doc.quality_coverage = invoice_quality_coverage(doc.invoice)
            if doc.quality_score >= QUALITY_OK_THRESHOLD:
                return doc
            if doc.quality_score > best_score:
                best_doc = doc
                best_score = doc.quality_score
        except Exception as e:
            perr = _classify_provider_error(extractor.name, e)
            last_provider_error = perr
            print(f"  {extractor.name} failed ({perr.kind}: {type(e).__name__})", flush=True)
            if not allow_fallback:
                raise perr from e
            # otherwise: continue to next extractor

    if best_doc is not None:
        return best_doc

    if tried:
        if last_provider_error is not None:
            raise ProviderError(
                provider=",".join(tried),
                kind=last_provider_error.kind,
                message=f"All extractors failed for {path.name}: {', '.join(tried)}",
                original=last_provider_error.original,
            )
        # Tried something but nothing returned and no exception — shouldn't happen.
        raise ProviderError(
            provider=",".join(tried), kind="other",
            message=f"All extractors returned no result for {path.name}",
        )
    raise NoExtractorFound(f"No extractor can handle {path.name}")


def _derive_missing_totals(inv) -> None:
    """Cross-fill data across vat_breakdown, line_items, and top-level totals.

    Models often populate only one of these representations. We propagate so every
    output sheet (Фактури / Артикули / ДДС) shows consistent numbers. Anything
    we fill in here is recorded via inv.mark_derived(path) so downstream
    consumers can distinguish extracted vs. computed values.
    """
    _fill_line_item_arithmetic(inv)
    _propagate_single_vat_rate(inv)
    _fill_line_item_vat_amount(inv)
    _derive_vat_breakdown_from_lines(inv)

    vat_base = vat_tax = None
    if inv.vat_breakdown:
        vat_base = round(sum(v.base_amount for v in inv.vat_breakdown), 2)
        vat_tax = round(sum(v.vat_amount for v in inv.vat_breakdown), 2)

    items_net = None
    if inv.line_items:
        totals = [li.total_without_vat for li in inv.line_items if li.total_without_vat is not None]
        if totals:
            items_net = round(sum(totals), 2)

    if inv.net_amount is None:
        filled = vat_base if vat_base is not None else items_net
        if filled is not None:
            inv.net_amount = filled
            inv.mark_derived("net_amount")

    if inv.total_to_pay is None and inv.net_amount is not None and vat_tax is not None:
        inv.total_to_pay = round(inv.net_amount + vat_tax, 2)
        inv.mark_derived("total_to_pay")

    if inv.total_to_pay is None and inv.net_amount is not None and not inv.vat_breakdown:
        inv.total_to_pay = inv.net_amount
        inv.mark_derived("total_to_pay")

    if inv.net_amount is None and inv.total_to_pay is not None and vat_tax is not None:
        inv.net_amount = round(inv.total_to_pay - vat_tax, 2)
        inv.mark_derived("net_amount")


def _fill_line_item_arithmetic(inv) -> None:
    """Fill missing unit_price / quantity / total_without_vat / price_after_discount."""
    for i, li in enumerate(inv.line_items or []):
        disc = (li.discount_percent or 0) / 100.0

        if li.price_after_discount is None and li.unit_price is not None:
            li.price_after_discount = round(li.unit_price * (1 - disc), 4)
            inv.mark_derived(f"line_items[{i}].price_after_discount")

        if li.total_without_vat is None and li.quantity is not None and li.unit_price is not None:
            base = li.price_after_discount if li.price_after_discount is not None else li.unit_price * (1 - disc)
            li.total_without_vat = round(li.quantity * base, 2)
            inv.mark_derived(f"line_items[{i}].total_without_vat")

        if li.unit_price is None and li.total_without_vat is not None and li.quantity:
            denom = li.quantity * (1 - disc) if (1 - disc) else li.quantity
            if denom:
                li.unit_price = round(li.total_without_vat / denom, 4)
                inv.mark_derived(f"line_items[{i}].unit_price")

        if li.quantity is None and li.total_without_vat is not None and li.unit_price:
            denom = li.unit_price * (1 - disc) if (1 - disc) else li.unit_price
            if denom:
                li.quantity = round(li.total_without_vat / denom, 4)
                inv.mark_derived(f"line_items[{i}].quantity")


def _propagate_single_vat_rate(inv) -> None:
    """If invoice has a single VAT rate, assign it to line items that lack vat_percent."""
    if not inv.line_items or not inv.vat_breakdown:
        return
    rates = {v.rate_percent for v in inv.vat_breakdown}
    if len(rates) != 1:
        return
    sole_rate = next(iter(rates))
    for i, li in enumerate(inv.line_items):
        if li.vat_percent is None:
            li.vat_percent = sole_rate
            inv.mark_derived(f"line_items[{i}].vat_percent")


def _fill_line_item_vat_amount(inv) -> None:
    """Compute line_item.vat_amount from total_without_vat × vat_percent when missing."""
    for i, li in enumerate(inv.line_items or []):
        if li.vat_amount is None and li.total_without_vat is not None and li.vat_percent is not None:
            li.vat_amount = round(li.total_without_vat * li.vat_percent / 100.0, 2)
            inv.mark_derived(f"line_items[{i}].vat_amount")


def _derive_vat_breakdown_from_lines(inv) -> None:
    """If vat_breakdown is empty but line_items carry vat_percent + totals, group and sum."""
    if inv.vat_breakdown:
        return
    if not inv.line_items:
        return
    from docflow.schema import VatBreakdown
    buckets: dict[float, dict[str, float]] = {}
    for li in inv.line_items:
        if li.vat_percent is None or li.total_without_vat is None:
            continue
        b = buckets.setdefault(li.vat_percent, {"base": 0.0, "vat": 0.0})
        b["base"] += li.total_without_vat
        b["vat"] += li.vat_amount if li.vat_amount is not None else li.total_without_vat * li.vat_percent / 100.0
    if not buckets:
        return
    inv.vat_breakdown = [
        VatBreakdown(rate_percent=rate, base_amount=round(b["base"], 2), vat_amount=round(b["vat"], 2))
        for rate, b in sorted(buckets.items())
    ]
    inv.mark_derived("vat_breakdown")


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
