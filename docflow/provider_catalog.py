"""User-facing catalog of extraction providers.

We show the **real model names** — users care which AI they are using and
typically bring their own API keys. The catalog wraps each model with a
short human-language description, one badge, and practical context (best for,
trade-offs, expected cost).

Do not paraphrase model names ("Препоръчителен режим" etc.) — that hides the
real choice. Use the model name as it appears in vendor documentation.
"""

from dataclasses import dataclass, field as dc_field


# Single, default-highlighted recommendation.
RECOMMENDED_PROVIDER = "gemini-3.1-flash-lite"


@dataclass
class ProviderProfile:
    provider: str           # raw alias (matches PROVIDER_ALIASES key)
    display: str            # the actual model name as the vendor presents it
    badge: str              # one-word category + emoji (kept short for chip rendering)
    headline: str           # one sentence: when to pick this
    description: str        # 1-2 sentences in plain BG, explaining the choice
    best_for: list[str]     # situations where it shines
    tradeoff: str           # what to know before choosing
    cost: str               # practical cost note, mentions API-key ownership


# Display order: recommended first, most-accurate second, specialised third.
PROFILES: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        provider="gemini-3.1-flash-lite",
        display="Gemini 3.1 Flash Lite",
        badge="⭐ Препоръчан",
        headline="Бърз и точен за повечето всекидневни фактури.",
        description=(
            "Стандартният избор. Работи бързо и се справя добре с типични "
            "български фактури. В нашия 10-файлов тест беше с най-добър баланс "
            "между скорост и точност."
        ),
        best_for=[
            "Всекидневна обработка",
            "Стандартни фактури",
            "Когато искаш резултат бързо",
        ],
        tradeoff="Понякога услугата е претоварена и една фактура трябва да се опита отново.",
        cost="Безплатно при малки обеми. Изисква собствен Google Gemini API ключ.",
    ),
    ProviderProfile(
        provider="qwen-3-vl-235b",
        display="Qwen 3 VL 235B",
        badge="🎯 Най-точен",
        headline="Най-точен в нашия тест, но по-бавен.",
        description=(
            "Голям модел с отлично разпознаване на български текст — "
            "включително скенирани документи, дребен шрифт и нестандартни "
            "формати. В нашия тест разпозна доставчика правилно във всеки случай."
        ),
        best_for=[
            "Скенирани фактури",
            "Дребен или неясен шрифт",
            "Нестандартни формати",
        ],
        tradeoff="Около 3 пъти по-бавен от Gemini Flash Lite.",
        cost="Заплаща се с малки суми (per-token). Изисква OpenRouter API ключ.",
    ),
    ProviderProfile(
        provider="mistral",
        display="Mistral OCR",
        badge="📄 За сложни таблици",
        headline="Силен на документи с гъсти таблици.",
        description=(
            "Подходящ когато фактурата има дълги списъци артикули или "
            "множество вложени таблици — например складови или дистрибуторски "
            "документи."
        ),
        best_for=[
            "Гъсти таблици",
            "Дълги списъци артикули",
            "Складови и логистични фактури",
        ],
        tradeoff="Все още не е тестван на широк набор от български фактури.",
        cost="Заплаща се с малки суми. Изисква собствен Mistral API ключ.",
    ),
)


def get_profile(provider: str) -> ProviderProfile | None:
    for p in PROFILES:
        if p.provider == provider:
            return p
    return None


def display_for(provider: str) -> str:
    p = get_profile(provider)
    return p.display if p else provider


def options_in_display_order(available: list[str]) -> list[str]:
    avail = set(available)
    ordered = [p.provider for p in PROFILES if p.provider in avail]
    for prov in available:
        if prov not in ordered:
            ordered.append(prov)
    return ordered


def measured_summary(provider: str) -> str | None:
    """One calm sentence about real-world performance. No raw numbers.

    Returns None for providers without a recorded benchmark.
    """
    from docflow.eval.benchmarks import get_benchmark

    b = get_benchmark(provider)
    if b is None:
        return None

    parts: list[str] = []
    if b.success_rate >= 1.0:
        parts.append(f"в наш тест обработи всичките {b.invoice_count} фактури")
    else:
        ok = int(round(b.success_rate * b.invoice_count))
        parts.append(f"в наш тест обработи {ok} от {b.invoice_count} фактури")
    if b.supplier_accuracy >= 1.0:
        parts.append("разпозна доставчика правилно във всеки случай")
    else:
        ok = int(round(b.supplier_accuracy * b.invoice_count))
        parts.append(f"разпозна доставчика правилно в {ok} от {b.invoice_count} случая")
    if b.avg_latency_s <= 12:
        parts.append("обикновено отговаря под 10 секунди на фактура")
    elif b.avg_latency_s <= 20:
        parts.append("отнема около 10–20 секунди на фактура")
    else:
        parts.append(f"отнема около {int(b.avg_latency_s)} секунди на фактура")
    return "В реални условия: " + "; ".join(parts) + "."
