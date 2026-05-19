"""Recorded provider benchmarks — facts, not marketing.

Each entry is a real measurement from a documented dataset run. If you add a
provider here, write down which invoices it was measured on and when. Don't
estimate or interpolate; if a provider hasn't been benchmarked, leave it out
of BENCHMARKS so the UI shows "не е тестван" honestly.

When you re-benchmark, append a new entry rather than mutating an old one —
the history matters for regression analysis.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderBenchmark:
    provider: str               # provider alias as exposed in PROVIDER_ALIASES
    dataset: str                # human description of the source files
    invoice_count: int
    avg_quality_score: float    # confidence score on the dataset, [0, 1]
    avg_latency_s: float        # average wall-clock seconds per invoice
    supplier_accuracy: float    # 0-1, fraction with correctly identified supplier
    success_rate: float         # 0-1, fraction that produced a usable result
    notes: str                  # one-liner takeaway
    measured_at: str            # YYYY-MM-DD


# Real measurements from the 10-file А1 telecom invoice benchmark (May 2026).
# Providers not listed here have NOT been benchmarked on a documented dataset
# and the UI displays them as "не е тестван" — informed-consent over guesses.
BENCHMARKS: tuple[ProviderBenchmark, ...] = (
    ProviderBenchmark(
        provider="qwen-3-vl-235b",
        dataset="А1 telecom invoices (BG, 10 files)",
        invoice_count=10,
        avg_quality_score=0.99,
        avg_latency_s=25.0,
        supplier_accuracy=1.00,
        success_rate=1.00,
        notes="Най-добра Cyrillic OCR точност. Разпознава типографски кавички. Най-бавен.",
        measured_at="2026-05-17",
    ),
    ProviderBenchmark(
        provider="gemini-3.1-flash-lite",
        dataset="А1 telecom invoices (BG, 10 files)",
        invoice_count=10,
        avg_quality_score=0.84,
        avg_latency_s=7.0,
        supplier_accuracy=1.00,
        success_rate=0.90,
        notes="Най-добър баланс скорост/качество за БГ фактури. Подвластен на quota.",
        measured_at="2026-05-17",
    ),
)


def get_benchmark(provider: str) -> ProviderBenchmark | None:
    """Return the recorded benchmark for `provider`, or None if not measured."""
    for b in BENCHMARKS:
        if b.provider == provider:
            return b
    return None


def benchmarks_as_rows() -> list[dict]:
    """Flat dict rows for tabular rendering (st.dataframe / DataFrame)."""
    return [
        {
            "Provider": b.provider,
            "Quality score": b.avg_quality_score,
            "Supplier acc": b.supplier_accuracy,
            "Success rate": b.success_rate,
            "Latency (s)": b.avg_latency_s,
            "Sample": b.invoice_count,
            "Dataset": b.dataset,
            "Бележка": b.notes,
            "Измерено": b.measured_at,
        }
        for b in BENCHMARKS
    ]
