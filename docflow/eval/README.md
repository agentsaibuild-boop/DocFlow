# DocFlow Evaluation Framework

Measures real extraction accuracy against human-verified invoice truth,
not against internal validators.

## Quick start

1. Put invoices + truth sidecars in a directory:

   ```
   eval_dataset/
     invoice_001.pdf
     invoice_001.truth.json
     invoice_002.pdf
     invoice_002.truth.json
     ...
   ```

   Invoices without a `.truth.json` sibling are silently ignored.

2. Run the evaluation:

   ```bash
   python -m docflow.eval run \
       --dataset eval_dataset \
       --providers qwen-3-vl-235b,gemini-3.1-flash-lite,claude \
       --output eval_output/
   ```

3. Read `eval_output/eval_report.md`. Also produced: `.csv`, `.xlsx`, `.json`.

## Truth file schema

Each sidecar is a JSON file with any subset of these keys. Omit a key →
that field is **not graded** (status `not_in_truth`), not "expected null".

```json
{
  "supplier_name": "Acme ООД",
  "supplier_eik": "123456789",
  "supplier_vat": "BG123456789",
  "iban": "BG80BNBG96611020345678",
  "invoice_number": "2026-0001",
  "issue_date": "2026-01-15",
  "net_amount": 100.0,
  "vat_amount": 20.0,
  "total_to_pay": 120.0,
  "currency": "BGN",
  "payment_method": "По банков път",
  "line_items": [
    {"description": "Артикул A", "quantity": 1, "unit_price": 100.0, "total_without_vat": 100.0}
  ],
  "notes": "Free-text human comment, not graded"
}
```

Numeric fields are matched within ±0.01 tolerance. String fields match either
exactly or after whitespace+case normalization.

## What gets reported

**Per-field statuses:**

| Status | Meaning |
|---|---|
| `exact_match` | Strings equal byte-for-byte; numbers exactly equal |
| `normalized_match` | Equal after whitespace+case fold; numbers within tolerance |
| `mismatch` | Both sides present, different value |
| `missing` | Truth has a value, extractor returned None |
| `not_in_truth` | Truth file omits this field; not graded |

**Aggregate metrics** (per provider):

- `invoice_accuracy` — % of invoices where every graded field matched.
- `critical_field_accuracy` — % of (invoice × critical field) cells matched.
  Critical fields: `supplier_eik`, `supplier_vat`, `iban`, `total_to_pay`, `net_amount`.
- `hallucination_rate` — `mismatch / total_graded`.
- `wrong_but_confident_rate` — % of invoices where `quality_score ≥ 0.7` but
  not all graded fields matched. **This is the most important warning signal.**
- `avg_quality_score` / `avg_quality_coverage` / `avg_processing_time_s`.
- `provider_failure_count` — quota / auth / network errors (separate from
  document-quality failures).

## Regression mode

Compare two runs to see what changed:

```bash
python -m docflow.eval regression \
    eval_output/previous.json \
    eval_output/current.json \
    --output eval_output/diff.md
```

Surfaces per-provider top-line deltas (Δ invoice accuracy, Δ critical accuracy,
etc.) and per-file `quality_score` changes ≥ ±0.05.

## Constraints

- The evaluator never modifies the codebase, the schema, or extractor
  behavior. It only **reads** what extraction produces and grades it.
- `allow_fallback` defaults to **False** for benchmark integrity: when
  measuring provider X, a silent fallback to provider Y would taint X's row.
- Critical-field accuracy and wrong-but-confident rate are the two numbers
  to watch. They answer: "is our high-confidence output actually correct?"
