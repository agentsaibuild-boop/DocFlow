"""Unified status engine.

A single function decides the per-row status used by both the Streamlit UI and
the batch xlsx writer. Previously the UI grew column-aware completeness logic
while the batch path kept the older validation-only OK/WARN/ERR strings — they
disagreed on what "good" meant. This module is the contract.
"""

from dataclasses import dataclass, field

from docflow.columns import get_row_value


@dataclass
class StatusResult:
    code: str
    label: str
    empty_columns: list[str] = field(default_factory=list)
    notes: str = ""


def compute_status(
    doc,
    validation_findings,
    required_column_keys: list[str] | None = None,
    extraction_error: str | None = None,
) -> StatusResult:
    """Decide a row status from extraction outcome + validation + optional column expectations.

    Precedence (first match wins):
      1. extraction_error → ERROR
      2. invoice missing or supplier.name empty → NO_DATA
      3. any validation finding with level=="error" → VALIDATION_ERROR
      4. required_column_keys provided and some empty → INCOMPLETE
      5. otherwise → OK
    """
    if extraction_error:
        return StatusResult(code="ERROR", label="❌ ERROR")

    inv = doc.invoice if doc is not None else None
    if inv is None or not inv.supplier or not inv.supplier.name:
        return StatusResult(code="NO_DATA", label="⚠️ Без данни")

    if any(getattr(f, "level", None) == "error" for f in (validation_findings or [])):
        return StatusResult(code="VALIDATION_ERROR", label="⚠️ Има грешки")

    if required_column_keys:
        empty: list[str] = []
        for key in required_column_keys:
            value = get_row_value(key, doc)
            if value is None or value == "":
                empty.append(key)
        if empty:
            return StatusResult(
                code="INCOMPLETE",
                label=f"⚠️ Непълни ({len(empty)})",
                empty_columns=empty,
                notes="Празни: " + ", ".join(empty),
            )

    return StatusResult(code="OK", label="✅ OK")
