from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr


class ExtractedTable(BaseModel):
    page: int
    rows: list[list[str]] = Field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None


class Party(BaseModel):
    name: str | None = None
    address: str | None = None
    eik: str | None = Field(default=None, description="ЕИК/ПИК — 9, 10, or 13 digit Bulgarian identifier (legal entity / sole proprietor / БУЛСТАТ)")
    vat_number: str | None = Field(default=None, description="ИН по ДДС — VAT number (BG + digits). Optional, only present if VAT-registered")
    mol: str | None = Field(default=None, description="МОЛ — material responsible person")
    phone: str | None = None
    email: str | None = None


class LineItem(BaseModel):
    number: int | None = None
    description: str | None = None
    quantity: float | None = None
    unit: str | None = Field(default=None, description="Unit of measure: бр, кг, м2, дм, etc.")
    unit_price: float | None = Field(default=None, description="Unit price without VAT")
    discount_percent: float | None = None
    price_after_discount: float | None = None
    vat_percent: float | None = None
    vat_amount: float | None = None
    total_without_vat: float | None = Field(default=None, description="Line total without VAT")


class VatBreakdown(BaseModel):
    rate_percent: float
    base_amount: float
    vat_amount: float


class InvoiceData(BaseModel):
    invoice_number: str | None = None
    issue_date: str | None = Field(default=None, description="Дата на издаване, ISO format if possible")
    payment_due_date: str | None = Field(default=None, description="Да се плати до")
    delivery_date: str | None = Field(default=None, description="Дата на предоставяне")
    supplier: Party | None = Field(default_factory=Party, description="Изпълнител")
    customer: Party | None = Field(default_factory=Party, description="Получател")
    line_items: list[LineItem] | None = Field(default_factory=list)
    subtotal: float | None = Field(default=None, description="Сума без отстъпка")
    discount: float | None = Field(default=None, description="Отстъпка (positive value)")
    net_amount: float | None = Field(default=None, description="Обща нетна сума")
    vat_breakdown: list[VatBreakdown] | None = Field(default_factory=list)
    total_to_pay: float | None = Field(default=None, description="За плащане")
    paid: float | None = Field(default=None, description="Платени")
    remaining: float | None = Field(default=None, description="Остава за плащане")
    payment_method: str | None = None
    iban: str | None = None
    bank: str | None = None
    bic: str | None = None
    currency: str | None = Field(
        default=None,
        description="Detected currency code (EUR/BGN/USD/...). None = not extracted; display layer falls back to 'EUR'.",
    )
    notes: str | None = None
    # Internal provenance metadata. Not part of the extraction contract — must
    # never appear in model_json_schema() handed to Gemini/Claude/Mistral, and
    # the LLM must not be asked to populate it. Use mark_derived() to record
    # which fields the pipeline filled by normalization (vs. extraction).
    _derived_fields: set[str] = PrivateAttr(default_factory=set)

    def mark_derived(self, path: str) -> None:
        self._derived_fields.add(path)

    def is_derived(self, path: str) -> bool:
        return path in self._derived_fields

    @property
    def derived_fields(self) -> tuple[str, ...]:
        """Read-only sorted view of dotted paths filled by normalization."""
        return tuple(sorted(self._derived_fields))


class ExtractedDocument(BaseModel):
    source_path: str
    extraction_method: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    page_count: int
    tables: list[ExtractedTable] = Field(default_factory=list)
    full_text: str = ""
    invoice: InvoiceData | None = None
    quality_score: float = 0.0     # confidence: extraction-vs-derivation discounted, validation-capped
    quality_coverage: float = 0.0  # how many signals are present; no discount, no cap

    @property
    def name(self) -> str:
        return Path(self.source_path).stem


def is_useful_invoice(inv: InvoiceData | None) -> bool:
    """An InvoiceData is useful if it has at least one identifying or financial field."""
    if inv is None:
        return False
    if inv.invoice_number:
        return True
    if inv.supplier and inv.supplier.eik:
        return True
    if inv.customer and inv.customer.eik:
        return True
    if inv.line_items and len(inv.line_items) > 0:
        return True
    if inv.total_to_pay is not None or inv.net_amount is not None:
        return True
    return False
