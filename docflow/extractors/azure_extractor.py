"""Azure AI Document Intelligence prebuilt-invoice v4.0 extractor.

Uses Microsoft's specialized invoice model — trained on millions of invoices,
returns structured fields with per-field confidence scores. Often best-in-class
for standard invoice fields; less flexible than general LLMs for non-invoice docs.
"""

import os
from pathlib import Path

from docflow.schema import (
    ExtractedDocument, ExtractedTable, InvoiceData, LineItem, Party, VatBreakdown,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heif"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}

MODEL_ID = "prebuilt-invoice"


class AzureExtractor:
    name = "azure_di_invoice"
    endpoint_env = "AZURE_DI_ENDPOINT"
    key_env = "AZURE_DI_KEY"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            endpoint = os.environ.get(self.endpoint_env)
            key = os.environ.get(self.key_env)
            if not endpoint or not key:
                raise RuntimeError(
                    f"{self.endpoint_env} and {self.key_env} not set. "
                    f"Create resource at https://portal.azure.com (Document Intelligence, Free F0 tier)"
                )
            self._client = DocumentIntelligenceClient(
                endpoint=endpoint, credential=AzureKeyCredential(key)
            )
        return self._client

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return False
        return bool(os.environ.get(self.endpoint_env)) and bool(os.environ.get(self.key_env))

    def extract(self, path: Path) -> ExtractedDocument:
        client = self._get_client()
        with open(path, "rb") as f:
            poller = client.begin_analyze_document(
                MODEL_ID, body=f.read(), content_type="application/octet-stream"
            )
        result = poller.result()

        invoice = self._azure_to_invoice(result)
        tables = self._invoice_to_tables(invoice)
        page_count = len(result.pages) if result.pages else 1

        return ExtractedDocument(
            source_path=str(path),
            extraction_method=f"{self.name}:v4.0",
            page_count=page_count,
            tables=tables,
            full_text=result.content or "",
            invoice=invoice,
        )

    @staticmethod
    def _val(field) -> str | None:
        """Extract string value from Azure DocumentField (handles content vs value variants)."""
        if field is None:
            return None
        v = getattr(field, "value_string", None) or getattr(field, "content", None)
        return v.strip() if v else None

    @staticmethod
    def _num(field) -> float | None:
        if field is None:
            return None
        v = getattr(field, "value_currency", None)
        if v is not None:
            return float(v.amount)
        v = getattr(field, "value_number", None)
        if v is not None:
            return float(v)
        return None

    @staticmethod
    def _date(field) -> str | None:
        if field is None:
            return None
        v = getattr(field, "value_date", None)
        if v:
            return v.isoformat() if hasattr(v, "isoformat") else str(v)
        return getattr(field, "content", None)

    @classmethod
    def _split_tax_id(cls, tax_id: str | None) -> tuple[str | None, str | None]:
        """Bulgarian VAT (BG...) → (eik, vat_number). Plain digits → (eik, None)."""
        if not tax_id:
            return None, None
        clean = tax_id.strip().replace(" ", "")
        if clean.upper().startswith("BG"):
            digits = clean[2:]
            if digits.isdigit() and len(digits) in (9, 10, 13):
                return digits, clean.upper()
            return None, clean.upper()
        if clean.isdigit() and len(clean) in (9, 10, 13):
            return clean, None
        return None, None

    @classmethod
    def _azure_to_invoice(cls, result) -> InvoiceData | None:
        if not result.documents:
            return None
        doc = result.documents[0]
        fields = doc.fields or {}

        def get(name):
            return fields.get(name)

        supplier_eik, supplier_vat = cls._split_tax_id(cls._val(get("VendorTaxId")))
        customer_eik, customer_vat = cls._split_tax_id(cls._val(get("CustomerTaxId")))

        supplier = Party(
            name=cls._val(get("VendorName")),
            address=cls._val(get("VendorAddress")),
            eik=supplier_eik,
            vat_number=supplier_vat,
        )
        customer = Party(
            name=cls._val(get("CustomerName")),
            address=cls._val(get("CustomerAddress")),
            eik=customer_eik,
            vat_number=customer_vat,
        )

        line_items: list[LineItem] = []
        items_field = get("Items")
        if items_field and hasattr(items_field, "value_array") and items_field.value_array:
            for i, item in enumerate(items_field.value_array, start=1):
                item_fields = item.value_object if hasattr(item, "value_object") else {}
                line_items.append(LineItem(
                    number=i,
                    description=cls._val(item_fields.get("Description")),
                    quantity=cls._num(item_fields.get("Quantity")),
                    unit=cls._val(item_fields.get("Unit")),
                    unit_price=cls._num(item_fields.get("UnitPrice")),
                    vat_percent=cls._num(item_fields.get("TaxRate")),
                    vat_amount=cls._num(item_fields.get("Tax")),
                    total_without_vat=cls._num(item_fields.get("Amount")),
                ))

        total_tax = cls._num(get("TotalTax"))
        subtotal = cls._num(get("SubTotal"))
        vat_breakdown: list[VatBreakdown] = []
        if total_tax is not None and subtotal is not None:
            rate = round((total_tax / subtotal * 100), 2) if subtotal else 0.0
            vat_breakdown.append(VatBreakdown(
                rate_percent=rate, base_amount=subtotal, vat_amount=total_tax,
            ))

        return InvoiceData(
            invoice_number=cls._val(get("InvoiceId")),
            issue_date=cls._date(get("InvoiceDate")),
            payment_due_date=cls._date(get("DueDate")),
            supplier=supplier,
            customer=customer,
            line_items=line_items,
            subtotal=subtotal,
            net_amount=subtotal,
            vat_breakdown=vat_breakdown,
            total_to_pay=cls._num(get("InvoiceTotal")) or cls._num(get("AmountDue")),
            paid=cls._num(get("AmountDue")) and cls._num(get("InvoiceTotal")) and (
                cls._num(get("InvoiceTotal")) - cls._num(get("AmountDue"))
            ) or None,
            payment_method=cls._val(get("PaymentTerm")),
        )

    @staticmethod
    def _invoice_to_tables(inv: InvoiceData | None) -> list[ExtractedTable]:
        if inv is None or not inv.line_items:
            return []
        header = ["№", "Описание", "К-во", "ME", "Ед. цена",
                  "Отстъпка %", "Цена с отст.", "ДДС %", "ДДС", "Общо без ДДС"]
        rows = [header]
        for li in inv.line_items:
            rows.append([
                str(li.number) if li.number is not None else "",
                li.description or "",
                str(li.quantity) if li.quantity is not None else "",
                li.unit or "",
                str(li.unit_price) if li.unit_price is not None else "",
                "", "",
                str(li.vat_percent) if li.vat_percent is not None else "",
                str(li.vat_amount) if li.vat_amount is not None else "",
                str(li.total_without_vat) if li.total_without_vat is not None else "",
            ])
        return [ExtractedTable(page=1, rows=rows)]
