"""Shared extraction-prompt fragments.

Kept in one place so Gemini / OpenRouter / Claude / text-LLM all ask for the
same line-item contract. The models already receive `InvoiceData` as a schema;
this text is what actually makes them fill `line_items` instead of stopping at
header totals.
"""

LINE_ITEMS_SECTION = """
═══ LINE ITEMS / АРТИКУЛИ (REQUIRED) ═══
Every invoice has billed goods or services — a table labelled Артикули, Стоки и
услуги, Описание, Description, Item, Qty, or similar. You MUST extract EVERY
data row into `line_items`. Header totals alone are not enough.

Include, as a separate LineItem for each billed row:
  • `description` — the FULL name as printed (product/service + code/SKU + extra
    lines that belong to that row). Do not truncate or summarise.
  • `quantity`, `unit` (бр, кг, м2, ч, hours, pcs, …)
  • `unit_price` without VAT, `discount_percent`, `price_after_discount`
  • `vat_percent`, `vat_amount`, `total_without_vat`
  • `number` — the row № from the table when shown

Exclude:
  • the table header row
  • summary rows: Общо, Данъчна основа, ДДС, Отстъпка, За плащане, Total, Subtotal
  • bank details, notes, stamp/signature lines

Do NOT collapse several articles into one description.
Do NOT return an empty `line_items` array if any goods/services table is visible.
If the invoice is a single lump-sum service with no table, emit one LineItem
with that description and the net amount.
"""
