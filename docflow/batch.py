from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

from docflow.output import write_excel
from docflow.pipeline import NoExtractorFound, extract
from docflow.registry import SupplierRegistry
from docflow.schema import ExtractedDocument
from docflow.status import compute_status
from docflow.validators import validate

SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class BatchResult:
    source: Path
    output: Path | None
    doc: ExtractedDocument | None
    error: str | None
    enrich_findings: list
    validation_findings: list
    registry_findings: list


def discover(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def process_one(path: Path, registry: SupplierRegistry, provider: str = "auto") -> BatchResult:
    try:
        doc = extract(path, provider=provider)
    except Exception as e:
        return BatchResult(path, None, None, f"{type(e).__name__}: {e}", [], [], [])

    enrich_findings = []
    if doc.invoice:
        doc.invoice, enrich_findings = registry.enrich(doc.invoice)

    validation_findings = validate(doc)
    registry_findings = registry.record(doc.invoice, human_confirmed=False) if doc.invoice else []

    return BatchResult(
        path, None, doc, None,
        enrich_findings, validation_findings, registry_findings,
    )


def run_batch(folder: Path, output_dir: Path, provider: str = "auto") -> list[BatchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover(folder)
    if not files:
        return []
    registry = SupplierRegistry()
    results = []
    for f in files:
        print(f"[{len(results) + 1}/{len(files)}] {f.relative_to(folder)}", flush=True)
        results.append(process_one(f, registry, provider=provider))
    write_consolidated(results, output_dir / "Фактури.xlsx")
    return results


def write_consolidated(results: list[BatchResult], output_path: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)

    inv_sheet = wb.create_sheet("Фактури")
    inv_headers = [
        "Файл", "Статус", "Метод",
        "Доставчик", "ЕИК доставчик", "ВАТ доставчик",
        "Получател", "ЕИК получател", "ВАТ получател",
        "Номер", "Дата издаване", "Дата доставка", "Срок плащане",
        "Валута",
        "Сума без отстъпка", "Отстъпка", "Нетна сума",
        "За плащане", "Платени", "Остава",
        "IBAN", "Банка", "BIC", "Метод плащане",
        "Грешка",
    ]
    inv_sheet.append(inv_headers)
    for cell in inv_sheet[1]:
        cell.font = bold

    items_sheet = wb.create_sheet("Артикули")
    items_sheet.append([
        "Файл", "Доставчик", "Номер фактура",
        "№", "Описание", "К-во", "ME",
        "Ед. цена", "Отстъпка %", "Цена с отст.",
        "ДДС %", "ДДС", "Общо без ДДС",
    ])
    for cell in items_sheet[1]:
        cell.font = bold

    vat_sheet = wb.create_sheet("ДДС")
    vat_sheet.append(["Файл", "Доставчик", "Номер фактура", "Ставка %", "Основа", "ДДС"])
    for cell in vat_sheet[1]:
        cell.font = bold

    val_sheet = wb.create_sheet("Validation")
    val_sheet.append(["Файл", "Източник", "Ниво", "Код", "Съобщение"])
    for cell in val_sheet[1]:
        cell.font = bold

    for r in results:
        if r.error:
            status_label = compute_status(None, [], extraction_error=r.error).label
            inv_sheet.append([r.source.name, status_label, "", *[""] * 21, r.error])
            continue
        doc = r.doc
        inv = doc.invoice
        status_label = compute_status(doc, r.validation_findings).label

        if inv is None:
            inv_sheet.append([r.source.name, status_label, doc.extraction_method, *[""] * 22])
        else:
            inv_sheet.append([
                r.source.name, status_label, doc.extraction_method,
                inv.supplier.name, inv.supplier.eik, inv.supplier.vat_number,
                inv.customer.name, inv.customer.eik, inv.customer.vat_number,
                inv.invoice_number, inv.issue_date, inv.delivery_date, inv.payment_due_date,
                inv.currency or "EUR",
                inv.subtotal, inv.discount, inv.net_amount,
                inv.total_to_pay, inv.paid, inv.remaining,
                inv.iban, inv.bank, inv.bic, inv.payment_method,
                "",
            ])

            for li in (inv.line_items or []):
                items_sheet.append([
                    r.source.name, inv.supplier.name, inv.invoice_number,
                    li.number, li.description, li.quantity, li.unit,
                    li.unit_price, li.discount_percent, li.price_after_discount,
                    li.vat_percent, li.vat_amount, li.total_without_vat,
                ])

            for v in (inv.vat_breakdown or []):
                vat_sheet.append([
                    r.source.name, inv.supplier.name, inv.invoice_number,
                    v.rate_percent, v.base_amount, v.vat_amount,
                ])

        for f in r.enrich_findings:
            val_sheet.append([r.source.name, "registry_enrich", f.level, f.code, f.message])
        for f in r.validation_findings:
            val_sheet.append([r.source.name, "validation", f.level, f.code, f.message])
        for f in r.registry_findings:
            val_sheet.append([r.source.name, "registry_record", f.level, f.code, f.message])

    meta_sheet = wb.create_sheet("_meta")
    meta_sheet.append(["batch_run_at", datetime.now().isoformat()])
    meta_sheet.append(["files_total", len(results)])
    meta_sheet.append(["files_ok", sum(1 for r in results if r.error is None)])
    meta_sheet.append(["files_error", sum(1 for r in results if r.error is not None)])
    method_counts = {}
    for r in results:
        if r.doc:
            method_counts[r.doc.extraction_method] = method_counts.get(r.doc.extraction_method, 0) + 1
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        meta_sheet.append([f"метод: {m}", c])

    inv_sheet.freeze_panes = "A2"
    items_sheet.freeze_panes = "A2"
    vat_sheet.freeze_panes = "A2"
    val_sheet.freeze_panes = "A2"

    inv_sheet.column_dimensions["A"].width = 35
    inv_sheet.column_dimensions["D"].width = 30
    inv_sheet.column_dimensions["G"].width = 30
    items_sheet.column_dimensions["A"].width = 35
    items_sheet.column_dimensions["B"].width = 25
    items_sheet.column_dimensions["E"].width = 35
    val_sheet.column_dimensions["A"].width = 35
    val_sheet.column_dimensions["E"].width = 60

    wb.save(output_path)
