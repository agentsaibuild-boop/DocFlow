from pathlib import Path

import pdfplumber

from docflow.schema import ExtractedDocument, ExtractedTable


class PdfplumberExtractor:
    name = "pdfplumber"

    def can_handle(self, path: Path) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    if (page.extract_text() or "").strip():
                        return True
        except Exception:
            return False
        return False

    def extract(self, path: Path) -> ExtractedDocument:
        tables: list[ExtractedTable] = []
        text_parts: list[str] = []

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)

                for raw_table in page.extract_tables() or []:
                    rows = [
                        [(cell or "").strip() for cell in row]
                        for row in raw_table
                    ]
                    if not rows or all(not any(cell for cell in row) for row in rows):
                        continue
                    tables.append(ExtractedTable(page=i, rows=rows))

        return ExtractedDocument(
            source_path=str(path),
            extraction_method=self.name,
            page_count=page_count,
            tables=tables,
            full_text="\n\n".join(text_parts),
        )
