"""First-page preview PNG for the results review pane."""

from __future__ import annotations

import io
import threading
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
_PDFIUM_LOCK = threading.Lock()
PDF_DPI = 110
MAX_WIDTH = 1200


def render_preview(
    path: Path | None = None,
    data: bytes | None = None,
    suffix: str | None = None,
) -> bytes | None:
    """Return PNG bytes of the first page / image, or None if unsupported."""
    ext = None
    if suffix:
        ext = suffix.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
    elif path is not None:
        ext = path.suffix.lower()

    raw = data
    if raw is None and path is not None:
        try:
            raw = path.read_bytes()
        except OSError:
            return None
    if not raw or not ext:
        return None

    try:
        if ext in PDF_EXTS:
            return _pdf_first_page_png(raw)
        if ext in IMAGE_EXTS:
            return _image_to_png(raw)
    except Exception:
        return None
    return None


def _image_to_png(raw: bytes) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if im.width > MAX_WIDTH:
        ratio = MAX_WIDTH / im.width
        im = im.resize((MAX_WIDTH, max(1, int(im.height * ratio))))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _pdf_first_page_png(raw: bytes) -> bytes | None:
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(raw)
        try:
            if len(pdf) == 0:
                return None
            page = pdf[0]
            try:
                pil_image = page.render(scale=PDF_DPI / 72).to_pil()
            finally:
                page.close()
        finally:
            pdf.close()

    if pil_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / pil_image.width
        pil_image = pil_image.resize((MAX_WIDTH, max(1, int(pil_image.height * ratio))))
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
