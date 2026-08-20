"""Локално HR приложение (не Streamlit). Старт: python старт.py"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
for path in (str(REPO), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from docflow.env import load_env_file

load_env_file(REPO / ".env")

from docflow.preview import render_preview
from identity import (
    HrResult,
    apply_field_edits,
    build_hr_result,
    document_is_expired,
    excel_filename,
    extract_identity,
    identity_field_rows,
    identity_to_excel,
    list_archive,
    load_archive,
    merge_hr_results,
    order_front_back,
    person_full_name,
    result_matches,
    rows_to_clipboard,
    save_archive,
    upright_card_preview,
)


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base} ({n})" in taken:
        n += 1
    return f"{base} ({n})"


def _process_path(path: Path) -> tuple[HrResult, bytes | None]:
    preview = upright_card_preview(render_preview(path=path))
    try:
        doc = extract_identity(path)
        return build_hr_result(path.name, doc), preview
    except Exception as exc:
        return HrResult(name=path.name, error=str(exc)), preview


def run_batch(
    files: list[Path], pair: bool,
) -> tuple[list[HrResult], dict[str, bytes], dict[str, bytes]]:
    results: list[HrResult] = []
    previews: dict[str, bytes] = {}
    backs: dict[str, bytes] = {}
    taken: set[str] = set()
    i = 0
    while i < len(files):
        first, preview_a = _process_path(files[i])
        if pair and i + 1 < len(files) and first.doc is not None:
            second, preview_b = _process_path(files[i + 1])
            if second.doc is not None:
                front, back, swapped = order_front_back(first, second)
                if swapped:
                    preview_a, preview_b = preview_b, preview_a
                merged = merge_hr_results(front, back)
                merged.name = _unique_name(person_full_name(merged.doc) or merged.name, taken)
                taken.add(merged.name)
                results.append(merged)
                if preview_a:
                    previews[merged.name] = preview_a
                if preview_b:
                    backs[merged.name] = preview_b
                i += 2
                continue
            first.name = _unique_name(person_full_name(first.doc) or first.name, taken)
            taken.add(first.name)
            results.append(first)
            if preview_a:
                previews[first.name] = preview_a
            second.name = _unique_name(person_full_name(second.doc) or second.name, taken)
            taken.add(second.name)
            results.append(second)
            if preview_b:
                previews[second.name] = preview_b
            i += 2
            continue
        first.name = _unique_name(person_full_name(first.doc) or first.name, taken)
        taken.add(first.name)
        results.append(first)
        if preview_a:
            previews[first.name] = preview_a
        i += 1
    return results, previews, backs


class Worker(QThread):
    finished_ok = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, files: list[Path], pair: bool) -> None:
        super().__init__()
        self.files = files
        self.pair = pair

    def run(self) -> None:
        try:
            results, previews, backs = run_batch(self.files, self.pair)
            self.finished_ok.emit(results, previews, backs)
        except Exception as exc:
            self.failed.emit(str(exc))


def _pixmap(raw: bytes | None) -> QPixmap | None:
    if not raw:
        return None
    image = QImage.fromData(raw)
    if image.isNull():
        from PIL import Image

        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        image = QImage.fromData(buf.getvalue())
    if image.isNull():
        return None
    pix = QPixmap.fromImage(image)
    return pix.scaled(400, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class HrWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Човешки ресурси")
        self.resize(1280, 780)
        self.results: list[HrResult] = []
        self.previews: dict[str, bytes] = {}
        self.backs: dict[str, bytes] = {}
        self.chosen: list[Path] = []
        self.worker: Worker | None = None
        self._build()
        self._refresh_archive()

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        bar = QHBoxLayout()
        pick = QPushButton("Избери файлове")
        pick.clicked.connect(self._pick_files)
        bar.addWidget(pick)
        self.pair = QCheckBox("Лице, после гръб = един човек")
        self.pair.setChecked(True)
        bar.addWidget(self.pair)
        self.hide_empty = QCheckBox("Скрий празни полета")
        self.hide_empty.setChecked(True)
        self.hide_empty.toggled.connect(lambda _: self._redraw_fields())
        bar.addWidget(self.hide_empty)
        process = QPushButton("Обработи")
        process.clicked.connect(self._process)
        bar.addWidget(process)
        save = QPushButton("Запази поправки")
        save.clicked.connect(self._save_edits)
        bar.addWidget(save)
        excel = QPushButton("Свали Excel")
        excel.clicked.connect(self._save_excel)
        bar.addWidget(excel)
        bar.addStretch()
        layout.addLayout(bar)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Търсене"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("ЕГН, име или адрес")
        self.search.textChanged.connect(self._refresh_list)
        search_row.addWidget(self.search)
        layout.addLayout(search_row)

        split = QSplitter()
        layout.addWidget(split, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(QLabel("Документи"))
        self.docs = QListWidget()
        self.docs.currentRowChanged.connect(lambda _: self._on_select())
        left_l.addWidget(self.docs)
        left_l.addWidget(QLabel("Локален архив"))
        self.archive = QListWidget()
        left_l.addWidget(self.archive)
        open_arch = QPushButton("Отвори от архива")
        open_arch.clicked.connect(self._open_archive)
        left_l.addWidget(open_arch)
        split.addWidget(left)

        mid = QWidget()
        mid_l = QVBoxLayout(mid)
        mid_l.addWidget(QLabel("Оригинал"))
        self.front = QLabel("Няма преглед")
        self.front.setAlignment(Qt.AlignCenter)
        self.back = QLabel("")
        self.back.setAlignment(Qt.AlignCenter)
        mid_l.addWidget(self.front)
        mid_l.addWidget(self.back)
        split.addWidget(mid)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.addWidget(QLabel("Извлечени полета — редактират се"))
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #8a5a00")
        right_l.addWidget(self.warn)
        copy = QHBoxLayout()
        for label, kind in (
            ("Копирай ЕГН", "egn"),
            ("Копирай имена", "name"),
            ("Копирай адрес", "address"),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=kind: self._copy(k))
            copy.addWidget(btn)
        copy_sel = QPushButton("Копирай избраните редове")
        copy_sel.clicked.connect(lambda: self._copy_rows(selected_only=True))
        copy.addWidget(copy_sel)
        copy_all = QPushButton("Копирай всички редове")
        copy_all.clicked.connect(lambda: self._copy_rows(selected_only=False))
        copy.addWidget(copy_all)
        right_l.addLayout(copy)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Поле", "Стойност"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        QShortcut(QKeySequence.Copy, self.table, lambda: self._copy_rows(True))
        QShortcut(QKeySequence.SelectAll, self.table, self.table.selectAll)
        right_l.addWidget(self.table)
        split.addWidget(right)
        split.setSizes([240, 420, 560])

        self.status = QLabel(
            "Локално приложение. Файловете стоят на този компютър. "
            "За разпознаване се пращат към Gemini; DocFlow не пази снимката."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _pick_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "Документи",
            str(Path.home()),
            "Документи (*.pdf *.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp);;Всички (*)",
        )
        self.chosen = [Path(name) for name in names]
        if self.chosen:
            self.status.setText(f"Избрани {len(self.chosen)} файла. Натисни Обработи.")

    def _process(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self.chosen:
            QMessageBox.information(self, "Човешки ресурси", "Първо избери файлове.")
            return
        self.status.setText("Обработка…")
        self.worker = Worker(list(self.chosen), self.pair.isChecked())
        self.worker.finished_ok.connect(self._done_ok)
        self.worker.failed.connect(self._done_error)
        self.worker.start()

    def _done_error(self, message: str) -> None:
        self.status.setText("Грешка при обработка.")
        QMessageBox.critical(self, "Човешки ресурси", message)

    def _done_ok(self, results, previews, backs) -> None:
        self.results = results
        self.previews = previews
        self.backs = backs
        if results:
            save_archive(results)
        self._refresh_list()
        self._refresh_archive()
        if self.results:
            self.docs.setCurrentRow(0)
        self.status.setText(f"Готово: {len(results)} записа. Архивът е само на този компютър.")

    def _visible(self) -> list[HrResult]:
        return [item for item in self.results if result_matches(item, self.search.text())]

    def _refresh_list(self) -> None:
        self.docs.clear()
        for item in self._visible():
            self.docs.addItem(item.name)

    def _current(self) -> HrResult | None:
        visible = self._visible()
        row = self.docs.currentRow()
        if row < 0 or row >= len(visible):
            return None
        return visible[row]

    def _on_select(self) -> None:
        current = self._current()
        self._show_previews(current)
        self._redraw_fields()

    def _show_previews(self, current: HrResult | None) -> None:
        name = current.name if current else ""
        front = _pixmap(self.previews.get(name))
        back = _pixmap(self.backs.get(name))
        if front:
            self.front.setPixmap(front)
        else:
            self.front.setText("Няма преглед (лице)")
        if back:
            self.back.setPixmap(back)
        else:
            self.back.setText("Няма гръб" if front else "")

    def _redraw_fields(self) -> None:
        current = self._current()
        self.table.setRowCount(0)
        if current is None:
            self.warn.setText("")
            return
        warnings: list[str] = []
        if current.error:
            warnings.append(current.error)
        elif current.doc is None:
            warnings.append("Няма извлечени данни.")
        else:
            if current.egn_ok is False:
                warnings.append("ЕГН-то не минава контролната сума.")
            if document_is_expired(current.doc):
                warnings.append("Документът е с изтекла валидност.")
        self.warn.setText("  ".join(warnings))
        if current.doc is None:
            return
        rows = identity_field_rows(
            current,
            include_empty=not self.hide_empty.isChecked(),
            include_egn_check=False,
        )
        for extra in current.extra:
            if self.hide_empty.isChecked() and not extra.value:
                continue
            rows.append({"Поле": extra.label, "Стойност": extra.value})
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(row["Поле"]))
            self.table.setItem(i, 1, QTableWidgetItem(row.get("Стойност") or ""))

    def _collect_rows(self) -> list[dict[str, str]]:
        rows = []
        for i in range(self.table.rowCount()):
            label = self.table.item(i, 0)
            value = self.table.item(i, 1)
            rows.append({
                "Поле": label.text().strip() if label else "",
                "Стойност": value.text() if value else "",
            })
        return rows

    def _save_edits(self) -> None:
        current = self._current()
        if current is None or current.doc is None:
            return
        updated = apply_field_edits(current, self._collect_rows())
        self.results = [updated if item.name == current.name else item for item in self.results]
        save_archive(self.results)
        self._refresh_list()
        self.status.setText("Поправките са записани локално.")
        QMessageBox.information(self, "Човешки ресурси", "Поправките са записани.")

    def _save_excel(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Човешки ресурси", "Няма данни за Excel.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Запиши Excel",
            str(Path.home() / excel_filename(self.results)),
            "Excel (*.xlsx)",
        )
        if not path:
            return
        Path(path).write_bytes(identity_to_excel(self.results))
        self.status.setText(f"Записан Excel: {path}")

    def _copy(self, kind: str) -> None:
        current = self._current()
        if current is None or current.doc is None:
            return
        text = {
            "egn": current.doc.egn or "",
            "name": person_full_name(current.doc),
            "address": current.doc.permanent_address or "",
        }.get(kind, "")
        if not text:
            QMessageBox.information(self, "Човешки ресурси", "Няма стойност за копиране.")
            return
        QApplication.clipboard().setText(text)
        self.status.setText(f"Копирано: {text}")

    def _rows_from_table(self, indexes: list[int] | None = None) -> list[dict[str, str]]:
        if indexes is None:
            indexes = list(range(self.table.rowCount()))
        rows = []
        for i in indexes:
            label = self.table.item(i, 0)
            value = self.table.item(i, 1)
            rows.append({
                "Поле": label.text() if label else "",
                "Стойност": value.text() if value else "",
            })
        return rows

    def _copy_rows(self, selected_only: bool = True) -> None:
        if selected_only:
            indexes = sorted({item.row() for item in self.table.selectedItems()})
            if not indexes:
                indexes = list(range(self.table.rowCount()))
        else:
            indexes = list(range(self.table.rowCount()))
        if not indexes:
            return
        text = rows_to_clipboard(self._rows_from_table(indexes))
        QApplication.clipboard().setText(text)
        self.status.setText(f"Копирани {len(indexes)} реда. Ctrl+V ги слага в Excel като таблица.")

    def _refresh_archive(self) -> None:
        self.archive.clear()
        for path in list_archive():
            self.archive.addItem(path.name)

    def _open_archive(self) -> None:
        row = self.archive.currentRow()
        files = list_archive()
        if row < 0 or row >= len(files):
            return
        self.results = load_archive(files[row])
        self.previews = {}
        self.backs = {}
        self._refresh_list()
        if self.results:
            self.docs.setCurrentRow(0)
        self.status.setText(f"Отворен архив: {files[row].name}")


def main() -> None:
    app = QApplication(sys.argv)
    window = HrWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
