"""DocFlow web app — Streamlit interface for invoice extraction.

Run with:
    streamlit run app.py
"""

import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from docflow.env import load_env_file
load_env_file(Path(__file__).parent / ".env")

from docflow.batch import BatchResult
from docflow.columns import COLUMN_CATALOG, get_row_value
from docflow.pipeline import AVAILABLE_PROVIDERS, extract, list_providers
from docflow.registry import SupplierRegistry
from docflow.status import compute_status
from docflow.validators import validate

DEFAULT_PROVIDER = "gemini-3.1-flash-lite"

st.set_page_config(page_title="DocFlow", page_icon="📄", layout="wide")
st.title("📄 DocFlow")
st.caption("Извличане на данни от български фактури в табличен вид")


with st.sidebar:
    st.header("⚙️ Настройки")

    providers_status = list_providers()
    available_only = [alias for alias, ok, _ in providers_status if ok]
    available_only = ["auto"] + available_only
    default_idx = available_only.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in available_only else 0

    provider = st.selectbox(
        "Модел за извличане",
        options=available_only,
        index=default_idx,
        help="Кой API ще се ползва. 'auto' пробва всички по реда им.",
    )

    st.divider()
    st.subheader("📋 Колони за експорт")
    categories: dict[str, list] = {}
    for key, label, cat, _ in COLUMN_CATALOG:
        categories.setdefault(cat, []).append((key, label))

    label_to_key = {label: key for key, label, _, _ in COLUMN_CATALOG}
    selected_columns = []

    for cat, items in categories.items():
        category_defaults = [label for key, label, c, d in COLUMN_CATALOG if c == cat and d]
        category_labels = [label for _, label in items]
        picked = st.multiselect(
            cat,
            options=category_labels,
            default=category_defaults,
            key=f"ms_{cat}",
        )
        for label in picked:
            selected_columns.append(label_to_key[label])

    st.session_state["selected_columns"] = selected_columns
    st.caption(f"✓ Активни колони: **{len(selected_columns)}**")

    st.divider()
    st.subheader("Статус на provider-и")
    for alias, ok, note in providers_status:
        icon = "✅" if ok else "⚠️"
        st.text(f"{icon} {alias}")
        if not ok:
            st.caption(f"   {note}")


tab_upload, tab_results, tab_registry = st.tabs(["📤 Качване", "📊 Резултати", "🏢 Регистър"])


MAX_PARALLEL = 5


def _process_single(name, get_bytes, get_path, provider):
    """Worker: runs in thread, no Streamlit calls."""
    tmp_path = get_path()
    cleanup = False
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tf:
            tf.write(get_bytes())
            tmp_path = Path(tf.name)
        cleanup = True
    try:
        doc = extract(tmp_path, provider=provider)
        doc.source_path = name
        return ("ok", name, doc, None)
    except Exception as e:
        msg = str(e).replace(str(tmp_path), name)
        if "All extractors failed" in msg or "Gemini still unavailable" in msg or "503" in msg:
            msg = f"{provider} върна 503/quota. Пробвай пак или избери друг model."
        return ("err", name, None, f"{type(e).__name__}: {msg}")
    finally:
        if cleanup:
            tmp_path.unlink(missing_ok=True)


def process_files(file_sources, registry, provider):
    """file_sources: list of (display_name, get_bytes_callable, source_path_callable)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[BatchResult] = []
    progress = st.progress(0, text="Подготовка...")
    completed = 0
    total = len(file_sources)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {
            ex.submit(_process_single, name, get_bytes, get_path, provider): name
            for name, get_bytes, get_path in file_sources
        }
        pending_results = []
        for future in as_completed(futures):
            completed += 1
            name = futures[future]
            progress.progress(completed / total,
                              text=f"[{completed}/{total}] завършен: {name}")
            status, name, doc, err = future.result()
            pending_results.append((name, status, doc, err))

    name_to_order = {(name): i for i, (name, _, _) in enumerate(file_sources)}
    pending_results.sort(key=lambda r: name_to_order.get(r[0], 999999))

    for name, status, doc, err in pending_results:
        if status == "err":
            results.append(BatchResult(Path(name), None, None, err, [], [], []))
            continue
        enrich_findings = []
        if doc.invoice:
            doc.invoice, enrich_findings = registry.enrich(doc.invoice)
        validation_findings = validate(doc)
        registry_findings = registry.record(doc.invoice, human_confirmed=False) if doc.invoice else []
        results.append(BatchResult(
            Path(name), None, doc, None,
            enrich_findings, validation_findings, registry_findings,
        ))

    progress.progress(1.0, text=f"Готово: {len(results)} файла")
    return results


with tab_upload:
    mode = st.radio(
        "Източник на фактурите",
        ["📤 Качи файлове", "📁 Папка от път (за големи batch-ове)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    file_sources = []

    if mode == "📤 Качи файлове":
        uploaded_files = st.file_uploader(
            "PDF, JPG, PNG (избери един или много)",
            accept_multiple_files=True,
            type=["pdf", "jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded_files:
            file_sources = [
                (u.name, (lambda u=u: u.getvalue()), (lambda: None))
                for u in uploaded_files
            ]
            st.info(f"📎 {len(uploaded_files)} файла готови. Provider: **{provider}**")
    else:
        from docflow.batch import discover
        default_path = str(Path.home() / "Desktop")

        col_pick, col_path = st.columns([1, 4])
        with col_pick:
            if st.button("📁 Избери папка", use_container_width=True):
                import subprocess

                start = st.session_state.get("folder_path", default_path)
                try:
                    r = subprocess.run(
                        ["zenity", "--file-selection", "--directory",
                         "--title=Избери папка с фактури", f"--filename={start}/"],
                        capture_output=True, text=True, timeout=120,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        st.session_state["folder_path"] = r.stdout.strip()
                        st.rerun()
                except FileNotFoundError:
                    st.error("zenity липсва. `sudo apt install zenity` или ползвай ръчно поле.")
                except subprocess.TimeoutExpired:
                    pass

        with col_path:
            current_path = st.session_state.get("folder_path", default_path)
            folder_path = st.text_input(
                "или въведи път ръчно",
                value=current_path,
                label_visibility="collapsed",
                placeholder="/path/to/invoices",
            )
            if folder_path != current_path:
                st.session_state["folder_path"] = folder_path

        if folder_path:
            path = Path(folder_path).expanduser()
            if not path.exists():
                st.error(f"❌ Папката не съществува: {path}")
            elif not path.is_dir():
                st.error(f"❌ Не е директория: {path}")
            else:
                files = discover(path)
                if not files:
                    st.warning(f"⚠️ Няма поддържани файлове в {path}")
                else:
                    st.success(f"📂 Намерени **{len(files)}** файла в `{path.name}/`. Provider: **{provider}**")
                    file_sources = [
                        (f.name, (lambda: b""), (lambda f=f: f))
                        for f in files
                    ]

    process_btn = st.button(
        "🚀 Обработи",
        type="primary",
        disabled=not file_sources,
        use_container_width=False,
    )

    if process_btn and file_sources:
        registry = SupplierRegistry()
        results = process_files(file_sources, registry, provider)
        st.session_state["last_results"] = results

        ok = sum(1 for r in results if r.error is None and r.doc and r.doc.invoice
                 and r.doc.invoice.supplier and r.doc.invoice.supplier.name)
        err = len(results) - ok
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Файлове", len(results))
        col_b.metric("С пълни данни", ok)
        col_c.metric("Изискват преглед", err)

        st.success("✅ Обработено. Виж раздел **Резултати** за детайли.")


with tab_results:
    if "last_results" not in st.session_state:
        st.info("Качи и обработи фактури в раздел **Качване**.")
    else:
        results: list[BatchResult] = st.session_state["last_results"]
        selected = st.session_state.get("selected_columns", [k for k, _, _, d in COLUMN_CATALOG if d])
        label_for = {k: lbl for k, lbl, _, _ in COLUMN_CATALOG}

        import pandas as pd
        rows = []
        for r in results:
            row = {"Файл": r.source.name}
            if r.error:
                status = compute_status(None, [], extraction_error=r.error)
                row["Статус"] = status.label
                row["Метод"] = "—"
                for col in selected:
                    row[label_for[col]] = ""
                row["Бележки"] = r.error[:80]
            else:
                doc = r.doc
                row["Метод"] = doc.extraction_method.replace("gemini:", "").replace(":claude-sonnet-4-6", "")

                for col in selected:
                    row[label_for[col]] = get_row_value(
                        col, doc,
                        validation_findings=r.validation_findings,
                        registry_findings=r.registry_findings,
                    )

                status = compute_status(
                    doc,
                    r.validation_findings,
                    required_column_keys=selected,
                )
                row["Статус"] = status.label
                if status.code == "INCOMPLETE":
                    empty_labels = [label_for[k] for k in status.empty_columns]
                    row["Бележки"] = "Празни: " + ", ".join(empty_labels)
                else:
                    row["Бележки"] = status.notes
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        xlsx_buf = io.BytesIO()
        df.to_excel(xlsx_buf, index=False, sheet_name="Фактури")

        st.download_button(
            "⬇️ Свали Excel",
            data=xlsx_buf.getvalue(),
            file_name="Фактури.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        st.subheader(f"Преглед — {len(selected)} избрани колони")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("💾 Запис в регистър на доставчик")
        st.caption(
            "Регистърът се записва **само** след ръчно потвърждение. "
            "Това предотвратява автоматично закотвяне на грешно извлечени данни."
        )

        recordable: list[tuple[int, BatchResult]] = []
        for idx, r in enumerate(results):
            if r.error or not r.doc or not r.doc.invoice:
                continue
            inv = r.doc.invoice
            if not inv.supplier or not inv.supplier.name:
                continue
            if not (inv.supplier.eik or inv.supplier.vat_number):
                continue
            recordable.append((idx, r))

        if not recordable:
            st.info("Няма фактури с достатъчно данни за запис (нужни са име на доставчик + ЕИК или ИН по ДДС).")
        else:
            options = {
                f"{r.source.name} — {r.doc.invoice.supplier.name}": (idx, r)
                for idx, r in recordable
            }
            choice = st.selectbox(
                "Избери фактура за запис на доставчика",
                options=list(options.keys()),
            )
            if choice and st.button(
                "Потвърждавам данните и записвам доставчика",
                type="primary",
                key="confirm_record",
            ):
                _, chosen = options[choice]
                registry = SupplierRegistry()
                findings = registry.record(chosen.doc.invoice, human_confirmed=True)
                for f in findings:
                    if f.level == "ok":
                        st.success(f.message)
                    elif f.level == "warning":
                        st.warning(f.message)
                    else:
                        st.info(f.message)

        with st.expander("🔍 Validation проблеми (детайли)"):
            issues_rows = []
            for r in results:
                for f in r.validation_findings:
                    if f.level in ("warning", "error"):
                        issues_rows.append({
                            "Файл": r.source.name,
                            "Ниво": f.level,
                            "Код": f.code,
                            "Съобщение": f.message,
                        })
            if issues_rows:
                st.dataframe(pd.DataFrame(issues_rows), use_container_width=True, hide_index=True)
            else:
                st.success("Няма validation проблеми ✓")


with tab_registry:
    st.subheader("Регистър на познатите доставчици")
    st.caption("Системата запомня доставчиците от обработените фактури и автоматично "
               "коригира известни полета при следващи фактури от същия доставчик.")
    registry = SupplierRegistry()
    suppliers = registry.all()
    if not suppliers:
        st.info("Регистърът е празен. Обработи няколко фактури за да се запълни.")
    else:
        import pandas as pd
        df = pd.DataFrame(suppliers)
        for col in ["aliases", "first_seen"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        st.dataframe(df, use_container_width=True, hide_index=True)
