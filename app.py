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

from docflow.batch import BatchResult, write_consolidated
from docflow.pipeline import AVAILABLE_PROVIDERS, extract, list_providers
from docflow.registry import SupplierRegistry
from docflow.validators import validate

DEFAULT_PROVIDER = "gemini-3.1-flash-lite"

# (Колона за Excel, Етикет, Категория, По подразбиране)
COLUMN_CATALOG = [
    ("supplier_name",  "Доставчик",          "Основни",  True),
    ("supplier_vat",   "Номер по ДДС",       "Основни",  True),
    ("supplier_iban",  "IBAN доставчик",     "Основни",  True),
    ("invoice_number", "Номер фактура",      "Основни",  True),
    ("invoice_date",   "Дата фактура",       "Основни",  True),
    ("net_amount",     "Цена без ДДС",       "Основни",  True),
    ("vat_total",      "ДДС",                "Основни",  True),
    ("total_to_pay",   "Крайна сума",        "Основни",  True),

    ("supplier_eik",     "ЕИК доставчик",      "Доставчик",  False),
    ("supplier_address", "Адрес доставчик",    "Доставчик",  False),
    ("supplier_mol",     "МОЛ доставчик",      "Доставчик",  False),
    ("supplier_phone",   "Телефон доставчик",  "Доставчик",  False),
    ("supplier_email",   "Email доставчик",    "Доставчик",  False),
    ("supplier_bank",    "Банка доставчик",    "Доставчик",  False),
    ("supplier_bic",     "BIC доставчик",      "Доставчик",  False),

    ("customer_name",    "Получател",          "Получател",  False),
    ("customer_eik",     "ЕИК получател",      "Получател",  False),
    ("customer_vat",     "Номер по ДДС получ.","Получател",  False),
    ("customer_address", "Адрес получател",    "Получател",  False),
    ("customer_mol",     "МОЛ получател",      "Получател",  False),

    ("delivery_date",    "Дата доставка",      "Дати",       False),
    ("due_date",         "Срок плащане",       "Дати",       False),

    ("currency",         "Валута",             "Плащане",    False),
    ("subtotal",         "Сума без отстъпка",  "Плащане",    False),
    ("discount",         "Отстъпка",           "Плащане",    False),
    ("payment_method",   "Метод плащане",      "Плащане",    False),
    ("paid",             "Платени",            "Плащане",    False),
    ("remaining",        "Остава",             "Плащане",    False),
]


def get_row_value(field: str, doc):
    """Map a column key to the actual value from ExtractedDocument."""
    if doc is None or doc.invoice is None:
        return ""
    inv = doc.invoice
    s = inv.supplier
    c = inv.customer
    if inv.vat_breakdown:
        vat_total = sum(v.vat_amount for v in inv.vat_breakdown)
    elif inv.total_to_pay is not None and inv.net_amount is not None:
        vat_total = round(inv.total_to_pay - inv.net_amount, 2)
    else:
        vat_total = None
    return {
        "supplier_name":    s.name if s else "",
        "supplier_eik":     s.eik if s else "",
        "supplier_vat":     s.vat_number if s else "",
        "supplier_address": s.address if s else "",
        "supplier_mol":     s.mol if s else "",
        "supplier_phone":   s.phone if s else "",
        "supplier_email":   s.email if s else "",
        "supplier_iban":    inv.iban,
        "supplier_bank":    inv.bank,
        "supplier_bic":     inv.bic,
        "customer_name":    c.name if c else "",
        "customer_eik":     c.eik if c else "",
        "customer_vat":     c.vat_number if c else "",
        "customer_address": c.address if c else "",
        "customer_mol":     c.mol if c else "",
        "invoice_number":   inv.invoice_number,
        "invoice_date":     inv.issue_date,
        "delivery_date":    inv.delivery_date,
        "due_date":         inv.payment_due_date,
        "currency":         inv.currency,
        "subtotal":         inv.subtotal,
        "discount":         inv.discount,
        "net_amount":       inv.net_amount,
        "vat_total":        vat_total,
        "total_to_pay":     inv.total_to_pay,
        "payment_method":   inv.payment_method,
        "paid":             inv.paid,
        "remaining":        inv.remaining,
    }.get(field, "")

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

    selected_columns = []
    for cat, items in categories.items():
        defaults = {key for key, _, c, d in COLUMN_CATALOG if c == cat and d}
        with st.expander(cat, expanded=(cat == "Основни")):
            for key, label in items:
                checked = st.checkbox(
                    label, value=(key in defaults), key=f"col_{key}",
                )
                if checked:
                    selected_columns.append(key)
    st.session_state["selected_columns"] = selected_columns

    st.divider()
    st.subheader("Статус на provider-и")
    for alias, ok, note in providers_status:
        icon = "✅" if ok else "⚠️"
        st.text(f"{icon} {alias}")
        if not ok:
            st.caption(f"   {note}")


tab_upload, tab_results, tab_registry = st.tabs(["📤 Качване", "📊 Резултати", "🏢 Регистър"])


def process_files(file_sources, registry, provider):
    """file_sources: list of (display_name, get_bytes_callable, source_path_callable)."""
    results: list[BatchResult] = []
    progress = st.progress(0, text="Подготовка...")
    for i, (name, get_bytes, get_path) in enumerate(file_sources, start=1):
        progress.progress((i - 1) / len(file_sources),
                          text=f"[{i}/{len(file_sources)}] {name}")
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
            enrich_findings = []
            if doc.invoice:
                doc.invoice, enrich_findings = registry.enrich(doc.invoice)
            validation_findings = validate(doc)
            registry_findings = registry.record(doc.invoice) if doc.invoice else []
            results.append(BatchResult(
                Path(name), None, doc, None,
                enrich_findings, validation_findings, registry_findings,
            ))
        except Exception as e:
            msg = str(e).replace(str(tmp_path), name)
            if "All extractors failed" in msg or "Gemini still unavailable" in msg or "503" in msg:
                msg = f"{provider} върна 503/quota. Пробвай пак или избери друг model."
            results.append(BatchResult(
                Path(name), None, None,
                f"{type(e).__name__}: {msg}", [], [], [],
            ))
        finally:
            if cleanup:
                tmp_path.unlink(missing_ok=True)
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

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
            output_path = Path(tf.name)
        write_consolidated(results, output_path)
        st.session_state["last_results"] = results
        st.session_state["last_xlsx"] = output_path.read_bytes()
        output_path.unlink(missing_ok=True)

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
                row["Статус"] = "❌ ERROR"
                row["Метод"] = "—"
                for col in selected:
                    row[label_for[col]] = ""
                row["Бележки"] = r.error[:80]
            else:
                doc = r.doc
                inv = doc.invoice
                if inv is None or not inv.supplier or not inv.supplier.name:
                    row["Статус"] = "⚠️ Без данни"
                elif any(f.level == "error" for f in r.validation_findings):
                    row["Статус"] = "⚠️ Има грешки"
                else:
                    row["Статус"] = "✅ OK"
                row["Метод"] = doc.extraction_method.replace("gemini:", "").replace(":claude-sonnet-4-6", "")
                for col in selected:
                    row[label_for[col]] = get_row_value(col, doc)
                row["Бележки"] = ""
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        xlsx_buf = io.BytesIO()
        df.to_excel(xlsx_buf, index=False, sheet_name="Фактури")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Свали Excel (избрани колони)",
                data=xlsx_buf.getvalue(),
                file_name="Фактури.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "⬇️ Свали пълен Excel (всички листи)",
                data=st.session_state["last_xlsx"],
                file_name="Фактури_пълен.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.subheader(f"Преглед — {len(selected)} избрани колони")
        st.dataframe(df, use_container_width=True, hide_index=True)

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
