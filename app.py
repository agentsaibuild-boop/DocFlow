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

# Streamlit Cloud / Streamlit-hosted deployments don't read .env — they pass
# secrets through st.secrets. Mirror known provider keys into os.environ so
# the rest of the code (extractors, env helpers) keeps working unchanged
# without any cloud-specific branching downstream.
def _hydrate_env_from_streamlit_secrets() -> None:
    try:
        import streamlit as _st
        secrets = dict(_st.secrets)
    except Exception:
        return
    for key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        if key not in os.environ and key in secrets:
            os.environ[key] = str(secrets[key])


_hydrate_env_from_streamlit_secrets()

from docflow.batch import BatchResult, summarize_batch
from docflow.columns import COLUMN_CATALOG, get_row_value
from docflow.eval.benchmarks import BENCHMARKS, get_benchmark
from docflow.line_items import (
    LINE_ITEM_HEADERS,
    VAT_HEADERS,
    detail_line_item_rows,
    detail_vat_rows,
    invoice_review_pairs,
    iter_line_item_rows,
    iter_vat_rows,
)
from docflow.pipeline import AVAILABLE_PROVIDERS, ProviderError, extract, list_providers
from docflow.preview import render_preview
from docflow.provider_catalog import (
    PROFILES, RECOMMENDED_PROVIDER, get_profile, measured_summary,
    options_in_display_order,
)
from docflow.registry import SupplierRegistry
from docflow.status import compute_status
from docflow.validators import validate

from contextlib import contextmanager


def _resolve_key_overrides(state) -> dict[str, str]:
    """Read session-scoped API-key overrides from a Streamlit-like state mapping.

    Returns {} when the user has not opted into custom keys. Keys not provided
    by the user (or whitespace-only) are omitted so the fallback to env/secrets
    stays in effect for that provider.
    """
    if not state.get("use_custom_keys"):
        return {}
    overrides: dict[str, str] = {}
    gemini = (state.get("custom_gemini_key") or "").strip()
    openrouter = (state.get("custom_openrouter_key") or "").strip()
    if gemini:
        overrides["GEMINI_API_KEY"] = gemini
    if openrouter:
        overrides["OPENROUTER_API_KEY"] = openrouter
    return overrides


@contextmanager
def _override_env(vars_dict: dict[str, str]):
    """Temporarily set environment variables for the duration of a block.

    Used to inject session-scoped API keys into os.environ around the
    extraction call so extractors keep reading from os.environ unchanged.
    Values are restored exactly to their prior state on exit (including
    being removed if they didn't exist before).
    """
    saved: dict[str, str | None] = {}
    try:
        for k, v in vars_dict.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


st.set_page_config(page_title="DocFlow", page_icon="📄", layout="wide")
st.title("📄 DocFlow")
st.caption("Извличане на данни от български фактури в табличен вид — включително артикулите")


with st.sidebar:
    st.header("⚙️ Настройки")

    # ─── API key режим ──────────────────────────────────────────────────
    # Session-scoped overrides. Keys never leave st.session_state — no file,
    # database, logs, or repo persistence; cleared automatically on session end.
    st.subheader("🔑 API режим")
    use_custom_keys = st.checkbox(
        "Use my own API keys",
        value=st.session_state.get("use_custom_keys", False),
        key="use_custom_keys",
    )
    if use_custom_keys:
        st.text_input(
            "Gemini API key",
            type="password",
            key="custom_gemini_key",
            help="Пази се само в текущата сесия; не се записва никъде.",
        )
        st.text_input(
            "OpenRouter API key",
            type="password",
            key="custom_openrouter_key",
            help="Пази се само в текущата сесия; не се записва никъде.",
        )
    else:
        st.caption(
            "Демо версията позволява ограничен брой опити на ден. "
            "За пълно тестване включи „Use my own API keys” и въведи своите."
        )

    st.divider()

    providers_status = list_providers()
    available_only = [alias for alias, ok, _ in providers_status if ok]
    ordered = options_in_display_order(available_only)

    def _format(provider_id: str) -> str:
        prof = get_profile(provider_id)
        if not prof:
            return provider_id
        return f"{prof.badge}  {prof.display}"

    default_idx = (
        ordered.index(RECOMMENDED_PROVIDER) if RECOMMENDED_PROVIDER in ordered else 0
    )
    provider = st.selectbox(
        "Модел",
        options=ordered,
        index=default_idx,
        format_func=_format,
        help="Избор на AI модел за извличане на данни от фактурата.",
    )

    _prof = get_profile(provider)
    if _prof is not None:
        st.caption(_prof.headline)
        st.caption(_prof.description)
        if _prof.best_for:
            st.caption("**Подходящо за:** " + " · ".join(_prof.best_for))
        st.caption(f"ℹ️ {_prof.tradeoff}")
        st.caption(f"💰 {_prof.cost}")
        _summary = measured_summary(provider)
        if _summary:
            st.caption(f"📐 {_summary}")
    else:
        st.caption(f"Текущ модел: `{provider}`")

    # Fallback orchestration is kept internal. The capability exists in the
    # pipeline, but exposing it as a UI toggle implied user control over
    # provider routing that we don't yet document well enough (which model
    # takes over, latency change, cost impact). Default behavior: stay on
    # the chosen model; failure is surfaced as a provider error.
    allow_fallback = False

    st.divider()
    st.subheader("📋 Колони за експорт")

    # Flat checkbox UX. Default-on business columns visible at the top,
    # optional business columns behind one expander. Diagnostic columns
    # (quality score, validation errors, registry status, derived fields)
    # are internal telemetry and never appear in the standard flow — they
    # surface only when "Диагностичен режим" is explicitly enabled.
    _visible_keys     = [k for k, _, cat, default in COLUMN_CATALOG
                         if default and cat != "Диагностика"]
    _diagnostic_keys  = [k for k, _, cat, _ in COLUMN_CATALOG if cat == "Диагностика"]
    _optional_keys    = [k for k, _, cat, default in COLUMN_CATALOG
                         if (not default) and cat != "Диагностика"]
    _label_by_key     = {k: lbl for k, lbl, _, _ in COLUMN_CATALOG}
    _default_by_key   = {k: d for k, _, _, d in COLUMN_CATALOG}

    selected_columns: list[str] = []

    def _column_checkbox(key: str) -> None:
        if st.checkbox(_label_by_key[key], value=_default_by_key[key], key=f"col_{key}"):
            selected_columns.append(key)

    for k in _visible_keys:
        _column_checkbox(k)

    with st.expander("Допълнителни колони"):
        for k in _optional_keys:
            _column_checkbox(k)

    st.caption(f"✓ Активни колони: **{len(selected_columns)}**")
    st.session_state["selected_columns"] = selected_columns

    diagnostic_mode = st.toggle(
        "🔧 Диагностичен режим",
        value=False,
        help=(
            "Добавя технически колони към експорта — quality score, validation "
            "errors, registry status, производни полета. Полезно за одит и "
            "проверка как е работила системата на конкретна фактура."
        ),
    )
    if diagnostic_mode:
        st.caption(
            "_Диагностичните колони са вътрешни — не са част от стандартния "
            "счетоводен експорт. Включват се само за одит / debug._"
        )
        for k in _diagnostic_keys:
            _column_checkbox(k)

    st.divider()
    st.subheader("Статус на provider-и")
    for alias, ok, note in providers_status:
        icon = "✅" if ok else "⚠️"
        st.text(f"{icon} {alias}")
        if not ok:
            st.caption(f"   {note}")


NAV_UPLOAD = "Качване"
NAV_RESULTS = "Резултати"
NAV_MODELS = "Модели"
if st.session_state.pop("_go_results", False):
    st.session_state["nav"] = NAV_RESULTS
if "nav" not in st.session_state:
    st.session_state["nav"] = NAV_UPLOAD
nav = st.segmented_control(
    "Раздел",
    options=[NAV_UPLOAD, NAV_RESULTS, NAV_MODELS],
    key="nav",
    label_visibility="collapsed",
)
if nav is None:
    nav = NAV_UPLOAD


MAX_PARALLEL = 5

MAX_FILES_PER_BATCH = 25
MAX_FILE_SIZE_MB    = 25


def _is_over_batch_limit(file_count: int) -> bool:
    """True when the batch exceeds MAX_FILES_PER_BATCH. Processing must block;
    we do not silently truncate."""
    return file_count > MAX_FILES_PER_BATCH


def _over_batch_limit_message(file_count: int) -> str:
    return (
        f"Публичната демо версия приема максимум {MAX_FILES_PER_BATCH} "
        f"файла наведнъж. Моля, качи до {MAX_FILES_PER_BATCH} файла."
    )


def _folder_over_limit_message(file_count: int) -> str:
    return (
        f"Папката съдържа {file_count} файла. "
        f"Максимумът за едно качване е {MAX_FILES_PER_BATCH}. "
        "Намалете съдържанието на папката или използвайте CLI batch режима."
    )


def _process_single(name, get_bytes, get_path, provider, allow_fallback):
    """Worker: runs in thread, no Streamlit calls.

    Returns ("ok", name, doc, None, None, None, preview)  on success
         or ("provider", name, None, None, kind, msg, preview)
         or ("err", name, None, msg, None, None, preview)
    Preview is PNG bytes of the first page (or None).
    """
    tmp_path = get_path()
    cleanup = False
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tf:
            tf.write(get_bytes())
            tmp_path = Path(tf.name)
        cleanup = True
    preview = render_preview(path=tmp_path)
    try:
        doc = extract(tmp_path, provider=provider, allow_fallback=allow_fallback)
        doc.source_path = name
        return ("ok", name, doc, None, None, None, preview)
    except ProviderError as pe:
        # Pass through with the tmp path masked; status.py decides what reaches
        # the UI based on the kind. One UI-safe layer (status.py), not two.
        msg = str(pe).replace(str(tmp_path), name)
        return ("provider", name, None, None, pe.kind, msg, preview)
    except Exception as e:
        msg = str(e).replace(str(tmp_path), name)
        return ("err", name, None, f"{type(e).__name__}: {msg}", None, None, preview)
    finally:
        if cleanup:
            tmp_path.unlink(missing_ok=True)


def process_files(file_sources, registry, provider, allow_fallback):
    """file_sources: list of (display_name, get_bytes_callable, source_path_callable).

    Returns (results, previews) where previews maps file name → PNG bytes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[BatchResult] = []
    previews: dict[str, bytes] = {}
    progress = st.progress(0, text="Подготовка...")
    completed = 0
    total = len(file_sources)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {
            ex.submit(_process_single, name, get_bytes, get_path, provider, allow_fallback): name
            for name, get_bytes, get_path in file_sources
        }
        pending_results = []
        for future in as_completed(futures):
            completed += 1
            name = futures[future]
            progress.progress(completed / total,
                              text=f"[{completed}/{total}] завършен: {name}")
            pending_results.append(future.result())

    name_to_order = {(name): i for i, (name, _, _) in enumerate(file_sources)}
    pending_results.sort(key=lambda r: name_to_order.get(r[1], 999999))

    for status, name, doc, err, perr_kind, perr_msg, preview in pending_results:
        if preview:
            previews[name] = preview
        if status == "provider":
            results.append(BatchResult(
                Path(name), None, None, None, [], [], [],
                provider_error=perr_msg, provider_error_kind=perr_kind,
            ))
            continue
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
    return results, previews


if nav == NAV_UPLOAD:
    mode = st.radio(
        "Източник на фактурите",
        ["📤 Качи файлове", "📁 Папка от път"],
        horizontal=True,
        label_visibility="collapsed",
    )

    file_sources = []

    if mode == "📤 Качи файлове":
        uploaded_files = st.file_uploader(
            "Изберете един или повече файлове с фактури (PDF, JPG, PNG, WEBP, TIF, BMP)",
            accept_multiple_files=True,
            type=["pdf", "jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )
        st.caption(
            f"Публичната демо версия приема до {MAX_FILES_PER_BATCH} файла наведнъж · "
            f"до {MAX_FILE_SIZE_MB} MB на файл. В реална/инсталирана версия "
            "лимитът може да бъде настроен според нуждите."
        )
        if uploaded_files:
            if _is_over_batch_limit(len(uploaded_files)):
                # Hard block — do NOT populate file_sources. The "🚀 Обработи"
                # button stays disabled (it tests `not file_sources`).
                st.error(_over_batch_limit_message(len(uploaded_files)))
            else:
                file_sources = [
                    (u.name, (lambda u=u: u.getvalue()), (lambda: None))
                    for u in uploaded_files
                ]
                st.info(f"📎 {len(uploaded_files)} файла готови. Модел: **{provider}**")
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
                elif _is_over_batch_limit(len(files)):
                    # Hard block — same policy as the upload mode.
                    st.error(_folder_over_limit_message(len(files)))
                else:
                    st.success(f"📂 Готови за обработка **{len(files)}** файла в `{path.name}/`. Модел: **{provider}**")
                    file_sources = [
                        (f.name, (lambda: b""), (lambda f=f: f))
                        for f in files
                    ]

    # ─── Demo / API banner ─────────────────────────────────────────────
    # Visible immediately on first load, directly under the upload area.
    # Bordered container is dark-theme aware via Streamlit's native theme.
    with st.container(border=True):
        st.markdown("**🔑 Demo mode** · **Демо режим**")
        st.markdown(
            "Демо версията позволява ограничен брой опити на ден чрез нашите "
            "API ключове. За пълно тестване отвори страничния панел → "
            "„🔑 API режим” → включи „Use my own API keys” и въведи свои "
            "Gemini / OpenRouter ключове. Ключовете остават само в текущата "
            "сесия и не се записват никъде."
        )
        st.markdown(
            "_The public demo runs on our limited API keys with a daily cap. "
            "For unrestricted testing, open the left sidebar → „🔑 API режим” → "
            "enable „Use my own API keys” and paste your own Gemini / "
            "OpenRouter keys. They stay in this session only and are never "
            "written to disk, logs, or repository._"
        )

    process_btn = st.button(
        "🚀 Обработи",
        type="primary",
        disabled=not file_sources,
        use_container_width=False,
    )

    if process_btn and file_sources:
        registry = SupplierRegistry()
        with _override_env(_resolve_key_overrides(st.session_state)):
            results, previews = process_files(file_sources, registry, provider, allow_fallback)
        st.session_state["last_results"] = results
        st.session_state["previews"] = previews
        if results:
            st.session_state["selected_invoice"] = results[0].source.name
        st.session_state["_go_results"] = True
        st.rerun()

    # ─── Accessibility info ───────────────────────────────────────────
    # Visible in the main page (not behind a sidebar expander) so screen-reader
    # users and EU evaluators can find it without exploring widgets.
    with st.container(border=True):
        st.markdown("**♿ Accessibility · Достъпност**")
        st.markdown(
            "Приложението използва текстови етикети, описателни инструкции и "
            "стандартни Streamlit контроли, за да бъде по-достъпно за "
            "потребители със screen reader."
        )
        st.markdown(
            "_The application uses text labels, descriptive instructions, and "
            "standard Streamlit controls to improve accessibility for screen "
            "reader users._"
        )


elif nav == NAV_RESULTS:
    if "last_results" not in st.session_state:
        st.info("Качи и обработи фактури в раздел **Качване**.")
    else:
        results: list[BatchResult] = st.session_state["last_results"]
        selected = st.session_state.get("selected_columns", [k for k, _, _, d in COLUMN_CATALOG if d])
        label_for = {k: lbl for k, lbl, _, _ in COLUMN_CATALOG}

        import pandas as pd

        summary = summarize_batch(results)
        c1, c2, c3 = st.columns(3)
        c1.metric("Обработени", summary["processed"])
        c2.metric("Извлечени OK", summary["extracted_ok"])
        c3.metric("Validation грешки", summary["validation_errors"])
        if summary["provider_failures"]:
            st.warning(
                f"{summary['provider_failures']} файла не са обработени поради "
                "грешка на модела (квота, връзка или ключ). "
                "Документите не са лоши — опитай отново или избери друг модел."
            )

        rows = []
        for r in results:
            row = {"Файл": r.source.name}
            if r.provider_error:
                status = compute_status(
                    None, [],
                    provider_error=r.provider_error,
                    provider_error_kind=r.provider_error_kind,
                )
                row["Статус"] = status.label
                row["Метод"] = "—"
                for col in selected:
                    row[label_for[col]] = ""
                row["Бележки"] = status.notes
            elif r.error:
                status = compute_status(None, [], extraction_error=r.error)
                row["Статус"] = status.label
                row["Метод"] = "—"
                for col in selected:
                    row[label_for[col]] = ""
                row["Бележки"] = status.notes
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
        items_rows = iter_line_item_rows(results)
        vat_rows = iter_vat_rows(results)
        items_df = pd.DataFrame(items_rows, columns=LINE_ITEM_HEADERS)
        vat_df = pd.DataFrame(vat_rows, columns=VAT_HEADERS)

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Фактури")
            items_df.to_excel(writer, index=False, sheet_name="Артикули")
            vat_df.to_excel(writer, index=False, sheet_name="ДДС")

        st.download_button(
            "⬇️ Свали Excel",
            data=xlsx_buf.getvalue(),
            file_name="Фактури.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.caption(
            f"Excel: **Фактури**, **Артикули** ({len(items_rows)} реда), **ДДС**."
        )

        names = [r.source.name for r in results]
        if not names:
            st.info("Няма резултати.")
        else:
            current = st.session_state.get("selected_invoice")
            if current not in names:
                st.session_state["selected_invoice"] = names[0]
            chosen = st.selectbox(
                "Фактура за преглед",
                options=names,
                key="selected_invoice",
            )
            chosen_result = next(r for r in results if r.source.name == chosen)

            col_orig, col_fields = st.columns(2, gap="large")
            with col_orig:
                st.subheader("Оригинал")
                preview = (st.session_state.get("previews") or {}).get(chosen)
                if preview:
                    st.image(preview, width="stretch")
                else:
                    st.caption("Няма преглед за този файл.")

            with col_fields:
                st.subheader("Извлечени полета")
                if chosen_result.provider_error or chosen_result.error:
                    status = compute_status(
                        None, [],
                        provider_error=chosen_result.provider_error,
                        provider_error_kind=chosen_result.provider_error_kind,
                        extraction_error=chosen_result.error,
                    )
                    st.error(status.label)
                    st.caption(status.notes)
                else:
                    status = compute_status(
                        chosen_result.doc,
                        chosen_result.validation_findings,
                        required_column_keys=selected,
                    )
                    st.markdown(f"**{status.label}**")
                    if status.notes:
                        st.caption(status.notes)
                    pairs = invoice_review_pairs(
                        chosen_result.doc,
                        validation_findings=chosen_result.validation_findings,
                        registry_findings=chosen_result.registry_findings,
                    )
                    st.dataframe(
                        pd.DataFrame(pairs, columns=["Поле", "Стойност"]),
                        width="stretch",
                        hide_index=True,
                    )

            item_headers, item_detail = detail_line_item_rows(chosen_result)
            st.subheader(f"Артикули — {len(item_detail)}")
            if item_detail:
                st.dataframe(
                    pd.DataFrame(item_detail, columns=item_headers),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "Няма извлечени артикули за тази фактура. "
                    "Провери дали таблицата със стоки/услуги се вижда на оригинала."
                )

            vat_headers, vat_detail = detail_vat_rows(chosen_result)
            if vat_detail:
                st.subheader("ДДС")
                st.dataframe(
                    pd.DataFrame(vat_detail, columns=vat_headers),
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Всички фактури"):
            st.dataframe(df, width="stretch", hide_index=True)

        with st.expander("Validation проблеми"):
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
                st.dataframe(pd.DataFrame(issues_rows), width="stretch", hide_index=True)
            else:
                st.success("Няма validation проблеми")


elif nav == NAV_MODELS:
    st.subheader("🤖 Модели за извличане")
    st.caption(
        "Подкрепени AI модели. Изборът е твой — описанието под всеки казва "
        "за какъв вид фактури е силен и какво носят със себе си."
    )

    for prof in PROFILES:
        if prof.provider not in AVAILABLE_PROVIDERS:
            continue
        with st.container(border=True):
            st.markdown(f"### {prof.badge}  {prof.display}")
            st.markdown(f"_{prof.headline}_")
            st.markdown(prof.description)
            if prof.best_for:
                st.markdown("**Подходящо за:** " + " · ".join(prof.best_for))
            st.markdown(f"ℹ️ {prof.tradeoff}")
            st.markdown(f"💰 {prof.cost}")
            _sum = measured_summary(prof.provider)
            if _sum:
                st.markdown(f"📐 {_sum}")

    with st.expander("Сравнителна таблица (за напреднали)"):
        import pandas as pd
        rows = []
        for prof in PROFILES:
            if prof.provider not in AVAILABLE_PROVIDERS:
                continue
            rows.append({
                "Модел":          prof.display,
                "Etикет":         prof.badge,
                "За какво е":     prof.headline,
                "Подходящо за":   " · ".join(prof.best_for),
                "Какво да знаеш": prof.tradeoff,
                "Цена":           prof.cost,
                "От нашия тест":  measured_summary(prof.provider) or "Все още нямаме достатъчно реални тестове.",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
