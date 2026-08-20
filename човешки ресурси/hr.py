"""Човешки ресурси — Streamlit екран на http://localhost:8501."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from docflow.org import get_department
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
    rotate_preview,
    rows_to_clipboard,
    save_archive,
    search_archive,
    upright_card_preview,
)


def _process_upload(item) -> tuple[HrResult, bytes | None]:
    suffix = Path(item.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(item.getvalue())
        tmp = Path(tf.name)
    try:
        preview = upright_card_preview(render_preview(path=tmp))
        doc = extract_identity(tmp)
        return build_hr_result(item.name, doc), preview
    except Exception as exc:
        return HrResult(name=item.name, error=str(exc)), upright_card_preview(render_preview(path=tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base} ({n})" in taken:
        n += 1
    return f"{base} ({n})"


def render_hr_workspace() -> None:
    dept = get_department("hr")
    st.title(f"{dept.icon}  {dept.name}" if dept else "Човешки ресурси")
    st.caption(
        "Качи лична карта, паспорт, книжка, трудов договор или молба за отпуск. "
        "Поправи полетата и свали Excel. Файлът се чете тук; за разпознаване отива към Gemini."
    )

    NAV_UPLOAD = "Качване"
    NAV_RESULTS = "Резултати"
    NAV_ARCHIVE = "Архив"
    if st.session_state.pop("_hr_go_results", False):
        st.session_state["hr_nav"] = NAV_RESULTS
    if "hr_nav" not in st.session_state:
        st.session_state["hr_nav"] = NAV_UPLOAD
    nav = st.segmented_control(
        "Раздел",
        options=[NAV_UPLOAD, NAV_RESULTS, NAV_ARCHIVE],
        key="hr_nav",
        label_visibility="collapsed",
    )
    if nav is None:
        nav = NAV_UPLOAD

    if nav == NAV_UPLOAD:
        uploaded = st.file_uploader(
            "Снимка или PDF (лице, гръб, договор, молба)",
            accept_multiple_files=True,
            type=["pdf", "jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
        )
        pair_sides = st.checkbox(
            "Обедини по две: първи файл = лице, втори = гръб",
            value=True,
        )
        process = st.button("Обработи", type="primary", disabled=not uploaded)
        if process and uploaded:
            results: list[HrResult] = []
            previews: dict[str, bytes] = {}
            backs: dict[str, bytes] = {}
            taken: set[str] = set()
            progress = st.progress(0, text="Обработка...")
            items = list(uploaded)
            i = 0
            done = 0
            while i < len(items):
                first, preview_a = _process_upload(items[i])
                done += 1
                progress.progress(done / len(items), text=items[i].name)
                if pair_sides and i + 1 < len(items) and first.doc is not None:
                    second, preview_b = _process_upload(items[i + 1])
                    done += 1
                    progress.progress(done / len(items), text=items[i + 1].name)
                    if second.doc is not None:
                        front, back, swapped = order_front_back(first, second)
                        if swapped:
                            preview_a, preview_b = preview_b, preview_a
                        merged = merge_hr_results(front, back)
                        merged.name = _unique_name(
                            person_full_name(merged.doc) or merged.name, taken,
                        )
                        taken.add(merged.name)
                        results.append(merged)
                        if preview_a:
                            previews[merged.name] = preview_a
                        if preview_b:
                            backs[merged.name] = preview_b
                        i += 2
                        continue
                    results.append(first)
                    taken.add(first.name)
                    if preview_a:
                        previews[first.name] = preview_a
                    results.append(second)
                    taken.add(second.name)
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
            progress.progress(1.0, text="Готово")
            st.session_state["hr_results"] = results
            st.session_state["hr_previews"] = previews
            st.session_state["hr_preview_backs"] = backs
            if results:
                st.session_state["hr_selected"] = results[0].name
                save_archive(results)
            st.session_state["_hr_go_results"] = True
            st.rerun()
        return

    if nav == NAV_ARCHIVE:
        _render_archive()
        return

    results: list[HrResult] | None = st.session_state.get("hr_results")
    if not results:
        st.info("Качи документ в **Качване** и натисни Обработи.")
        return

    query = st.text_input("Търсене по ЕГН, име или адрес", key="hr_search")
    visible = [item for item in results if result_matches(item, query)]
    if not visible:
        st.warning("Няма запис по това търсене.")
        return

    names = [item.name for item in visible]
    if st.session_state.get("hr_selected") not in names:
        st.session_state["hr_selected"] = names[0]
    chosen = st.selectbox("Документ", options=names, key="hr_selected")
    current = next(item for item in visible if item.name == chosen)
    idx = next(i for i, item in enumerate(results) if item.name == chosen)

    hide_empty = st.checkbox("Скрий празните полета", value=True, key="hr_hide_empty")

    st.subheader("Оригинал")
    preview = (st.session_state.get("hr_previews") or {}).get(chosen)
    back = (st.session_state.get("hr_preview_backs") or {}).get(chosen)
    shown_front = upright_card_preview(preview) or preview
    shown_back = upright_card_preview(back) or back
    front_col, back_col = st.columns(2)
    with front_col:
        st.caption("Лице")
        if shown_front:
            st.image(shown_front, width="stretch")
            if st.button("Завърти лице", key=f"rot_front_{chosen}"):
                store = dict(st.session_state.get("hr_previews") or {})
                store[chosen] = rotate_preview(shown_front, 90)
                st.session_state["hr_previews"] = store
                st.rerun()
        else:
            st.caption("Няма преглед за лице.")
    with back_col:
        st.caption("Гръб")
        if shown_back:
            st.image(shown_back, width="stretch")
            if st.button("Завърти гръб", key=f"rot_back_{chosen}"):
                store = dict(st.session_state.get("hr_preview_backs") or {})
                store[chosen] = rotate_preview(shown_back, 90)
                st.session_state["hr_preview_backs"] = store
                st.rerun()
        elif shown_front:
            st.caption("Няма гръб.")
        else:
            st.caption("Няма преглед за този файл.")
    if shown_front and shown_back and st.button("Размени лице и гръб"):
        faces = dict(st.session_state.get("hr_previews") or {})
        backs = dict(st.session_state.get("hr_preview_backs") or {})
        faces[chosen], backs[chosen] = shown_back, shown_front
        st.session_state["hr_previews"] = faces
        st.session_state["hr_preview_backs"] = backs
        st.rerun()

    st.subheader("Извлечени полета")
    if current.error:
        st.error(current.error)
    elif current.doc is None:
        st.warning("Няма извлечени данни.")
    else:
        current = build_hr_result(current.name, current.doc)
        if current.egn_ok is False:
            st.warning("ЕГН-то не минава контролната сума — провери спрямо оригинала.")
        if document_is_expired(current.doc):
            st.warning("Документът е с изтекла валидност.")
        field_rows = identity_field_rows(
            current, include_empty=not hide_empty, include_egn_check=False,
        )
        for extra in current.extra:
            if hide_empty and not extra.value:
                continue
            field_rows.append({"Файл": current.name, "Поле": extra.label, "Стойност": extra.value})
        table = pd.DataFrame(field_rows)[["Поле", "Стойност"]] if field_rows else pd.DataFrame(
            columns=["Поле", "Стойност"]
        )
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            height=min(42 + 37 * max(len(table), 1), 1600),
            num_rows="dynamic",
            disabled=["Поле"],
            column_config={
                "Поле": st.column_config.TextColumn("Поле", width="medium"),
                "Стойност": st.column_config.TextColumn("Стойност", width="large"),
            },
            key=f"hr_editor_{chosen}_{hide_empty}",
        )
        save_col, copy_col = st.columns(2)
        with save_col:
            if st.button("Запази поправки", type="primary"):
                updated = apply_field_edits(current, edited.to_dict("records"))
                results[idx] = updated
                st.session_state["hr_results"] = results
                save_archive(results)
                st.success("Поправките са записани.")
                st.rerun()
        with copy_col:
            st.download_button(
                "Копирай всички редове",
                data=rows_to_clipboard(edited.to_dict("records")),
                file_name="полета.tsv",
                mime="text/tab-separated-values",
                help="Сваля същата таблица (Поле / Стойност), за да я отвориш в Excel.",
            )

    st.download_button(
        "Свали Excel",
        data=identity_to_excel(results),
        file_name=excel_filename(results),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def _render_archive() -> None:
    files = list_archive()
    query = st.text_input("Търсене в архива по ЕГН или име", key="hr_archive_search")
    if query.strip():
        hits = search_archive(query)
        if not hits:
            st.warning("Няма съвпадение в архива.")
            return
        st.caption(f"{len(hits)} записа")
        for path, result in hits:
            label = person_full_name(result.doc) or result.name
            extra = result.doc.egn if result.doc else ""
            st.markdown(f"**{label}** · {extra} · `{path.name}`")
            if st.button("Отвори пакета", key=f"open_hit_{path.name}_{result.name}"):
                st.session_state["hr_results"] = load_archive(path)
                st.session_state["hr_previews"] = {}
                st.session_state["hr_preview_backs"] = {}
                st.session_state["hr_selected"] = st.session_state["hr_results"][0].name
                st.session_state["_hr_go_results"] = True
                st.rerun()
        return

    if not files:
        st.info("Все още няма локален архив. След обработка записите се пазят на този компютър.")
        return

    for path in files:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"`{path.name}`")
        with col_b:
            if st.button("Отвори", key=f"open_arch_{path.name}"):
                loaded = load_archive(path)
                st.session_state["hr_results"] = loaded
                st.session_state["hr_previews"] = {}
                st.session_state["hr_preview_backs"] = {}
                if loaded:
                    st.session_state["hr_selected"] = loaded[0].name
                st.session_state["_hr_go_results"] = True
                st.rerun()
