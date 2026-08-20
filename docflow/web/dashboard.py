"""Public department board, then per-department login and workspace."""

from __future__ import annotations

import streamlit as st

from docflow.auth import User
from docflow.org import Department, get_department, list_departments


def render_public_board() -> None:
    st.title("DocFlow")
    st.caption("Общ дашборд — избери отдел, после влез с име и парола.")

    departments = list_departments()
    cols = st.columns(len(departments) or 1, gap="medium")
    for col, dept in zip(cols, departments):
        with col:
            with st.container(border=True, height=360, key=f"dept_card_{dept.id}"):
                st.markdown(f"# {dept.icon}")
                st.markdown(f"### {dept.name}")
                st.caption(dept.tagline)
                with st.container(height="stretch", vertical_alignment="bottom"):
                    if st.button(
                        "Вход",
                        key=f"enter_{dept.id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["selected_department"] = dept.id
                        st.session_state["view"] = "login"
                        st.rerun()


def render_department_login(dept: Department) -> tuple[str, str, bool]:
    if st.button("← Всички отдели"):
        st.session_state.pop("selected_department", None)
        st.session_state["view"] = "home"
        st.rerun()

    st.title(f"{dept.icon}  {dept.name}")
    st.caption("Вход само за този отдел.")

    with st.form("department_login"):
        username = st.text_input("Име")
        password = st.text_input("Парола", type="password")
        submitted = st.form_submit_button("Вход", type="primary")

    return username, password, submitted


def render_department_home(user: User) -> None:
    dept = get_department(user.department)
    if dept is None:
        st.error("Непознат отдел за този акаунт. Свържи се с администратора.")
        return

    st.title(f"{dept.icon}  {dept.name}")
    st.caption(f"{user.display_name} · {dept.tagline}")
    st.markdown("Избери модул. Всеки отдел ще получава свои екрани според работата си.")

    cols = st.columns(max(1, min(3, len(dept.modules))))
    for col, module in zip(cols, dept.modules):
        with col:
            with st.container(border=True):
                status_label = "Готов" if module.status == "ready" else "В разработка"
                st.subheader(module.title)
                st.caption(status_label)
                st.markdown(module.summary)
                label = "Отвори" if module.status == "ready" else "Отвори (чернова)"
                if st.button(label, key=f"open_{module.id}", type="primary" if module.status == "ready" else "secondary"):
                    st.session_state["view"] = module.id
                    st.rerun()
