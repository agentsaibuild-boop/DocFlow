"""DocFlow — public department board, then per-department login.

Run with:
    streamlit run app.py
"""

from pathlib import Path
import sys

import streamlit as st

from docflow.env import load_env_file

load_env_file(Path(__file__).parent / ".env")

from docflow.projects import HR_DIR, INVOICES_DIR, LOGISTICS_DIR

for _project_dir in (HR_DIR, LOGISTICS_DIR, INVOICES_DIR):
    _path = str(_project_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from docflow.auth import User, authenticate, ensure_users_file, find_user, verify_password
from docflow.org import Department, get_department, user_can_open
from docflow.web.dashboard import (
    render_department_home,
    render_department_login,
    render_public_board,
)
from hr import render_hr_workspace
from invoices import (
    MAX_FILES_PER_BATCH,
    MAX_FILE_SIZE_MB,
    _folder_over_limit_message,
    _is_over_batch_limit,
    _over_batch_limit_message,
    _override_env,
    _resolve_key_overrides,
    render_invoices_module,
)

st.set_page_config(page_title="DocFlow", page_icon="📄", layout="wide")
ensure_users_file()


def user_from_session() -> User | None:
    raw = st.session_state.get("user")
    if not isinstance(raw, dict):
        return None
    return find_user(str(raw.get("username", "")))


def _logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _chrome(user: User) -> None:
    dept = get_department(user.department)
    with st.sidebar:
        st.markdown(f"**{user.display_name}**")
        st.caption(dept.name if dept else user.department)
        if user.department not in ("hr", "invoices") and st.button("Модули", use_container_width=True):
            st.session_state["view"] = "workspace"
            st.rerun()
        if st.button("Всички отдели", use_container_width=True):
            _logout()
            st.rerun()
        if st.button("Изход", use_container_width=True):
            _logout()
            st.rerun()
        st.divider()


def _render_module(user: User, module_id: str) -> None:
    if not user_can_open(user.department, module_id):
        st.error("Този модул не е достъпен за вашия отдел.")
        if st.button("Към модулите"):
            st.session_state["view"] = "workspace"
            st.rerun()
        return
    if module_id == "invoices":
        render_invoices_module()
        return
    if module_id == "hr_docs":
        render_hr_workspace()
        return
    st.session_state["view"] = "workspace"
    st.rerun()


def _handle_department_login(dept: Department) -> None:
    username, password, submitted = render_department_login(dept)
    if not submitted:
        return
    user = find_user(username)
    if user is not None and verify_password(password, user.password_hash) and user.department != dept.id:
        st.error(f"Този акаунт не е за отдел {dept.name}.")
        return
    authed = authenticate(username, password, department_id=dept.id)
    if authed is None:
        st.error("Грешно име или парола.")
        return
    st.session_state["user"] = {
        "username": authed.username,
        "display_name": authed.display_name,
        "department": authed.department,
    }
    st.session_state["view"] = "workspace"
    st.rerun()


user = user_from_session()
view = st.session_state.get("view", "home")
selected_dept_id = st.session_state.get("selected_department")

if user is None and view not in ("login", "home"):
    view = "home"

if user is None and view != "login":
    render_public_board()
elif user is None:
    dept = get_department(str(selected_dept_id or ""))
    if dept is None:
        st.session_state["view"] = "home"
        st.rerun()
    else:
        _handle_department_login(dept)
else:
    _chrome(user)
    if user.department == "hr":
        render_hr_workspace()
    elif user.department == "invoices":
        render_invoices_module()
    elif view in ("home", "login", "workspace"):
        render_department_home(user)
    else:
        _render_module(user, view)
