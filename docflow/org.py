"""Departments and the modules each one can open.

New department requests add a Department + Module here. The public board
shows every department; login happens after the user picks one.

The products live in sibling folders at the repo root:
`логистика/`, `фактури/` (invoice scan) and `човешки ресурси/` (HR).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Module:
    id: str
    title: str
    summary: str
    status: str  # ready | building


@dataclass(frozen=True)
class Department:
    id: str
    name: str
    tagline: str
    icon: str
    modules: tuple[Module, ...]


INVOICES = Module(
    id="invoices",
    title="Фактури",
    summary="Качване, сканиране и извличане на полета и артикули от фактури към Excel.",
    status="ready",
)

HR_DOCS = Module(
    id="hr_docs",
    title="Документи за самоличност",
    summary="Лични карти, паспорти, договори и молби — преглед, поправка, Excel.",
    status="ready",
)

DEPARTMENTS: dict[str, Department] = {
    "logistics": Department(
        id="logistics",
        name="Логистика",
        tagline="Фактури, доставчици и складови документи към Excel.",
        icon="📦",
        modules=(INVOICES,),
    ),
    "invoices": Department(
        id="invoices",
        name="Фактури",
        tagline="Сканиране на фактури и извличане на полета към Excel.",
        icon="🧾",
        modules=(INVOICES,),
    ),
    "hr": Department(
        id="hr",
        name="Човешки ресурси",
        tagline="Лични карти, договори и молби за отпуск.",
        icon="👥",
        modules=(HR_DOCS,),
    ),
}

DEPARTMENT_ORDER = ("logistics", "invoices", "hr")


def list_departments() -> tuple[Department, ...]:
    return tuple(DEPARTMENTS[dept_id] for dept_id in DEPARTMENT_ORDER if dept_id in DEPARTMENTS)


def get_department(department_id: str) -> Department | None:
    return DEPARTMENTS.get(department_id)


def user_can_open(department_id: str, module_id: str) -> bool:
    dept = get_department(department_id)
    if dept is None:
        return False
    return any(m.id == module_id for m in dept.modules)
