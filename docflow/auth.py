"""Local username/password accounts bound to a department.

Passwords are stored as PBKDF2-SHA256 hashes in a JSON file that is not
committed. On first local run, seed accounts are created for each department
if the file is missing. Later departments are appended when still absent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from docflow.org import DEPARTMENTS

USERS_FILENAME = "users.json"
PBKDF2_ROUNDS = 200_000
_SCHEME = "pbkdf2_sha256"


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    department: str
    password_hash: str


def users_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parent.parent
    override = os.environ.get("DOCFLOW_USERS_FILE")
    if override:
        return Path(override).expanduser()
    return base / USERS_FILENAME


def hash_password(password: str, salt_hex: str | None = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    return f"{_SCHEME}${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, rounds_s, salt_hex, digest_hex = stored.split("$")
        rounds = int(rounds_s)
    except ValueError:
        return False
    if scheme != _SCHEME or rounds < 1:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        rounds,
    )
    return hmac.compare_digest(candidate.hex(), digest_hex)


def _user_from_dict(raw: dict) -> User:
    username = str(raw.get("username", "")).strip()
    department = str(raw.get("department", "")).strip()
    display = str(raw.get("display_name", "")).strip() or username
    password_hash = str(raw.get("password_hash", "")).strip()
    if not username or not department or not password_hash:
        raise ValueError("user record is missing username, department, or password_hash")
    if department not in DEPARTMENTS:
        raise ValueError(f"unknown department {department!r} for user {username!r}")
    return User(
        username=username,
        display_name=display,
        department=department,
        password_hash=password_hash,
    )


def load_users(path: Path | None = None) -> list[User]:
    target = path or users_path()
    if not target.exists():
        return []
    data = json.loads(target.read_text(encoding="utf-8"))
    rows = data.get("users", data) if isinstance(data, dict) else data
    return [_user_from_dict(row) for row in rows]


def save_users(users: list[User], path: Path | None = None) -> None:
    target = path or users_path()
    payload = {
        "users": [
            {
                "username": u.username,
                "display_name": u.display_name,
                "department": u.department,
                "password_hash": u.password_hash,
            }
            for u in users
        ]
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_user(username: str, users: list[User] | None = None) -> User | None:
    needle = username.strip().lower()
    if not needle:
        return None
    for user in users if users is not None else load_users():
        if user.username.lower() == needle:
            return user
    return None


def authenticate(
    username: str,
    password: str,
    users: list[User] | None = None,
    department_id: str | None = None,
) -> User | None:
    user = find_user(username, users)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    if department_id and user.department != department_id:
        return None
    return user


def _seed_account(department: str) -> User:
    if department == "logistics":
        return User(
            username=os.environ.get("AUTH_LOGISTICS_USER", "logistika"),
            display_name=os.environ.get("AUTH_LOGISTICS_NAME", "Логистика"),
            department="logistics",
            password_hash=hash_password(os.environ.get("AUTH_LOGISTICS_PASSWORD", "logistika")),
        )
    if department == "hr":
        return User(
            username=os.environ.get("AUTH_HR_USER", "hr"),
            display_name=os.environ.get("AUTH_HR_NAME", "Човешки ресурси"),
            department="hr",
            password_hash=hash_password(os.environ.get("AUTH_HR_PASSWORD", "hr")),
        )
    if department == "invoices":
        return User(
            username=os.environ.get("AUTH_INVOICES_USER", "fakturi"),
            display_name=os.environ.get("AUTH_INVOICES_NAME", "Фактури"),
            department="invoices",
            password_hash=hash_password(os.environ.get("AUTH_INVOICES_PASSWORD", "fakturi")),
        )
    raise ValueError(f"no seed account for department {department!r}")


def seed_local_users(path: Path | None = None) -> list[User]:
    """Create department accounts when no users file exists yet.

    Passwords come from AUTH_*_PASSWORD env vars, or the local defaults
    logistika / hr / fakturi (meant only for a private office install).
    """
    target = path or users_path()
    if target.exists():
        return load_users(target)

    users = [_seed_account(dept_id) for dept_id in ("logistics", "invoices", "hr")]
    save_users(users, target)
    return users


def ensure_users_file(path: Path | None = None) -> list[User]:
    target = path or users_path()
    users = load_users(target) if target.exists() else seed_local_users(target)
    have = {user.department for user in users}
    missing = [dept_id for dept_id in ("logistics", "invoices", "hr") if dept_id not in have]
    if missing:
        users = [*users, *(_seed_account(dept_id) for dept_id in missing)]
        save_users(users, target)
    return users
