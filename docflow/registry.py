import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docflow.schema import InvoiceData


DEFAULT_DB_PATH = Path("registry.db")


@dataclass
class RegistryFinding:
    level: str
    code: str
    message: str
    field: str | None = None
    known: str | None = None
    seen: str | None = None


class SupplierRegistry:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS suppliers (
                    eik TEXT PRIMARY KEY,
                    name TEXT,
                    address TEXT,
                    vat_number TEXT,
                    mol TEXT,
                    phone TEXT,
                    email TEXT,
                    iban TEXT,
                    bank TEXT,
                    bic TEXT,
                    invoice_count INTEGER NOT NULL DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)

    def lookup(self, eik: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM suppliers WHERE eik = ?", (eik,)).fetchone()
            return dict(row) if row else None

    def record(self, inv: InvoiceData, human_confirmed: bool = False) -> list[RegistryFinding]:
        if not human_confirmed:
            return [RegistryFinding(
                level="info",
                code="registry_skipped",
                message="Registry write skipped — requires human confirmation",
            )]
        if not inv.supplier:
            return []
        eik = _derive_eik(inv.supplier.eik, inv.supplier.vat_number)
        if not eik:
            return []
        now = datetime.now().isoformat()
        existing = self.lookup(eik)
        findings: list[RegistryFinding] = []

        if existing is None:
            with self._conn() as c:
                c.execute("""
                    INSERT INTO suppliers
                    (eik, name, address, vat_number, mol, phone, email, iban, bank, bic,
                     invoice_count, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """, (
                    eik, inv.supplier.name, inv.supplier.address, inv.supplier.vat_number,
                    inv.supplier.mol, inv.supplier.phone, inv.supplier.email,
                    inv.iban, inv.bank, inv.bic, now, now,
                ))
            findings.append(RegistryFinding(
                "ok", "supplier_new",
                f"Нов доставчик записан: {inv.supplier.name} (ЕИК {eik})",
            ))
            return findings

        comparisons = [
            ("name", existing.get("name"), inv.supplier.name),
            ("address", existing.get("address"), inv.supplier.address),
            ("vat_number", existing.get("vat_number"), inv.supplier.vat_number),
            ("iban", existing.get("iban"), inv.iban),
            ("bank", existing.get("bank"), inv.bank),
            ("bic", existing.get("bic"), inv.bic),
        ]
        fills_to_apply: dict[str, str] = {}
        for field, known, seen in comparisons:
            if seen is None:
                continue
            if known is None:
                fills_to_apply[field] = str(seen)
                findings.append(RegistryFinding(
                    "ok", "supplier_field_filled",
                    f"{field}: ново попълнено в регистъра '{seen}'",
                    field=field, seen=str(seen),
                ))
                continue
            if _norm(known) != _norm(seen):
                findings.append(RegistryFinding(
                    "warning", "supplier_field_changed",
                    f"{field}: познато '{known}' → видяно '{seen}' — възможна OCR грешка или реална промяна",
                    field=field, known=str(known), seen=str(seen),
                ))

        update_sql = "UPDATE suppliers SET invoice_count = invoice_count + 1, last_seen = ?"
        params: list = [now]
        for field, value in fills_to_apply.items():
            update_sql += f", {field} = ?"
            params.append(value)
        update_sql += " WHERE eik = ?"
        params.append(eik)
        with self._conn() as c:
            c.execute(update_sql, params)

        if not findings:
            findings.append(RegistryFinding(
                "ok", "supplier_match",
                f"Доставчикът съвпада с регистъра ({existing['invoice_count'] + 1}-та фактура)",
            ))
        return findings

    def all(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM suppliers ORDER BY last_seen DESC")]

    def correct(self, eik: str, **fields) -> None:
        if not fields:
            return
        keys = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [eik]
        with self._conn() as c:
            c.execute(f"UPDATE suppliers SET {keys} WHERE eik = ?", values)

    def enrich(self, inv: InvoiceData) -> tuple[InvoiceData, list[RegistryFinding]]:
        """Replace supplier fields with registry values where known.

        Trusted fields (registry wins on mismatch): name, address, vat_number, iban, bank, bic.
        Volatile fields (extraction wins): mol, phone, email.
        Returns a deep-copied invoice; the input is not mutated.
        """
        enriched = inv.model_copy(deep=True)
        if not enriched.supplier:
            return enriched, []
        eik = _derive_eik(enriched.supplier.eik, enriched.supplier.vat_number)
        if not eik:
            return enriched, []
        known = self.lookup(eik)
        if known is None:
            return enriched, []
        enriched.supplier.eik = eik

        trusted_party = {
            "name": "supplier.name",
            "address": "supplier.address",
            "vat_number": "supplier.vat_number",
        }
        trusted_top = {
            "iban": "iban",
            "bank": "bank",
            "bic": "bic",
        }

        findings: list[RegistryFinding] = []

        for field, label in trusted_party.items():
            known_val = known.get(field)
            seen_val = getattr(enriched.supplier, field)
            if known_val is None:
                continue
            if seen_val is None:
                setattr(enriched.supplier, field, known_val)
                findings.append(RegistryFinding(
                    "ok", "registry_filled",
                    f"{label}: попълнено от регистъра '{known_val}'",
                    field=label, known=str(known_val),
                ))
            elif _norm(known_val) != _norm(seen_val):
                setattr(enriched.supplier, field, known_val)
                findings.append(RegistryFinding(
                    "warning", "registry_override",
                    f"{label}: '{seen_val}' заменено с '{known_val}' от регистъра",
                    field=label, known=str(known_val), seen=str(seen_val),
                ))

        for field, label in trusted_top.items():
            known_val = known.get(field)
            seen_val = getattr(enriched, field)
            if known_val is None:
                continue
            if seen_val is None:
                setattr(enriched, field, known_val)
                findings.append(RegistryFinding(
                    "ok", "registry_filled",
                    f"{label}: попълнено от регистъра '{known_val}'",
                    field=label, known=str(known_val),
                ))
            elif _norm(known_val) != _norm(seen_val):
                setattr(enriched, field, known_val)
                findings.append(RegistryFinding(
                    "warning", "registry_override",
                    f"{label}: '{seen_val}' заменено с '{known_val}' от регистъра",
                    field=label, known=str(known_val), seen=str(seen_val),
                ))

        return enriched, findings


def _norm(v: str) -> str:
    return " ".join(str(v).split()).lower()


def _derive_eik(eik: str | None, vat_number: str | None) -> str | None:
    """ЕИК in BG: 9 digits (legal entity), 10 digits (ЕГН), or 13 digits (БУЛСТАТ)."""
    if eik and eik.strip():
        return eik.strip()
    if vat_number and vat_number.strip().upper().startswith("BG"):
        digits = vat_number.strip()[2:]
        if digits.isdigit() and len(digits) in (9, 10, 13):
            return digits
    return None
