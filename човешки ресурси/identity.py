"""HR identity documents — Gemini 3.1 Flash Lite, same key as logistics."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from docflow.extractors.gemini_extractor import GeminiExtractor
from docflow.pipeline import EXTRACTORS

# Same model the invoice pipeline uses (GeminiExtractor override, not GEMINI_MODEL env).
HR_GEMINI_NAME = "gemini_3.1_flash_lite"
HR_API_KEY_ENV = "GEMINI_API_KEY"

KIND_LABELS = {
    "id_card": "Лична карта",
    "passport": "Паспорт",
    "driving_license": "Шофьорска книжка",
    "residence_permit": "Документ за пребиваване",
    "employment_contract": "Трудов договор",
    "leave_request": "Молба за отпуск",
    "other": "Друг документ",
}


class ExtraField(BaseModel):
    label: str = Field(description="Printed label as on the document")
    value: str = Field(description="Printed value next to that label")


class IdentityDocument(BaseModel):
    document_kind: str | None = Field(
        default=None,
        description=(
            "One of: id_card (лична карта / identity card), passport (паспорт), "
            "driving_license (шофьорска книжка), residence_permit, other"
        ),
    )
    country: str | None = Field(default=None, description="Issuing country, e.g. България / BGR")
    document_number: str | None = Field(default=None, description="Document / card number")
    surname: str | None = Field(default=None, description="Фамилия / Surname")
    given_names: str | None = Field(default=None, description="Имена / Given names")
    patronymic: str | None = Field(default=None, description="Презиме / father's name if printed separately")
    egn: str | None = Field(default=None, description="ЕГН — 10 digits, no spaces")
    nationality: str | None = Field(default=None, description="Гражданство / Nationality")
    date_of_birth: str | None = Field(default=None, description="Дата на раждане, ISO YYYY-MM-DD")
    place_of_birth: str | None = Field(default=None, description="Място на раждане")
    sex: str | None = Field(default=None, description="Пол as printed (М/Ж, M/F, male/female)")
    height_cm: str | None = Field(default=None, description="Ръст if printed")
    eye_color: str | None = Field(default=None, description="Цвят на очите / eye colour as printed")
    can: str | None = Field(default=None, description="CAN / card access number if printed on the card")
    date_of_issue: str | None = Field(default=None, description="Дата на издаване, ISO YYYY-MM-DD")
    date_of_expiry: str | None = Field(default=None, description="Валидна до, ISO YYYY-MM-DD")
    issuing_authority: str | None = Field(default=None, description="Издаващ орган / МВР / authority")
    permanent_address: str | None = Field(default=None, description="Постоянен адрес if on the back or second page")
    mrz: str | None = Field(default=None, description="Full MRZ, lines separated by newline")
    extra_fields: list[ExtraField] = Field(
        default_factory=list,
        description=(
            "Every other labeled field printed on the document that has no dedicated "
            "schema field. Do not skip unusual labels."
        ),
    )
    employer_name: str | None = Field(default=None, description="Работодател / company name on a contract")
    position: str | None = Field(default=None, description="Длъжност / job title")
    contract_number: str | None = Field(default=None, description="Номер на трудов договор")
    leave_type: str | None = Field(default=None, description="Вид отпуск: платен, неплатен, болничен, etc.")
    leave_start: str | None = Field(default=None, description="Начало на отпуск, ISO YYYY-MM-DD")
    leave_end: str | None = Field(default=None, description="Край на отпуск, ISO YYYY-MM-DD")


IDENTITY_PROMPT = """\
You are an expert extractor for Bulgarian and EU identity documents, with Cyrillic (кирилица) as a first-class script.

Read the image or PDF (front, back, or both pages) and fill the schema.

═══ DOCUMENT KIND ═══
- id_card: лична карта / identity card / ID card
- passport: паспорт / passport
- driving_license: шофьорска книжка / driving licence
- residence_permit: карта за пребиваване / residence permit
- employment_contract: трудов договор / employment contract
- leave_request: молба за отпуск / leave request
- other: anything else (still extract every readable field)

For contracts capture employer_name, position, contract_number, and the employee's identity fields if printed.
For leave requests capture leave_type, leave_start, leave_end, and the employee name / ЕГН if printed.

═══ CYRILLIC ═══
Keep names, places, and addresses in the script printed on the document.
Write them as a person would type them: Николай, not НИКОЛАЙ. Do not keep full-word ALL CAPS from the card.
Do not transliterate unless the document itself shows a Latin line — then prefer the original Cyrillic AND put the Latin variant in extra_fields if both are printed.

═══ DATES ═══
Return ISO YYYY-MM-DD. Bulgarian printed dates are often DD.MM.YYYY.

═══ ЕГН ═══
10 digits, no spaces. If OCR is uncertain, still return the best reading; do not invent a number that is not on the document.

═══ MRZ ═══
If a machine-readable zone is visible, copy it fully (all lines). Use it to cross-check document number, dates, and names. Prefer MRZ when visual text is blurry.

═══ BULGARIAN ID CARD — EVERY PRINTED LINE ═══
Front: surname, given names, patronymic (Cyrillic in schema; Latin printed line in extra_fields),
sex, nationality, EGN, date of birth, date of expiry, document number, CAN if printed.
Back: place of birth, the FULL permanent address including обл. / общ. / гр. / ул. / вх. / ет. / ап.
if those labels are printed, height, eye colour, issuing authority, date of issue, full 3-line MRZ.
Do not drop oblast (обл.) or apartment (ап.).

═══ EXTRA FIELDS ═══
Capture every remaining labeled value (eye colour if not in eye_color, blood type, CAN,
place of issue, restrictions on a driving licence, Latin name lines, etc.).
If both Cyrillic and Latin are printed, put Cyrillic in the schema field and the Latin
line in extra_fields as "Фамилия (латиница)", "Имена (латиница)", etc.

Return null only when a field is genuinely absent. Do not hallucinate.
"""

REVIEW_FIELDS: tuple[tuple[str, str], ...] = (
    ("document_kind", "Вид документ"),
    ("country", "Държава"),
    ("document_number", "Номер на документ"),
    ("surname", "Фамилия"),
    ("given_names", "Имена"),
    ("patronymic", "Презиме"),
    ("egn", "ЕГН"),
    ("nationality", "Гражданство"),
    ("date_of_birth", "Дата на раждане"),
    ("place_of_birth", "Място на раждане"),
    ("sex", "Пол"),
    ("height_cm", "Ръст"),
    ("eye_color", "Цвят на очите"),
    ("date_of_issue", "Дата на издаване"),
    ("date_of_expiry", "Валидна до"),
    ("issuing_authority", "Издаващ орган"),
    ("permanent_address", "Постоянен адрес"),
    ("can", "CAN"),
    ("employer_name", "Работодател"),
    ("position", "Длъжност"),
    ("contract_number", "№ договор"),
    ("leave_type", "Вид отпуск"),
    ("leave_start", "Отпуск от"),
    ("leave_end", "Отпуск до"),
    ("mrz", "MRZ"),
)

_EGN_WEIGHTS = (2, 4, 8, 5, 10, 9, 7, 3, 6)
_HUMANIZE_NAME_FIELDS = (
    "surname",
    "given_names",
    "patronymic",
    "country",
    "nationality",
    "place_of_birth",
    "permanent_address",
    "eye_color",
)
_SMALL_WORDS = {
    "на", "и", "от", "по", "в", "във", "за", "при", "към", "с", "със", "или",
    "the", "of", "and",
}
_ADDRESS_ABBR = {
    "гр": "гр.",
    "ул": "ул.",
    "бул": "бул.",
    "пл": "пл.",
    "жк": "ж.к.",
    "обл": "обл.",
    "общ": "общ.",
}
_ACRONYMS = {
    "МВР", "ЕС", "EU", "BG", "BGR", "BGN", "UN", "NATO", "ДАНС", "РЗИ", "МВнР",
}

_TOKEN_RE = re.compile(r"(\s+)")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_BG_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
DATE_FIELDS = (
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "leave_start",
    "leave_end",
)
LABEL_TO_KEY = {label: key for key, label in REVIEW_FIELDS}
KIND_FROM_LABEL = {label.casefold(): key for key, label in KIND_LABELS.items()}
ARCHIVE_DIR = Path(__file__).resolve().parent / "архив"


def egn_checksum_ok(egn: str) -> bool:
    digits = "".join(c for c in (egn or "") if c.isdigit())
    if len(digits) != 10:
        return False
    total = sum(int(d) * w for d, w in zip(digits[:9], _EGN_WEIGHTS))
    check = total % 11
    if check == 10:
        check = 0
    return check == int(digits[9])


def _mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


def _humanize_word(word: str, *, first: bool, style: str) -> str:
    if "-" in word.strip("-"):
        bits = word.split("-")
        return "-".join(
            _humanize_word(bit, first=first and i == 0, style=style)
            for i, bit in enumerate(bits)
        )
    match = re.match(r"^(\W*)(.*?)(\W*)$", word, re.UNICODE)
    if not match:
        return word
    lead, core, trail = match.group(1), match.group(2), match.group(3)
    if not core:
        return word
    key = core.replace(".", "")
    lower = key.casefold()
    if lower in _ADDRESS_ABBR:
        return f"{lead}{_ADDRESS_ABBR[lower]}"
    if key.upper() in _ACRONYMS:
        token = key.upper()
    elif not first and lower in _SMALL_WORDS:
        token = lower
    elif style == "sentence" and not first:
        token = core.lower()
    else:
        token = core.capitalize()
    return f"{lead}{token}{trail}"


def humanize_written(text: str | None, *, style: str = "title") -> str | None:
    """Turn ID-card ALL CAPS into normal writing: НИКОЛАЙ → Николай."""
    if text is None:
        return None
    raw = text.strip()
    if not raw or not _mostly_upper(raw):
        return text
    parts = _TOKEN_RE.split(raw)
    word_i = 0
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        out.append(_humanize_word(part, first=word_i == 0, style=style))
        word_i += 1
    return "".join(out)


def humanize_document(doc: IdentityDocument) -> IdentityDocument:
    doc = _promote_known_extras(doc)
    updates: dict = {}
    extras: list[ExtraField] = []
    seen_labels: set[str] = set()

    def add_extra(label: str, value: str | None) -> None:
        text = (value or "").strip()
        if not text:
            return
        key = label.casefold()
        if key in seen_labels:
            return
        seen_labels.add(key)
        extras.append(ExtraField(label=label, value=text))

    for item in doc.extra_fields or []:
        add_extra(
            humanize_written(item.label, style="sentence") or item.label,
            to_bg_date(item.value) if _looks_like_date(item.value) else (
                humanize_written(item.value, style="title") or item.value
            ),
        )

    for key in _HUMANIZE_NAME_FIELDS:
        value = getattr(doc, key)
        if not value:
            continue
        cyr, latin = split_bilingual(value)
        updates[key] = humanize_written(cyr, style="title") if cyr else cyr
        if latin and key in _LATIN_EXTRA_LABEL:
            latin_text = latin.upper() if latin.upper() in _ACRONYMS else (
                humanize_written(latin, style="title") or latin
            )
            add_extra(_LATIN_EXTRA_LABEL[key], latin_text)
    if doc.issuing_authority:
        cyr, latin = split_bilingual(doc.issuing_authority)
        updates["issuing_authority"] = humanize_written(cyr, style="sentence") if cyr else cyr
        if latin:
            add_extra(_LATIN_EXTRA_LABEL["issuing_authority"], latin)
    for key in ("employer_name", "position", "leave_type"):
        value = getattr(doc, key)
        if value:
            updates[key] = humanize_written(value, style="title")
    if doc.sex:
        updates["sex"] = normalize_sex(doc.sex)
    for key in DATE_FIELDS:
        value = getattr(doc, key)
        if value:
            updates[key] = to_bg_date(value)
    for key, latin in _latin_from_mrz(doc.mrz).items():
        label = _LATIN_EXTRA_LABEL.get(key)
        if label:
            add_extra(label, humanize_written(latin, style="title") or latin)
    updates["extra_fields"] = extras
    return doc.model_copy(update=updates)


def to_bg_date(value: str | None) -> str | None:
    if not value:
        return value
    raw = value.strip()
    match = _BG_DATE_RE.match(raw)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"{day:02d}.{month:02d}.{year}"
    match = _ISO_DATE_RE.match(raw)
    if match:
        year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
        return f"{day:02d}.{month:02d}.{year}"
    return value


def _looks_like_date(value: str | None) -> bool:
    if not value:
        return False
    raw = value.strip()
    return bool(_BG_DATE_RE.match(raw) or _ISO_DATE_RE.match(raw))


def parse_bg_or_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = value.strip()
    match = _BG_DATE_RE.match(raw)
    if match:
        try:
            return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            return None
    match = _ISO_DATE_RE.match(raw)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def document_is_expired(doc: IdentityDocument | None) -> bool:
    if doc is None:
        return False
    expiry = parse_bg_or_iso_date(doc.date_of_expiry)
    return expiry is not None and expiry < date.today()


def normalize_sex(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return value
    male = {"м", "m", "male", "мъж", "man", "м."}
    female = {"ж", "f", "female", "жена", "woman", "w", "ж."}
    for part in re.split(r"[/|,;]+", raw):
        key = part.strip().casefold()
        if key in male:
            return "м"
        if key in female:
            return "ж"
    return humanize_written(raw, style="title") or raw


def _has_cyrillic(text: str) -> bool:
    return any("А" <= ch <= "я" or ch in "ЁёІіЇїЄє" for ch in text)


def _has_latin(text: str) -> bool:
    return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)


def split_bilingual(value: str | None) -> tuple[str | None, str | None]:
    """'СОФИЯ/SOFIA' → ('СОФИЯ', 'SOFIA'). Leave addresses and dates alone."""
    if not value or "/" not in value:
        return value, None
    left, right = value.split("/", 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return value, None
    if _has_cyrillic(left) and _has_latin(right) and " " not in right.split()[0]:
        # Keep long addresses (общ. ... гр.СОФИЯ/SOFIA ул. ...) in one field.
        if any(token in left.casefold() for token in ("ул.", "ул ", "обл", "общ", "вх", "ет.", "ап")):
            return value, None
        return left, right
    return value, None


_LATIN_EXTRA_LABEL = {
    "surname": "Фамилия (латиница)",
    "given_names": "Имена (латиница)",
    "patronymic": "Презиме (латиница)",
    "nationality": "Гражданство (латиница)",
    "place_of_birth": "Място на раждане (латиница)",
    "country": "Държава (латиница)",
    "issuing_authority": "Издаващ орган (латиница)",
    "eye_color": "Цвят на очите (латиница)",
}

_EXTRA_TO_FIELD = (
    (("цвят на очите", "color of eyes", "eye colour", "eye color"), "eye_color"),
    (("can", "card access", "номер на чип"), "can"),
)


def _promote_known_extras(doc: IdentityDocument) -> IdentityDocument:
    extras: list[ExtraField] = []
    updates: dict = {}
    for item in doc.extra_fields or []:
        label = (item.label or "").casefold()
        promoted = False
        for needles, key in _EXTRA_TO_FIELD:
            if any(needle in label for needle in needles) and not getattr(doc, key):
                updates[key] = item.value
                promoted = True
                break
        if not promoted:
            extras.append(item)
    if not updates and extras == list(doc.extra_fields or []):
        return doc
    updates["extra_fields"] = extras
    return doc.model_copy(update=updates)


def _latin_from_mrz(mrz: str | None) -> dict[str, str]:
    if not mrz:
        return {}
    lines = [ln.strip().replace(" ", "") for ln in mrz.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) == 1 and len(lines[0]) >= 90:
        blob = lines[0]
        lines = [blob[0:30], blob[30:60], blob[60:90]]
    if len(lines) < 3:
        return {}
    names = lines[2].rstrip("<")
    surname, _, rest = names.partition("<<")
    bits = [bit for bit in rest.split("<") if bit]
    out: dict[str, str] = {}
    if surname:
        out["surname"] = surname.replace("<", " ").strip()
    if bits:
        out["given_names"] = bits[0]
    if len(bits) > 1:
        out["patronymic"] = bits[1]
    return out


def kind_label(kind: str | None) -> str:
    if not kind:
        return ""
    return KIND_LABELS.get(kind, kind)


def _hr_gemini() -> GeminiExtractor:
    for extractor in EXTRACTORS:
        if extractor.name == HR_GEMINI_NAME:
            return extractor
    return GeminiExtractor(model="gemini-3.1-flash-lite", name=HR_GEMINI_NAME)


def extract_identity(path: Path) -> IdentityDocument:
    parsed, _, _ = _hr_gemini().extract_structured(path, IdentityDocument, IDENTITY_PROMPT)
    return parsed


def identity_review_pairs(doc: IdentityDocument) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in REVIEW_FIELDS:
        value = getattr(doc, key)
        if key == "document_kind":
            value = kind_label(value) or value
        if value is None or value == "":
            display = "—"
        else:
            display = str(value)
        rows.append((label, display))
    return rows


@dataclass
class HrResult:
    name: str
    doc: IdentityDocument | None = None
    error: str | None = None
    egn_ok: bool | None = None
    extra: list[ExtraField] = field(default_factory=list)


def build_hr_result(name: str, doc: IdentityDocument) -> HrResult:
    doc = humanize_document(doc)
    egn_ok = None
    if doc.egn:
        egn_ok = egn_checksum_ok(doc.egn)
    return HrResult(name=name, doc=doc, egn_ok=egn_ok, extra=list(doc.extra_fields or []))


def identity_field_rows(
    result: HrResult,
    *,
    include_empty: bool = True,
    include_egn_check: bool = True,
    include_mrz: bool = True,
) -> list[dict[str, str]]:
    """One row per field: Файл | Поле | Стойност — same layout as extra fields."""
    rows: list[dict[str, str]] = []
    if result.error or result.doc is None:
        rows.append({
            "Файл": result.name,
            "Поле": "Грешка",
            "Стойност": result.error or "няма данни",
        })
        return rows
    for label, value in identity_review_pairs(result.doc):
        if not include_mrz and label == "MRZ":
            continue
        text = "" if value == "—" else value
        if not include_empty and not text:
            continue
        rows.append({"Файл": result.name, "Поле": label, "Стойност": text})
    if include_egn_check and result.egn_ok is not None:
        rows.append({
            "Файл": result.name,
            "Поле": "ЕГН валидно",
            "Стойност": "да" if result.egn_ok else "не",
        })
    return rows


def identity_to_excel(results: list[HrResult]) -> bytes:
    import pandas as pd

    columns = ["Файл", "Поле", "Стойност"]
    rows: list[dict[str, str]] = []
    extra_rows: list[dict[str, str]] = []
    for r in results:
        rows.extend(identity_field_rows(r))
        for extra in r.extra:
            extra_rows.append({"Файл": r.name, "Поле": extra.label, "Стойност": extra.value})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=columns).to_excel(writer, index=False, sheet_name="Документи")
        pd.DataFrame(extra_rows, columns=columns).to_excel(
            writer, index=False, sheet_name="Допълнителни полета",
        )
    return buf.getvalue()


def kind_from_label(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if raw in KIND_LABELS:
        return raw
    return KIND_FROM_LABEL.get(raw.casefold(), raw)


def person_full_name(doc: IdentityDocument | None) -> str:
    if doc is None:
        return ""
    return " ".join(
        part for part in (doc.given_names, doc.patronymic, doc.surname) if part
    )


def result_search_blob(result: HrResult) -> str:
    parts = [result.name, result.error or ""]
    if result.doc:
        parts.extend(str(getattr(result.doc, key) or "") for key, _ in REVIEW_FIELDS)
        parts.extend(f"{item.label} {item.value}" for item in (result.doc.extra_fields or []))
    return " ".join(parts).casefold()


def result_matches(result: HrResult, query: str) -> bool:
    needle = query.strip().casefold()
    if not needle:
        return True
    return needle in result_search_blob(result)


def merge_identity_docs(front: IdentityDocument, back: IdentityDocument) -> IdentityDocument:
    data = front.model_dump()
    other = back.model_dump()
    for key, value in other.items():
        if key == "extra_fields":
            continue
        current = data.get(key)
        if value in (None, "", []):
            continue
        if current in (None, "", []):
            data[key] = value
            continue
        if key == "permanent_address" and len(str(value)) > len(str(current)):
            data[key] = value
    seen: set[tuple[str, str]] = set()
    extras: list[dict] = []
    for item in list(data.get("extra_fields") or []) + list(other.get("extra_fields") or []):
        pair = (str(item.get("label") or ""), str(item.get("value") or ""))
        if pair in seen:
            continue
        seen.add(pair)
        extras.append(item)
    data["extra_fields"] = extras
    return IdentityDocument.model_validate(data)


def merge_hr_results(first: HrResult, second: HrResult) -> HrResult:
    if first.doc is None:
        return second
    if second.doc is None:
        return first
    merged = merge_identity_docs(first.doc, second.doc)
    name = person_full_name(merged) or f"{first.name} + {second.name}"
    return build_hr_result(name, merged)


def pair_front_back(results: list[HrResult]) -> list[HrResult]:
    paired: list[HrResult] = []
    index = 0
    while index < len(results):
        current = results[index]
        nxt = results[index + 1] if index + 1 < len(results) else None
        if nxt is not None and current.doc is not None and nxt.doc is not None:
            front, back, _ = order_front_back(current, nxt)
            paired.append(merge_hr_results(front, back))
            index += 2
            continue
        paired.append(current)
        index += 1
    return paired


def card_back_score(doc: IdentityDocument | None) -> int:
    """Higher score → more likely the reverse of an ID card."""
    if doc is None:
        return 0
    score = 0
    if doc.mrz:
        score += 3
    if doc.permanent_address:
        score += 3
    if doc.height_cm:
        score += 1
    if doc.given_names or doc.surname:
        score -= 2
    return score


def order_front_back(
    first: HrResult, second: HrResult,
) -> tuple[HrResult, HrResult, bool]:
    """Put the photo side first. Returns (front, back, swapped)."""
    if card_back_score(first.doc) > card_back_score(second.doc):
        return second, first, True
    return first, second, False


def upright_card_preview(png: bytes | None) -> bytes | None:
    """ID cards are landscape — straighten phone photos that landed on their side."""
    if not png:
        return None
    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(png))
    im.load()
    im = ImageOps.exif_transpose(im) or im
    if im.height > im.width:
        im = im.rotate(90, expand=True)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def rotate_preview(png: bytes | None, degrees: int = 90) -> bytes | None:
    if not png:
        return None
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    im.load()
    im = im.rotate(degrees, expand=True)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def apply_field_edits(result: HrResult, rows: list[dict[str, str]]) -> HrResult:
    if result.doc is None:
        return result
    updates: dict = {}
    extras: list[ExtraField] = []
    extra_rows_present = False
    for row in rows:
        label = str(row.get("Поле") or "").strip()
        raw_value = row.get("Стойност")
        value = "" if raw_value is None else str(raw_value).strip()
        if not label or label == "ЕГН валидно":
            continue
        key = LABEL_TO_KEY.get(label)
        if key is None:
            extra_rows_present = True
            if value:
                extras.append(ExtraField(label=label, value=value))
            continue
        if key == "document_kind":
            updates[key] = kind_from_label(value) if value else None
        elif key in DATE_FIELDS:
            updates[key] = to_bg_date(value) if value else None
        elif key == "sex":
            updates[key] = normalize_sex(value) if value else None
        else:
            updates[key] = value or None
    if extra_rows_present:
        updates["extra_fields"] = extras
    doc = result.doc.model_copy(update=updates)
    return build_hr_result(result.name, doc)


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "", text)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80] or "HR_документи"


def excel_filename(results: list[HrResult]) -> str:
    names = [person_full_name(item.doc) for item in results if item.doc]
    names = [sanitize_filename(item) for item in names if item]
    if len(names) == 1:
        return f"{names[0]}.xlsx"
    if names:
        return f"{names[0]}_и_още_{len(results)}.xlsx"
    return "HR_документи.xlsx"


def _result_to_dict(result: HrResult) -> dict:
    return {
        "name": result.name,
        "error": result.error,
        "doc": result.doc.model_dump() if result.doc else None,
    }


def result_from_dict(payload: dict) -> HrResult:
    raw_doc = payload.get("doc")
    if raw_doc:
        return build_hr_result(str(payload.get("name") or "документ"), IdentityDocument.model_validate(raw_doc))
    return HrResult(name=str(payload.get("name") or "документ"), error=payload.get("error"))


def save_archive(results: list[HrResult]) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_filename(person_full_name(results[0].doc) if results else "") or "пакет"
    path = ARCHIVE_DIR / f"{stamp}_{slug}.json"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "results": [_result_to_dict(item) for item in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_archive() -> list[Path]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(ARCHIVE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_archive(path: Path) -> list[HrResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    return [result_from_dict(item) for item in rows or []]


def search_archive(query: str) -> list[tuple[Path, HrResult]]:
    hits: list[tuple[Path, HrResult]] = []
    for path in list_archive():
        try:
            for result in load_archive(path):
                if result_matches(result, query):
                    hits.append((path, result))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return hits


def rows_to_clipboard(rows: list[dict[str, str]], *, header: bool = True) -> str:
    """TSV so several rows paste into Excel / Word as a table."""
    lines: list[str] = []
    if header:
        lines.append("Поле\tСтойност")
    for row in rows:
        field = str(row.get("Поле") or "").replace("\t", " ").replace("\n", " ")
        value = str(row.get("Стойност") or "").replace("\t", " ").replace("\n", " ")
        lines.append(f"{field}\t{value}")
    return "\n".join(lines) + ("\n" if lines else "")
