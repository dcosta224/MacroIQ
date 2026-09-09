"""Resolve recipe ingredient amounts to grams via unit conversion or USDA portions.

Mass units convert directly with unit_convert. Volume units look up
usda.food_portion rows for the matched fdc_id and scale gram_weight.
Count/discrete units match portion modifier/description labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from amount_kind import (
    classify_amount_kind,
    classify_from_parsed_row,
    infer_count_query,
    missing_quantity,
    normalize_count_unit,
    _normalize_parsed_unit,
)
from db import connect
from recipe_parse_rules import normalize_unit, rule_parse_fields
from unit_convert import (
    UnitConversionError,
    VOLUME_TO_ML,
    canonical_volume_token,
    convert_mass,
    convert_volume,
    normalize_mass_unit,
    normalize_volume_unit,
    unit_kind,
)
from usda_volume_units import MASS_PATTERN, VOLUME_PATTERN, text_has_fluid_ounce, text_has_volume

# measure_unit.id → canonical volume unit (USDA reference table)
STRUCTURED_VOLUME_MEASURE_UNIT_IDS: dict[int, str] = {
    1000: "cup",
    1001: "tablespoon",
    1002: "teaspoon",
    1003: "liter",
    1004: "milliliter",
    1008: "pint",
    1045: "quart",
}

MASS_MEASURE_UNIT_IDS: dict[int, str] = {
    1030: "pound",
    1038: "ounce",
}

# Backward-compatible alias
STRUCTURED_MEASURE_UNIT_IDS = STRUCTURED_VOLUME_MEASURE_UNIT_IDS

MEASURE_UNIT_NAME_TO_CANONICAL: dict[str, str] = {
    "cup": "cup",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "teaspoon": "teaspoon",
    "liter": "liter",
    "milliliter": "milliliter",
    "pint": "pint",
    "pints": "pint",
    "quart": "quart",
    "quarts": "quart",
    "fl oz": "fluid_ounce",
    "fluid ounce": "fluid_ounce",
    "fluid ounces": "fluid_ounce",
    "gallon": "gallon",
    "gallons": "gallon",
}

NON_INFORMATIVE_MEASURE_UNITS = frozenset(
    {
        "undetermined",
        "racc",
    }
)

MASS_MEASURE_UNIT_NAMES = frozenset(
    {
        "oz",
        "lb",
        "paired cooked w",
        "paired raw w",
        "dripping w",
        "orig ckd g",
        "orig rw g",
    }
)

# Regex capture group → unit_convert canonical (units supported for gram math)
VOLUME_TOKEN_TO_CANONICAL: dict[str, str] = {
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tablespoon",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "tsp": "teaspoon",
    "teaspoon": "teaspoon",
    "teaspoons": "teaspoon",
    "fl oz": "fluid_ounce",
    "fluid ounce": "fluid_ounce",
    "fluid ounces": "fluid_ounce",
    "liter": "liter",
    "litre": "liter",
    "liters": "liter",
    "litres": "liter",
    "ml": "milliliter",
    "milliliter": "milliliter",
    "milliliters": "milliliter",
    "pint": "pint",
    "pints": "pint",
    "quart": "quart",
    "quarts": "quart",
    "gallon": "gallon",
    "gallons": "gallon",
    "cc": "milliliter",
    "cubic centimeter": "milliliter",
    "cubic centimeters": "milliliter",
    "cubic cm": "milliliter",
    "cubic inch": "cubic_inch",
    "cubic inches": "cubic_inch",
}

SUPPORTED_VOLUME_UNITS = frozenset(VOLUME_TO_ML.keys())

LEADING_AMOUNT_RE = re.compile(
    r"^\s*([\d]+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+",
    re.IGNORECASE,
)
FNDDS_CODE_ONLY_RE = re.compile(r"^\d+$")
MASS_TOKEN_RE = re.compile(r"\b(oz|ounce|ounces|lb|lbs|pound|pounds|g\b|gram|grams|kg)\b", re.I)
COUNT_DESC_RE = re.compile(
    r"^\s*(?:[\d]+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+([a-z][a-z\s,/-]{0,40}?)\s*$",
    re.I,
)

SENTINEL_FDC_ID = 999_000_001
WATER_SENTINEL_FDC_ID = 999_000_002

PORTION_INDEX_SQL = """
SELECT fp.id, fp.fdc_id, fp.amount, fp.modifier, fp.portion_description,
       fp.gram_weight, fp.seq_num, fp.measure_unit_id, mu.name AS measure_unit_name,
       fp.data_points
FROM usda.food_portion fp
LEFT JOIN usda.measure_unit mu ON mu.id = fp.measure_unit_id
WHERE fp.gram_weight > 0 AND fp.fdc_id IS NOT NULL
ORDER BY fp.fdc_id, fp.seq_num
"""


class PortionGramError(ValueError):
    """Raised when gram resolution fails unexpectedly."""


@dataclass(frozen=True)
class PortionCandidate:
    portion_id: int
    fdc_id: int
    ref_amount: float
    ref_unit: str
    gram_weight: float
    seq_num: int
    data_points: int
    modifier: str
    portion_description: str
    fndds_code_only: bool
    has_readable_text: bool

    def score_for(self, recipe_unit: str) -> float:
        recipe_norm = normalize_volume_unit(recipe_unit)
        score = 0.0
        if self.ref_unit == recipe_norm:
            score += 100.0
        if self.has_readable_text and not self.fndds_code_only:
            score += 10.0
        if not self.fndds_code_only:
            score += 5.0
        score += min(self.data_points, 50)
        score -= self.seq_num * 0.01
        return score


@dataclass(frozen=True)
class CountPortionCandidate:
    portion_id: int
    fdc_id: int
    ref_amount: float
    count_label: str
    gram_weight: float
    seq_num: int
    data_points: int
    modifier: str
    portion_description: str

    def score_for(self, query_tokens: list[str]) -> float:
        label = self.count_label.lower()
        modifier = (self.modifier or "").lower()
        desc = (self.portion_description or "").lower()
        combined = f"{label} {modifier} {desc}"
        label_tokens = set(re.split(r"[\s,/-]+", label))
        score = 0.0
        size_tokens = {"small", "medium", "large", "extra-large", "jumbo"}
        for q in query_tokens:
            q_lower = q.lower().replace("extra large", "extra-large")
            if q_lower in size_tokens:
                if q_lower in modifier or q_lower.replace("-", " ") in modifier:
                    score += 70.0
                elif q_lower in combined:
                    score += 45.0
                continue
            if label == q_lower:
                score += 100.0
            elif q_lower in label_tokens:
                score += 80.0
            elif re.search(rf"\b{re.escape(q_lower)}\b", label):
                score += 40.0
            elif re.search(rf"\b{re.escape(q_lower)}\b", modifier) or re.search(
                rf"\b{re.escape(q_lower)}\b", desc
            ):
                score += 35.0
        if score > 0:
            score += min(self.data_points, 50)
            score -= self.seq_num * 0.01
        return score


@dataclass(frozen=True)
class MassPortionCandidate:
    portion_id: int
    fdc_id: int
    ref_amount: float
    mass_unit: str
    gram_weight: float
    seq_num: int
    data_points: int
    modifier: str
    portion_description: str


@dataclass(frozen=True)
class PortionCapabilitySets:
    volume_fdc_ids: frozenset[int]
    count_fdc_ids: frozenset[int]


@dataclass(frozen=True)
class PortionGramResult:
    grams: float | None
    status: str
    unit_kind: str | None
    portion_id: int | None = None
    portion_ref_amount: float | None = None
    portion_ref_unit: str | None = None
    method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and val != val:
        return True
    if val == "":
        return True
    return False


def _optional_int(val: object, default: int = 0) -> int:
    if _is_missing(val):
        return default
    return int(val)


def _parse_data_points(raw: object) -> int:
    if _is_missing(raw):
        return 0
    text = str(raw).strip()
    if not text or not text.isdigit():
        return 0
    return int(text)


def _canonical_from_volume_token(token: str) -> str | None:
    key = token.lower().replace(".", "").strip()
    key = re.sub(r"\s+", " ", key)
    mapped = VOLUME_TOKEN_TO_CANONICAL.get(key)
    if mapped:
        return mapped
    return canonical_volume_token(token)


def _extract_volume_unit_from_text(*parts: str) -> str | None:
    combined = " ".join(p for p in parts if p)
    match = VOLUME_PATTERN.search(combined)
    if not match:
        return None
    return _canonical_from_volume_token(match.group(1))


def _portion_text_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    modifier = str(row.get("modifier") or "").strip()
    portion_description = str(row.get("portion_description") or "").strip()
    measure_unit_name = str(row.get("measure_unit_name") or "").strip()
    return modifier, portion_description, measure_unit_name


def _informative_measure_unit(name: str) -> bool:
    if not name:
        return False
    return name.strip().lower() not in NON_INFORMATIVE_MEASURE_UNITS


def _portion_fields_to_search(row: dict[str, Any]) -> list[str]:
    """Search measure_unit, modifier, portion_description (skip FNDDS-only codes)."""
    modifier, portion_description, measure_unit_name = _portion_text_fields(row)
    fields: list[str] = []
    if _informative_measure_unit(measure_unit_name):
        fields.append(measure_unit_name)
    if modifier and not FNDDS_CODE_ONLY_RE.match(modifier):
        fields.append(modifier)
    if portion_description:
        fields.append(portion_description)
    return fields


def _field_has_mass(text: str) -> bool:
    if not text:
        return False
    if text_has_fluid_ounce(text):
        return False
    return bool(MASS_TOKEN_RE.search(text) or MASS_PATTERN.search(text))


def _volume_unit_from_single_field(text: str) -> str | None:
    if not text:
        return None
    token = text.lower().strip()
    mapped = MEASURE_UNIT_NAME_TO_CANONICAL.get(token)
    if mapped and mapped in SUPPORTED_VOLUME_UNITS:
        return mapped
    try:
        vol = normalize_volume_unit(text)
        if vol in SUPPORTED_VOLUME_UNITS:
            return vol
    except UnitConversionError:
        pass
    extracted = _extract_volume_unit_from_text(text)
    if extracted and extracted in SUPPORTED_VOLUME_UNITS:
        return extracted
    return None


def _resolve_volume_from_row(row: dict[str, Any]) -> tuple[str | None, float | None]:
    """Find volume unit by searching measure_unit, modifier, and portion_description."""
    measure_unit_id = row.get("measure_unit_id")
    if not _is_missing(measure_unit_id):
        mid = int(measure_unit_id)
        if mid in STRUCTURED_VOLUME_MEASURE_UNIT_IDS:
            return STRUCTURED_VOLUME_MEASURE_UNIT_IDS[mid], None

    for field in _portion_fields_to_search(row):
        unit = _volume_unit_from_single_field(field)
        if unit:
            return unit, _parse_leading_amount(field)

    modifier, portion_description, measure_unit_name = _portion_text_fields(row)
    if text_has_volume(modifier, portion_description, measure_unit_name):
        unit = _extract_volume_unit_from_text(
            modifier, portion_description, measure_unit_name
        )
        if unit and unit in SUPPORTED_VOLUME_UNITS:
            amt = _parse_leading_amount(portion_description) or _parse_leading_amount(modifier)
            return unit, amt

    return None, None


def _count_label_from_single_field(text: str) -> str | None:
    if not text or FNDDS_CODE_ONLY_RE.match(text):
        return None
    if text_has_volume(text) or _field_has_mass(text):
        return None

    norm = normalize_unit(text)
    if norm:
        return norm

    match = COUNT_DESC_RE.match(text)
    if match:
        return match.group(1).strip().lower()

    text_low = text.strip().lower()
    if text_low and len(text_low) <= 80:
        return text_low
    return None


def _resolve_count_label_from_row(row: dict[str, Any]) -> str | None:
    """Find count/discrete label by searching measure_unit, modifier, portion_description."""
    for field in _portion_fields_to_search(row):
        label = _count_label_from_single_field(field)
        if label:
            return label
    return None


def _resolve_mass_from_row(row: dict[str, Any]) -> bool:
    measure_unit_id = row.get("measure_unit_id")
    if not _is_missing(measure_unit_id) and int(measure_unit_id) in MASS_MEASURE_UNIT_IDS:
        return True

    modifier, portion_description, measure_unit_name = _portion_text_fields(row)
    if measure_unit_name.lower() in MASS_MEASURE_UNIT_NAMES:
        return True

    for field in _portion_fields_to_search(row):
        if _field_has_mass(field) and not text_has_volume(field):
            return True
    return False


def classify_food_portion_row(row: dict[str, Any]) -> str:
    """Classify a food_portion row as volume, count, mass, or other."""
    gram_weight = float(row.get("gram_weight") or 0)
    if gram_weight <= 0:
        return "other"
    if normalize_portion_row(row) is not None:
        return "volume"
    if normalize_count_portion_row(row) is not None:
        return "count"
    if _resolve_mass_from_row(row):
        return "mass"
    return "other"


def _parse_leading_amount(text: str) -> float | None:
    if not text:
        return None
    match = LEADING_AMOUNT_RE.match(str(text).strip())
    if not match:
        return None
    raw = match.group(1)
    if " " in raw and "/" in raw:
        whole, frac = raw.split(None, 1)
        try:
            from fractions import Fraction

            return float(whole) + float(Fraction(frac))
        except Exception:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_portion_row(row: dict[str, Any]) -> PortionCandidate | None:
    """Map a food_portion DB row to a volume PortionCandidate, or None."""
    gram_weight = float(row["gram_weight"])
    if gram_weight <= 0:
        return None

    amount = row.get("amount")
    ref_amount = float(amount) if amount not in (None, "") else 1.0
    if ref_amount <= 0:
        ref_amount = 1.0

    modifier, portion_description, measure_unit_name = _portion_text_fields(row)

    ref_unit, text_amount = _resolve_volume_from_row(row)
    if ref_unit is None or ref_unit not in SUPPORTED_VOLUME_UNITS:
        return None
    if text_amount is not None:
        ref_amount = text_amount

    fndds_code_only = bool(modifier and FNDDS_CODE_ONLY_RE.match(modifier))
    has_readable = bool(
        (modifier and not fndds_code_only)
        or portion_description
        or measure_unit_name
    )

    return PortionCandidate(
        portion_id=int(row["id"]),
        fdc_id=int(row["fdc_id"]),
        ref_amount=ref_amount,
        ref_unit=ref_unit,
        gram_weight=gram_weight,
        seq_num=_optional_int(row.get("seq_num")),
        data_points=_parse_data_points(row.get("data_points")),
        modifier=modifier,
        portion_description=portion_description,
        fndds_code_only=fndds_code_only,
        has_readable_text=has_readable,
    )


def _row_is_volume_candidate(row: dict[str, Any]) -> bool:
    return normalize_portion_row(row) is not None


def normalize_count_portion_row(row: dict[str, Any]) -> CountPortionCandidate | None:
    """Map a food_portion DB row to a count PortionCandidate, or None."""
    if _row_is_volume_candidate(row):
        return None

    gram_weight = float(row["gram_weight"])
    if gram_weight <= 0:
        return None

    modifier, portion_description, measure_unit_name = _portion_text_fields(row)

    if text_has_volume(modifier, portion_description, measure_unit_name):
        return None
    if _resolve_mass_from_row(row):
        return None

    count_label = _resolve_count_label_from_row(row)
    if not count_label:
        return None

    amount = row.get("amount")
    ref_amount = float(amount) if amount not in (None, "") else 1.0
    if ref_amount <= 0:
        ref_amount = 1.0

    return CountPortionCandidate(
        portion_id=int(row["id"]),
        fdc_id=int(row["fdc_id"]),
        ref_amount=ref_amount,
        count_label=count_label,
        gram_weight=gram_weight,
        seq_num=_optional_int(row.get("seq_num")),
        data_points=_parse_data_points(row.get("data_points")),
        modifier=modifier,
        portion_description=portion_description,
    )


def _mass_unit_from_text(text: str) -> str | None:
    if not text or text_has_volume(text):
        return None
    match = MASS_TOKEN_RE.search(text)
    if match:
        token = match.group(1).lower()
        try:
            return normalize_mass_unit(token)
        except UnitConversionError:
            pass
        mapped = normalize_unit(token)
        if mapped:
            try:
                return normalize_mass_unit(mapped)
            except UnitConversionError:
                return mapped
    return None


def _resolve_mass_unit_from_row(row: dict[str, Any]) -> str | None:
    measure_unit_id = row.get("measure_unit_id")
    if not _is_missing(measure_unit_id):
        mid = int(measure_unit_id)
        if mid in MASS_MEASURE_UNIT_IDS:
            return MASS_MEASURE_UNIT_IDS[mid]

    modifier, portion_description, measure_unit_name = _portion_text_fields(row)
    if measure_unit_name.lower() in MASS_MEASURE_UNIT_NAMES:
        name = measure_unit_name.lower()
        if name == "oz":
            return "ounce"
        if name in {"lb", "paired cooked w", "paired raw w"}:
            return "pound"

    for field in _portion_fields_to_search(row):
        unit = _mass_unit_from_text(field)
        if unit:
            return unit

    for field in (modifier, portion_description):
        unit = _mass_unit_from_text(field or "")
        if unit:
            return unit
    return None


def normalize_mass_portion_row(row: dict[str, Any]) -> MassPortionCandidate | None:
    """Map a food_portion DB row to a mass PortionCandidate, or None."""
    if normalize_portion_row(row) is not None:
        return None
    if normalize_count_portion_row(row) is not None:
        return None

    gram_weight = float(row["gram_weight"])
    if gram_weight <= 0:
        return None
    if not _resolve_mass_from_row(row):
        return None

    mass_unit = _resolve_mass_unit_from_row(row)
    if not mass_unit:
        return None

    amount = row.get("amount")
    ref_amount = float(amount) if amount not in (None, "") else 1.0
    if ref_amount <= 0:
        ref_amount = 1.0

    modifier, portion_description, _ = _portion_text_fields(row)
    return MassPortionCandidate(
        portion_id=int(row["id"]),
        fdc_id=int(row["fdc_id"]),
        ref_amount=ref_amount,
        mass_unit=mass_unit,
        gram_weight=gram_weight,
        seq_num=_optional_int(row.get("seq_num")),
        data_points=_parse_data_points(row.get("data_points")),
        modifier=modifier,
        portion_description=portion_description,
    )


def build_count_portion_index(conn) -> dict[int, list[CountPortionCandidate]]:
    """Load count-usable food_portion rows keyed by fdc_id."""
    index: dict[int, list[CountPortionCandidate]] = {}
    with conn.cursor() as cur:
        cur.execute(PORTION_INDEX_SQL)
        columns = [desc[0] for desc in cur.description]
        for db_row in cur.fetchall():
            row = dict(zip(columns, db_row, strict=True))
            candidate = normalize_count_portion_row(row)
            if candidate is None:
                continue
            index.setdefault(candidate.fdc_id, []).append(candidate)
    return index


def build_portion_capability_sets(conn) -> PortionCapabilitySets:
    """fdc_id sets with volume or count portion rows."""
    volume = set(build_portion_index(conn).keys())
    count = set(build_count_portion_index(conn).keys())
    return PortionCapabilitySets(
        volume_fdc_ids=frozenset(volume),
        count_fdc_ids=frozenset(count),
    )


def pick_best_count_portion(
    candidates: list[CountPortionCandidate],
    query_tokens: list[str],
) -> CountPortionCandidate | None:
    if not candidates or not query_tokens:
        return None
    scored = [(c, c.score_for(query_tokens)) for c in candidates]
    best_score = max(s for _, s in scored)
    if best_score <= 0:
        return None
    top = [c for c, s in scored if s == best_score]
    return min(top, key=lambda c: (c.seq_num, -c.data_points, c.portion_id))


SIZE_ONLY_LABELS = frozenset({"small", "medium", "large", "extra-large", "jumbo", "slice"})
GENERIC_SIZE_LADDER = ("medium", "large", "small", "extra-large", "jumbo")
CONTAINER_UNIT_TOKENS = frozenset(
    {"can", "cans", "jar", "jars", "box", "boxes", "package", "pkg", "bottle", "bottles"}
)


def _candidate_matches_size_token(candidate: CountPortionCandidate, size: str) -> bool:
    text = f"{candidate.count_label} {candidate.modifier} {candidate.portion_description}".lower()
    key = size.lower().replace("extra large", "extra-large")
    return key in text or key.replace("-", " ") in text


def _is_size_only_catalog(candidates: list[CountPortionCandidate]) -> bool:
    if not candidates:
        return False
    for c in candidates:
        label = (c.count_label or "").lower().strip()
        if not label:
            continue
        if label not in SIZE_ONLY_LABELS:
            return False
    return True


def pick_count_portion_with_fallbacks(
    candidates: list[CountPortionCandidate],
    query_tokens: list[str],
    *,
    size_tokens: list[str] | None = None,
) -> CountPortionCandidate | None:
    """Exact token match, then recipe size, then generic count size ladder."""
    if not candidates:
        return None

    exact = pick_best_count_portion(candidates, query_tokens)
    if exact is not None:
        return exact

    sizes = size_tokens or []
    for size in sizes:
        matches = [c for c in candidates if _candidate_matches_size_token(c, size)]
        if matches:
            return min(matches, key=lambda c: (c.seq_num, -c.data_points, c.portion_id))

    if sizes:
        return None

    if _is_size_only_catalog(candidates):
        for size in GENERIC_SIZE_LADDER:
            for c in candidates:
                if _candidate_matches_size_token(c, size):
                    return c

    return None


def pick_container_mass_portion(
    raw_rows: list[dict[str, Any]],
    unit_tokens: list[str],
) -> tuple[int, float] | None:
    """Match container unit (can/jar/box) to mass-class rows with container modifier."""
    wanted = {t.lower().rstrip(".") for t in unit_tokens} & CONTAINER_UNIT_TOKENS
    if not wanted:
        return None
    for row in raw_rows:
        gw = float(row.get("gram_weight") or 0)
        if gw <= 0:
            continue
        mod = str(row.get("modifier") or "").lower()
        if any(tok.rstrip("s") in mod or tok in mod for tok in wanted):
            if classify_food_portion_row(row) in ("mass", "other"):
                return int(row["id"]), gw
    return None


def pick_whole_item_mass_proxy(
    raw_rows: list[dict[str, Any]],
    *,
    quantity: float,
    name: str | None,
    size_tokens: list[str] | None = None,
) -> PortionGramResult | None:
    """Use whole-item mass modifier rows as count proxy (e.g. 1 medium eggplant)."""
    best_row: dict[str, Any] | None = None
    best_score = -1.0
    name_low = (name or "").lower()

    for row in raw_rows:
        gw = float(row.get("gram_weight") or 0)
        if gw <= 0 or classify_food_portion_row(row) != "mass":
            continue
        mod = str(row.get("modifier") or "").lower()
        if not mod:
            continue
        score = 0.0
        if name_low and name_low in mod:
            score += 10.0
        if any(k in mod for k in ("unpeeled", "peeled", "whole", "approx", "yield")):
            score += 5.0
        for sz in size_tokens or []:
            if sz in mod or sz.replace("-", " ") in mod:
                score += 20.0
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None and float(quantity) == 1.0:
        for row in raw_rows:
            gw = float(row.get("gram_weight") or 0)
            if gw <= 0 or classify_food_portion_row(row) != "mass":
                continue
            mod = str(row.get("modifier") or "").lower()
            if name_low and name_low in mod:
                best_row = row
                break

    if best_row is None:
        return None

    grams = round(float(quantity) * float(best_row["gram_weight"]), 4)
    return PortionGramResult(
        grams=grams,
        status="ok_count_portion",
        unit_kind="count",
        portion_id=int(best_row["id"]),
        method=f"whole-item mass proxy via portion#{best_row['id']}",
    )


def build_portion_index(conn) -> dict[int, list[PortionCandidate]]:
    """Load all volume-usable food_portion rows keyed by fdc_id."""
    index: dict[int, list[PortionCandidate]] = {}
    with conn.cursor() as cur:
        cur.execute(PORTION_INDEX_SQL)
        columns = [desc[0] for desc in cur.description]
        for db_row in cur.fetchall():
            row = dict(zip(columns, db_row, strict=True))
            candidate = normalize_portion_row(row)
            if candidate is None:
                continue
            index.setdefault(candidate.fdc_id, []).append(candidate)
    return index


def pick_best_portion(
    candidates: list[PortionCandidate],
    recipe_unit: str,
) -> PortionCandidate | None:
    if not candidates:
        return None
    recipe_norm = normalize_volume_unit(recipe_unit)
    readable = [c for c in candidates if not c.fndds_code_only or c.has_readable_text]
    pool = readable if readable else candidates
    exact = [c for c in pool if c.ref_unit == recipe_norm]
    if exact:
        pool = exact
    return max(pool, key=lambda c: c.score_for(recipe_unit))


def infer_matched_portion_id(
    fdc_id: int,
    *,
    amount_kind: str,
    unit: str | None = None,
    quantity: float | None = None,
    query_tokens: list[str] | None = None,
    portion_index: dict[int, list[PortionCandidate]] | None = None,
    count_portion_index: dict[int, list[CountPortionCandidate]] | None = None,
) -> int | None:
    """Pick best USDA portion for rules-based gram conversion."""
    fdc_int = int(fdc_id)
    if amount_kind == "volume" and unit and portion_index is not None:
        candidates = portion_index.get(fdc_int, [])
        portion = pick_best_portion(candidates, str(unit))
        if portion is None:
            return None
        if quantity is not None:
            result = _grams_from_volume_candidate(float(quantity), str(unit), portion)
            if result is None or result.grams is None:
                return None
        return portion.portion_id
    if amount_kind == "count" and count_portion_index is not None and query_tokens:
        candidates = count_portion_index.get(fdc_int, [])
        portion = pick_best_count_portion(candidates, list(query_tokens))
        return portion.portion_id if portion is not None else None
    return None


def resolve_matched_portion_id(
    fdc_id: int | None,
    matched_portion_id: int | None,
    *,
    amount_kind: str,
    unit: str | None = None,
    quantity: float | None = None,
    query_tokens: list[str] | None = None,
    portion_index: dict[int, list[PortionCandidate]] | None = None,
    count_portion_index: dict[int, list[CountPortionCandidate]] | None = None,
) -> tuple[int | None, bool]:
    """Validate judge portion pick; infer from indexes when missing or non-convertible."""
    if fdc_id is None or amount_kind not in ("volume", "count"):
        return matched_portion_id, False

    inferred = infer_matched_portion_id(
        int(fdc_id),
        amount_kind=amount_kind,
        unit=unit,
        quantity=quantity,
        query_tokens=query_tokens,
        portion_index=portion_index,
        count_portion_index=count_portion_index,
    )

    if matched_portion_id is None:
        return inferred, inferred is not None

    pid = int(matched_portion_id)
    if amount_kind == "volume" and unit and portion_index is not None:
        candidates = portion_index.get(int(fdc_id), [])
        chosen = next((c for c in candidates if c.portion_id == pid), None)
        if chosen is None:
            return inferred, inferred is not None
        qty = 1.0 if quantity is None else float(quantity)
        result = _grams_from_volume_candidate(qty, str(unit), chosen)
        if result is None or result.grams is None:
            return inferred, inferred is not None
        return pid, False

    if amount_kind == "count" and count_portion_index is not None:
        candidates = count_portion_index.get(int(fdc_id), [])
        if not any(c.portion_id == pid for c in candidates):
            return inferred, inferred is not None
        return pid, False

    return matched_portion_id, False


def resolve_quantity_fields(line: str, *, method: str = "rules") -> dict[str, Any]:
    """Parse quantity and unit from an ingredient line."""
    if method != "rules":
        raise ValueError(f"Unsupported parse method: {method}")

    fields = rule_parse_fields(line)
    amount_kind = classify_amount_kind(
        fields.get("quantity"),
        fields.get("unit"),
        fields.get("name"),
        ingredient_raw=line,
        parse_status=fields.get("parse_status"),
    )
    resolvable = amount_kind in ("mass", "volume", "count")

    return {
        **fields,
        "unit_kind": amount_kind if amount_kind != "unknown" else None,
        "amount_kind": amount_kind,
        "resolvable": resolvable,
    }


_PORTION_ROW_SQL = """
    SELECT fp.id, fp.fdc_id, fp.amount, fp.modifier, fp.portion_description,
           fp.gram_weight, fp.seq_num, fp.measure_unit_id,
           mu.name AS measure_unit_name, fp.data_points
    FROM usda.food_portion fp
    LEFT JOIN usda.measure_unit mu ON mu.id = fp.measure_unit_id
    WHERE fp.fdc_id = %s AND fp.gram_weight > 0
    ORDER BY fp.seq_num
"""


def _load_portion_rows_for_fdc(conn, fdc_int: int) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(_PORTION_ROW_SQL, (fdc_int,))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, db_row, strict=True)) for db_row in cur.fetchall()]


def load_portion_rows_cache(conn, fdc_ids: set[int] | list[int]) -> dict[int, list[dict[str, Any]]]:
    """Batch-load food_portion rows for many fdc_ids (one query)."""
    ids = sorted({int(x) for x in fdc_ids if x is not None})
    if not ids:
        return {}
    cache: dict[int, list[dict[str, Any]]] = {fid: [] for fid in ids}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fp.id, fp.fdc_id, fp.amount, fp.modifier, fp.portion_description,
                   fp.gram_weight, fp.seq_num, fp.measure_unit_id,
                   mu.name AS measure_unit_name, fp.data_points
            FROM usda.food_portion fp
            LEFT JOIN usda.measure_unit mu ON mu.id = fp.measure_unit_id
            WHERE fp.fdc_id = ANY(%s) AND fp.gram_weight > 0
            ORDER BY fp.fdc_id, fp.seq_num
            """,
            (ids,),
        )
        columns = [desc[0] for desc in cur.description]
        for db_row in cur.fetchall():
            row = dict(zip(columns, db_row, strict=True))
            cache.setdefault(int(row["fdc_id"]), []).append(row)
    return cache


def _grams_from_volume_candidate(
    q: float,
    unit_str: str,
    portion: PortionCandidate,
    *,
    via_judge: bool = False,
) -> PortionGramResult | None:
    try:
        recipe_ml = convert_volume(q, unit_str, "milliliter")
        ref_ml = convert_volume(portion.ref_amount, portion.ref_unit, "milliliter")
    except UnitConversionError:
        return None
    if ref_ml <= 0:
        return None
    grams = round((recipe_ml / ref_ml) * portion.gram_weight, 4)
    via = "judge " if via_judge else ""
    return PortionGramResult(
        grams=grams,
        status="ok_volume_portion",
        unit_kind="volume",
        portion_id=portion.portion_id,
        portion_ref_amount=portion.ref_amount,
        portion_ref_unit=portion.ref_unit,
        method=(
            f"volume:{q} {unit_str} via {via}portion#{portion.portion_id} "
            f"({portion.ref_amount} {portion.ref_unit}={portion.gram_weight}g)"
        ),
    )


def _resolve_water_sentinel_grams(
    quantity: Real | None,
    unit: str | None,
    *,
    name: str | None = None,
    ingredient_raw: str | None = None,
    amount_kind: str | None = None,
) -> PortionGramResult:
    """Resolve generic water lines to grams (~1 g/ml) for moisture tracking."""
    if quantity is None:
        return PortionGramResult(
            grams=None,
            status="unmeasurable",
            unit_kind=None,
            method="missing quantity",
        )
    kind = amount_kind or classify_amount_kind(
        quantity, unit, name, ingredient_raw=ingredient_raw
    )
    q = float(quantity)
    unit_str = str(unit) if unit is not None else ""

    if kind == "mass":
        mass_unit = normalize_unit(unit_str) or _normalize_parsed_unit(unit_str) or unit_str
        try:
            grams = convert_mass(q, mass_unit, "gram")
        except UnitConversionError:
            return PortionGramResult(
                grams=None,
                status="bad_unit",
                unit_kind="mass",
                method=f"unsupported mass unit {unit_str!r}",
            )
        return PortionGramResult(
            grams=round(grams, 4),
            status="ok_water_sentinel",
            unit_kind="mass",
            method=f"water_sentinel:mass:{unit_str}→g",
        )

    if kind == "volume":
        if not unit_str:
            return PortionGramResult(
                grams=None,
                status="bad_unit",
                unit_kind="volume",
                method="missing volume unit",
            )
        vol_unit = normalize_volume_unit(unit_str) or _normalize_parsed_unit(unit_str) or unit_str
        try:
            ml = convert_volume(q, vol_unit, "milliliter")
        except UnitConversionError:
            return PortionGramResult(
                grams=None,
                status="bad_unit",
                unit_kind="volume",
                method=f"unsupported volume unit {unit_str!r}",
            )
        return PortionGramResult(
            grams=round(ml, 4),
            status="ok_water_sentinel",
            unit_kind="volume",
            method=f"water_sentinel:volume:{unit_str}→g",
        )

    return PortionGramResult(
        grams=None,
        status="bad_unit",
        unit_kind=kind,
        method=f"water_sentinel:unsupported amount_kind={kind}",
    )


def resolve_grams(
    fdc_id: int | None,
    quantity: Real | None,
    unit: str | None,
    *,
    name: str | None = None,
    ingredient_raw: str | None = None,
    amount_kind: str | None = None,
    portion_index: dict[int, list[PortionCandidate]] | None = None,
    count_portion_index: dict[int, list[CountPortionCandidate]] | None = None,
    conn=None,
    matched_portion_id: int | None = None,
) -> PortionGramResult:
    """
    Resolve grams for a matched food and parsed amount.

    Pass ``portion_index`` / ``count_portion_index`` from build_* for batch runs,
    or ``conn`` to load portions for a single fdc_id on demand.
    """
    if fdc_id is None or int(fdc_id) == SENTINEL_FDC_ID:
        return PortionGramResult(
            grams=None,
            status="missing_fdc",
            unit_kind=None,
            method="no fdc_id",
        )

    if int(fdc_id) == WATER_SENTINEL_FDC_ID:
        return _resolve_water_sentinel_grams(
            quantity,
            unit,
            name=name,
            ingredient_raw=ingredient_raw,
            amount_kind=amount_kind,
        )

    if quantity is None:
        return PortionGramResult(
            grams=None,
            status="unmeasurable",
            unit_kind=None,
            method="missing quantity",
        )

    kind = amount_kind or classify_amount_kind(
        quantity, unit, name, ingredient_raw=ingredient_raw
    )
    if kind in ("unmeasurable", "unknown"):
        return PortionGramResult(
            grams=None,
            status="unmeasurable" if kind == "unmeasurable" else "bad_unit",
            unit_kind=kind,
            method=f"amount_kind={kind}",
        )

    q = float(quantity)
    unit_str = str(unit) if unit is not None else ""
    fdc_int = int(fdc_id)

    if kind == "mass":
        if not unit_str:
            return PortionGramResult(
                grams=None,
                status="bad_unit",
                unit_kind="mass",
                method="missing mass unit",
            )
        mass_unit = normalize_unit(unit_str) or _normalize_parsed_unit(unit_str) or unit_str
        try:
            grams = convert_mass(q, mass_unit, "gram")
        except UnitConversionError:
            return PortionGramResult(
                grams=None,
                status="bad_unit",
                unit_kind="mass",
                method=f"unsupported mass unit {unit_str!r}",
            )
        return PortionGramResult(
            grams=round(grams, 4),
            status="ok_mass",
            unit_kind="mass",
            method=f"mass:{unit_str}→g",
        )

    if kind == "count":
        query_tokens = infer_count_query(unit, name)
        if count_portion_index is not None:
            count_candidates = count_portion_index.get(fdc_int, [])
        elif conn is not None:
            count_candidates = [
                c
                for row in _load_portion_rows_for_fdc(conn, fdc_int)
                if (c := normalize_count_portion_row(row)) is not None
            ]
        else:
            raise PortionGramError("resolve_grams count path needs count_portion_index or conn")

        if matched_portion_id is not None:
            chosen = next(
                (c for c in count_candidates if c.portion_id == matched_portion_id),
                None,
            )
            if chosen is not None:
                grams = (q / chosen.ref_amount) * chosen.gram_weight
                return PortionGramResult(
                    grams=round(grams, 4),
                    status="ok_count_portion",
                    unit_kind="count",
                    portion_id=chosen.portion_id,
                    portion_ref_amount=chosen.ref_amount,
                    portion_ref_unit=chosen.count_label,
                    method=f"count via judge portion_id={chosen.portion_id}",
                )

        count_portion = pick_best_count_portion(count_candidates, query_tokens)
        if count_portion is None:
            return PortionGramResult(
                grams=None,
                status="no_portion",
                unit_kind="count",
                method=f"no count portion for fdc_id={fdc_int} tokens={query_tokens!r}",
            )
        grams = (q / count_portion.ref_amount) * count_portion.gram_weight
        return PortionGramResult(
            grams=round(grams, 4),
            status="ok_count_portion",
            unit_kind="count",
            portion_id=count_portion.portion_id,
            portion_ref_amount=count_portion.ref_amount,
            portion_ref_unit=count_portion.count_label,
            method=(
                f"count:{q} {unit_str or 'each'} via portion#{count_portion.portion_id} "
                f"({count_portion.ref_amount} {count_portion.count_label}="
                f"{count_portion.gram_weight}g)"
            ),
        )

    # Volume path
    vol_unit = normalize_unit(unit_str) or _normalize_parsed_unit(unit_str) or unit_str
    try:
        normalize_volume_unit(vol_unit)
    except UnitConversionError:
        return PortionGramResult(
            grams=None,
            status="bad_unit",
            unit_kind="volume",
            method=f"unsupported volume unit {unit_str!r}",
        )
    unit_str = vol_unit

    if portion_index is not None:
        candidates = portion_index.get(fdc_int, [])
    else:
        if conn is None:
            raise PortionGramError("resolve_grams volume path needs portion_index or conn")
        candidates = [
            c
            for row in _load_portion_rows_for_fdc(conn, fdc_int)
            if (c := normalize_portion_row(row)) is not None
        ]

    if matched_portion_id is not None:
        chosen = next((c for c in candidates if c.portion_id == matched_portion_id), None)
        if chosen is not None:
            result = _grams_from_volume_candidate(q, unit_str, chosen, via_judge=True)
            if result is not None:
                return result

    portion = pick_best_portion(candidates, unit_str)
    if portion is None:
        return PortionGramResult(
            grams=None,
            status="no_portion",
            unit_kind="volume",
            method=f"no volume portion for fdc_id={fdc_int}",
        )

    result = _grams_from_volume_candidate(q, unit_str, portion, via_judge=False)
    if result is None:
        return PortionGramResult(
            grams=None,
            status="no_portion",
            unit_kind="volume",
            portion_id=portion.portion_id,
            method="invalid portion reference volume",
        )
    return result


def load_portion_index_from_db() -> dict[int, list[PortionCandidate]]:
    """Convenience: connect and build the full volume portion index."""
    with connect() as conn:
        return build_portion_index(conn)


def load_count_portion_index_from_db() -> dict[int, list[CountPortionCandidate]]:
    """Convenience: connect and build the full count portion index."""
    with connect() as conn:
        return build_count_portion_index(conn)


def load_portion_capability_sets_from_db() -> PortionCapabilitySets:
    with connect() as conn:
        return build_portion_capability_sets(conn)


def _is_serving_only_portions(
    raw_rows: list[dict[str, Any]],
    count_candidates: list[CountPortionCandidate],
    query_tokens: list[str],
) -> bool:
    """True when only mass/serving rows exist and no count label matches query."""
    if count_candidates and pick_best_count_portion(count_candidates, query_tokens):
        return False
    if not raw_rows:
        return False
    serving_markers = ("serving", "servings", "racc", "undetermined")
    has_serving_like = False
    for row in raw_rows:
        cand = normalize_count_portion_row(row)
        if cand is not None:
            if pick_best_count_portion([cand], query_tokens):
                return False
            label_text = f"{cand.count_label} {cand.modifier} {cand.portion_description}".lower()
            if any(s in label_text for s in serving_markers):
                has_serving_like = True
                continue
            return False
        tag = classify_food_portion_row(row)
        if tag == "mass":
            mod = str(row.get("modifier") or "").lower()
            desc = str(row.get("portion_description") or "").lower()
            mu = str(row.get("measure_unit_name") or "").lower()
            text = f"{mod} {desc} {mu}"
            if any(s in text for s in serving_markers):
                has_serving_like = True
    return has_serving_like


def _resolve_mass_qty_unit(qty: float, unit: str) -> PortionGramResult | None:
    mass_unit = normalize_unit(unit) or _normalize_parsed_unit(unit) or unit
    try:
        grams = convert_mass(float(qty), mass_unit, "gram")
    except UnitConversionError:
        return PortionGramResult(
            grams=None,
            status="bad_unit",
            unit_kind="mass",
            method=f"unsupported mass unit {unit!r}",
        )
    return PortionGramResult(
        grams=round(grams, 4),
        status="ok_embedded_mass",
        unit_kind="mass",
        method=f"mass:{qty} {unit}→g",
    )


def resolve_grams_from_plan(
    plan: Any,
    fdc_id: int | None,
    *,
    name: str | None = None,
    ingredient_raw: str | None = None,
    portion_index: dict[int, list[PortionCandidate]] | None = None,
    count_portion_index: dict[int, list[CountPortionCandidate]] | None = None,
    conn=None,
    matched_portion_id: int | None = None,
    portion_rows_cache: dict[int, list[dict[str, Any]]] | None = None,
    llm_negligible_calories: bool = False,
) -> PortionGramResult:
    """Priority-ladder gram resolution from a ResolutionPlan."""
    from resolution_plan import ResolutionPlan, plan_from_parsed_row

    if not isinstance(plan, ResolutionPlan):
        if isinstance(plan, dict) and "resolution_paths" in plan:
            plan = ResolutionPlan(**{k: v for k, v in plan.items() if k in ResolutionPlan.__dataclass_fields__})
        else:
            plan = plan_from_parsed_row(plan)

    if fdc_id is None or int(fdc_id) == SENTINEL_FDC_ID:
        return PortionGramResult(grams=None, status="missing_fdc", unit_kind=None, method="no fdc_id")

    if "compound_ingredient" in plan.flags and plan.negligible_calories:
        return PortionGramResult(
            grams=0.0,
            status="compound_skipped",
            unit_kind=None,
            method="negligible compound ingredient",
        )

    fdc_int = int(fdc_id)

    for path in plan.resolution_paths:
        if path in ("embedded_mass", "parenthetical_mass_override"):
            qty = plan.parenthetical_mass_qty if path == "parenthetical_mass_override" else plan.embedded_mass_qty
            unit = (
                plan.parenthetical_mass_unit
                if path == "parenthetical_mass_override"
                else plan.embedded_mass_unit
            )
            if qty is not None and unit:
                result = _resolve_mass_qty_unit(qty, unit)
                if result and result.grams is not None:
                    return result

        if path == "explicit_mass":
            if plan.quantity is not None and plan.unit:
                result = _resolve_mass_qty_unit(plan.quantity, plan.unit)
                if result and result.grams is not None:
                    return result

        if path == "explicit_volume":
            if plan.quantity is not None and plan.unit:
                return resolve_grams(
                    fdc_id,
                    plan.quantity,
                    plan.unit,
                    name=name,
                    ingredient_raw=ingredient_raw,
                    amount_kind="volume",
                    portion_index=portion_index,
                    count_portion_index=count_portion_index,
                    conn=conn,
                    matched_portion_id=matched_portion_id,
                )

        if path == "count_portion":
            if plan.quantity is None:
                continue
            query_tokens = plan.count_query_tokens()
            if count_portion_index is not None:
                count_candidates = count_portion_index.get(fdc_int, [])
            elif conn is not None:
                count_candidates = [
                    c
                    for row in _load_portion_rows_for_fdc(conn, fdc_int)
                    if (c := normalize_count_portion_row(row)) is not None
                ]
            else:
                raise PortionGramError("count path needs count_portion_index or conn")

            if matched_portion_id is not None:
                chosen = next((c for c in count_candidates if c.portion_id == matched_portion_id), None)
                if chosen:
                    grams = (float(plan.quantity) / chosen.ref_amount) * chosen.gram_weight
                    return PortionGramResult(
                        grams=round(grams, 4),
                        status="ok_count_portion",
                        unit_kind="count",
                        portion_id=chosen.portion_id,
                        method=f"count via judge portion_id={chosen.portion_id}",
                    )

            count_portion = pick_count_portion_with_fallbacks(
                count_candidates,
                query_tokens,
                size_tokens=list(plan.count_size_tokens),
            )
            if count_portion is not None:
                q = float(plan.quantity)
                grams = (q / count_portion.ref_amount) * count_portion.gram_weight
                return PortionGramResult(
                    grams=round(grams, 4),
                    status="ok_count_portion",
                    unit_kind="count",
                    portion_id=count_portion.portion_id,
                    portion_ref_amount=count_portion.ref_amount,
                    portion_ref_unit=count_portion.count_label,
                    method=f"count:{q} via portion#{count_portion.portion_id}",
                )

            if portion_rows_cache is not None:
                raw_rows = portion_rows_cache.get(fdc_int, [])
            elif conn is not None:
                raw_rows = _load_portion_rows_for_fdc(conn, fdc_int)
            else:
                raw_rows = []

            container = pick_container_mass_portion(
                raw_rows,
                list(plan.count_unit_tokens) + query_tokens,
            )
            if container is not None:
                pid, gw = container
                q = float(plan.quantity)
                return PortionGramResult(
                    grams=round(q * gw, 4),
                    status="ok_count_portion",
                    unit_kind="count",
                    portion_id=pid,
                    method=f"container mass via portion#{pid}",
                )

            proxy = pick_whole_item_mass_proxy(
                raw_rows,
                quantity=float(plan.quantity),
                name=name,
                size_tokens=list(plan.count_size_tokens),
            )
            if proxy is not None:
                return proxy

            if _is_serving_only_portions(raw_rows, count_candidates, query_tokens):
                return PortionGramResult(
                    grams=None,
                    status="unresolvable_serving_only",
                    unit_kind="count",
                    method="only serving/mass portions; no item count",
                )

    if "ambiguous_quantity_accepted" in plan.flags:
        return PortionGramResult(
            grams=None,
            status="ambiguous_accepted",
            unit_kind=plan.primary_amount_kind,
            method="ambiguous quantity; accepted non-resolution",
        )

    if "no_quantity_specified" in plan.flags and missing_quantity(plan.quantity):
        return PortionGramResult(
            grams=None,
            status="no_quantity",
            unit_kind="unmeasurable",
            method="no quantity specified; grams not required",
        )

    if "vague_amount" in plan.flags and plan.quantity is None:
        return PortionGramResult(
            grams=None,
            status="vague_amount",
            unit_kind="unknown",
            method="vague amount without numeric estimate",
        )

    if llm_negligible_calories:
        return PortionGramResult(
            grams=0.0,
            status="negligible_calories",
            unit_kind=plan.primary_amount_kind,
            method="llm negligible calories; no portion",
        )

    return PortionGramResult(
        grams=None,
        status="no_portion",
        unit_kind=plan.primary_amount_kind,
        method="no resolution path succeeded",
    )


def resolve_grams_from_parsed_row(
    row: dict[str, Any] | Any,
    fdc_id: int | None,
    *,
    portion_index: dict[int, list[PortionCandidate]] | None = None,
    count_portion_index: dict[int, list[CountPortionCandidate]] | None = None,
    conn=None,
    matched_portion_id: int | None = None,
    portion_rows_cache: dict[int, list[dict[str, Any]]] | None = None,
    llm_negligible_calories: bool = False,
) -> PortionGramResult:
    """Resolve grams using fields from a parsed ingredient row."""
    from resolution_plan import build_resolution_plan, plan_from_parsed_row

    if hasattr(row, "get"):
        get = row.get
    else:
        get = lambda k, d=None: getattr(row, k, d)

    if fdc_id is not None and int(fdc_id) == WATER_SENTINEL_FDC_ID:
        return _resolve_water_sentinel_grams(
            get("quantity"),
            get("unit"),
            name=get("name"),
            ingredient_raw=get("ingredient") or get("ingredient_raw"),
            amount_kind=get("amount_kind_final") or get("amount_kind"),
        )

    if get("resolution_plan") is not None or get("line_enrichment") is not None:
        plan = plan_from_parsed_row(row)
        return resolve_grams_from_plan(
            plan,
            fdc_id,
            name=get("name"),
            ingredient_raw=get("ingredient") or get("ingredient_raw"),
            portion_index=portion_index,
            count_portion_index=count_portion_index,
            conn=conn,
            matched_portion_id=matched_portion_id or get("matched_portion_id"),
            portion_rows_cache=portion_rows_cache,
            llm_negligible_calories=llm_negligible_calories,
        )

    ingredient_raw = get("ingredient") or get("ingredient_raw")
    parse_fields = {
        "quantity": get("quantity"),
        "unit": get("unit"),
        "name": get("name"),
        "size": get("size"),
        "parse_status": get("parse_status"),
    }
    plan = build_resolution_plan(parse_fields, ingredient_raw=str(ingredient_raw or ""))
    if plan.resolution_paths and (
        len(plan.resolution_paths) > 1
        or plan.embedded_mass_qty
        or plan.flags
        or plan.count_size_tokens
    ):
        return resolve_grams_from_plan(
            plan,
            fdc_id,
            name=get("name"),
            ingredient_raw=ingredient_raw,
            portion_index=portion_index,
            count_portion_index=count_portion_index,
            conn=conn,
            matched_portion_id=matched_portion_id,
            portion_rows_cache=portion_rows_cache,
            llm_negligible_calories=llm_negligible_calories,
        )

    return resolve_grams(
        fdc_id,
        get("quantity"),
        get("unit"),
        name=get("name"),
        ingredient_raw=ingredient_raw,
        amount_kind=classify_from_parsed_row(row),
        portion_index=portion_index,
        count_portion_index=count_portion_index,
        conn=conn,
    )
