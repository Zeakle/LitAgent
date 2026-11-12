"""Withhold parameter quantity support without inferring scientific entailment."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from decimal import Decimal

_REVISION = "explicit-parameter-quantity-scope-v1"
_MODIFIER = r"(?:(?:trainable|model|network|camera|material)\s+)?"
_CHINESE_MODIFIER = r"(?:(?:可训练|模型|网络|相机|材质)的?)?"
_PARAMETERS = _MODIFIER + r"parameters?\b"
_CHINESE_PARAMETERS = _CHINESE_MODIFIER + r"参数(?!化|类型|种类|取值)"
_VALUE = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = (
    r"(?:million|billion|thousand|百万|十亿|万|亿|千|GB|MB|KB|"
    r"[KMBkmb]|%|％|(?!trainable\b|model\b|network\b|camera\b|material\b|"
    r"and\b|with\b|on\b|for\b|as\b|in\b|at\b|but\b|while\b|whereas\b)"
    r"[A-Za-z]+|兆|组|对|种|项)"
)
_QUANTITY = rf"(?P<value>{_VALUE})\s*(?P<unit>{_UNIT})?"
_COUNT_NOUN = (
    _MODIFIER
    + r"parameter\s+(?:count|number|budget|size)\b|"
    + _CHINESE_MODIFIER
    + r"参数(?:数量|个数|总数|规模|量)"
)
_UNPARSED_COUNT = (
    rf"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    rf"(?:(?:million|billion|thousand)\s+)?{_PARAMETERS}|"
    rf"(?<![A-Za-z0-9_.零一二三四五六七八九十百千万亿])"
    rf"[零一二三四五六七八九十百千万亿]+个?{_CHINESE_PARAMETERS}"
)
_COUNT_SCOPE = (
    _COUNT_NOUN + "|"
    rf"\b(?:number|count|amount|quantity)\s+of\s+(?:the\s+)?{_PARAMETERS}|"
    r"参数(?:数量|个数|总数|规模|量)|"
    rf"\b(?:more|fewer|less|additional|extra)\s+{_PARAMETERS}|"
    rf"\b(?:reduce[sd]?|reducing|increase[sd]?|increasing|decrease[sd]?|decreasing|"
    rf"halve[sd]?|halving|double[sd]?|doubling)\s+(?:the\s+)?{_PARAMETERS}|"
    rf"\b{_PARAMETERS}\s+(?:are|is|were|was)\s+"
    r"(?:reduced|increased|decreased|halved|doubled)\b|"
    r"(?:减少|削减|增加|压缩)(?:了|的)?(?:模型|网络|可训练)?参数(?!化|类型|取值)|"
    r"参数(?:更少|较少|更多|较多|减半|翻倍)"
)
_SIZE_SCOPE = r"\bmodel\s+size\b|模型大小|模型尺寸|模型体积"
_LEFT_COUNT = (
    _COUNT_NOUN + "|"
    rf"(?:number|count|amount|quantity)\s+of\s+(?:the\s+)?{_PARAMETERS}|"
    r"参数(?:数量|个数|总数|规模|量)|"
    rf"(?:reduce[sd]?|reducing|increase[sd]?|increasing|decrease[sd]?|decreasing)"
    rf"\s+(?:the\s+)?{_PARAMETERS}|"
    r"(?:减少|削减|增加|压缩)(?:了|的)?(?:模型|网络|可训练)?参数(?!化|类型|取值)"
)
_BRIDGE = (
    r"\s*(?:(?:is|are|was|were|of|to|by|at|from|has|equals?|"
    r"reduced|increased|decreased|approximately|about)\b\s*|"
    r"为|是|约|达到|降至|降到|减少|降低|增加|了|至|到|[:：=])*\s*"
)
_UNIT_NAMES = {
    "": "count",
    "k": "thousand",
    "thousand": "thousand",
    "千": "thousand",
    "m": "million",
    "million": "million",
    "百万": "million",
    "b": "billion",
    "billion": "billion",
    "十亿": "billion",
    "万": "ten_thousand",
    "亿": "hundred_million",
    "%": "percent",
    "％": "percent",
    "kb": "kilobytes",
    "mb": "megabytes",
    "gb": "gigabytes",
}
_CLAUSE_BOUNDARY = r"[。;；!?！？\n]|\.(?:\s|$)|[,，](?=\s*[A-Za-z])"


def _object_name(text: str, scope: str) -> str:
    """Keep explicit parameter kinds distinct without guessing aliases."""
    if scope == "model_size":
        return scope
    if re.search(r"\bcamera\b|相机", text, re.I):
        return "camera_parameters"
    if re.search(r"\bmaterial\b|材质", text, re.I):
        return "material_parameters"
    return "parameters"


def _scope_rows(text: str) -> list[dict]:
    """Locate only explicit quantity predicates and their adjacent expressions."""
    rows = []
    for clause in re.split(_CLAUSE_BOUNDARY, text):
        for match in re.finditer(_UNPARSED_COUNT, clause, re.I):
            # Recognize the quantity scope but never convert spelled-out numbers
            # into a purported match for a numeric source result.
            rows.append(
                {
                    "scope": "parameter_count",
                    "object": _object_name(match[0], "parameter_count"),
                    "quantity": ("unparsed", "unknown"),
                }
            )
        for scope, pattern in (
            ("parameter_count", _COUNT_SCOPE),
            ("model_size", _SIZE_SCOPE),
        ):
            for anchor in re.finditer(pattern, clause, re.I):
                rows.append(
                    {
                        "scope": scope,
                        "object": _object_name(anchor[0], scope),
                        "quantity": None,
                    }
                )
        # A bare parameter noun is a quantity scope only with a directly attached
        # numeric expression. A nearby 5D input or camera type cannot supply it.
        patterns = (
            (
                "parameter_count",
                rf"(?<![A-Za-z0-9_.-]){_QUANTITY}\s*(?:个|个的)?\s*"
                rf"(?P<object>{_PARAMETERS}|{_CHINESE_PARAMETERS})",
            ),
            (
                "parameter_count",
                rf"(?P<object>{_LEFT_COUNT}){_BRIDGE}{_QUANTITY}" r"(?![A-Za-z0-9_.])",
            ),
            (
                "model_size",
                rf"(?P<object>{_SIZE_SCOPE}){_BRIDGE}{_QUANTITY}" r"(?![A-Za-z0-9_.])",
            ),
        )
        for scope, pattern in patterns:
            for match in re.finditer(pattern, clause, re.I):
                unit = _UNIT_NAMES.get((match["unit"] or "").casefold(), "unknown")
                if scope == "model_size" and unit == "count":
                    unit = "unknown"
                if scope == "parameter_count" and unit.endswith("bytes"):
                    unit = "unknown"
                rows.append(
                    {
                        "scope": scope,
                        "object": _object_name(match["object"], scope),
                        "quantity": (
                            str(Decimal(match["value"]).normalize()),
                            unit,
                        ),
                    }
                )
    return rows


def parameter_scope_issue(text: str, quotes: list[str]) -> dict | None:
    """Reject absent or incompatible count scopes; None never grants support."""
    requested = _scope_rows(text)
    if not requested:
        return None
    available = [row for quote in quotes for row in _scope_rows(quote)]
    for row in requested:
        candidates = [
            source
            for source in available
            if source["scope"] == row["scope"] and source["object"] == row["object"]
        ]
        if not candidates:
            return {
                "issue": "support_relation_unverified",
                "aspect": "parameter_count",
                "quantity_scope": row["scope"],
            }
        quantity = row["quantity"]
        if quantity is not None and (
            quantity[1] == "unknown"
            or not any(source["quantity"] == quantity for source in candidates)
        ):
            return {
                "issue": "support_relation_unverified",
                "aspect": "parameter_count",
                "quantity_scope": row["scope"],
                "value": quantity[0],
                "unit": quantity[1],
            }
    return None


def relation_scope_identity() -> str:
    """Bind this optional negative guard and all of its parsing inputs."""
    return hashlib.sha256(
        json.dumps(
            [
                _REVISION,
                _MODIFIER,
                _CHINESE_MODIFIER,
                _PARAMETERS,
                _CHINESE_PARAMETERS,
                _VALUE,
                _UNIT,
                _QUANTITY,
                _COUNT_NOUN,
                _UNPARSED_COUNT,
                _COUNT_SCOPE,
                _SIZE_SCOPE,
                _LEFT_COUNT,
                _BRIDGE,
                _UNIT_NAMES,
                _CLAUSE_BOUNDARY,
                *[
                    inspect.getsource(function)
                    for function in (
                        _object_name,
                        _scope_rows,
                        parameter_scope_issue,
                        relation_scope_identity,
                    )
                ],
            ],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
