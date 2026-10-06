"""JSON schemas + tolerant parsing/validation for LLM outputs (profile, teacher ranking)."""

from __future__ import annotations

import json
import re
from typing import Any

PROFILE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "statement": {"type": "string"},
                    "polarity": {"type": "string", "enum": ["+", "-"]},
                    "evidence": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                },
                "required": ["aspect", "statement", "polarity", "evidence"],
            },
        }
    },
    "required": ["claims"],
}

RANK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "integer"},
                    "claims": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                },
                "required": ["item", "claims", "reason"],
            },
        },
    },
    "required": ["ranking"],
}


def extract_json(text: str) -> Any | None:
    """Return the first decodable JSON object in ``text`` (handles ```json fences / chatter)."""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _to_int(x) -> int | None:
    try:
        if isinstance(x, str):
            x = x.strip().strip("[]#").replace("item", "").strip()
        return int(x)
    except (TypeError, ValueError):
        return None


def validate_profile(obj: Any, max_claims: int = 8) -> tuple[list[dict], dict]:
    """Normalise a profile; returns (claims, report). Claims keep the raw evidence ids (validated later by G)."""
    report = {"parsed": obj is not None, "n_raw": 0, "n_dropped_malformed": 0}
    if isinstance(obj, list):
        obj = {"claims": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        report["parsed"] = False
        return [], report
    claims = []
    for c in obj["claims"]:
        report["n_raw"] += 1
        if not isinstance(c, dict):
            report["n_dropped_malformed"] += 1
            continue
        st = str(c.get("statement", "")).strip()
        pol = str(c.get("polarity", "+")).strip()
        pol = "-" if pol in ("-", "negative", "neg", "dislike") else "+"
        ev_raw = c.get("evidence", [])
        if not isinstance(ev_raw, list):
            ev_raw = [ev_raw]
        ev = [v for v in (_to_int(e) for e in ev_raw) if v is not None]
        if not st:
            report["n_dropped_malformed"] += 1
            continue
        claims.append({"aspect": str(c.get("aspect", "")).strip()[:60], "statement": st[:300],
                       "polarity": pol, "evidence": ev})
        if len(claims) >= max_claims:
            break
    return claims, report


def validate_ranking(obj: Any, candidates: list[int]) -> tuple[list[int], list[dict], dict]:
    """Return (full ranking over candidates, reasons, report).

    Unknown / duplicate ids are dropped; candidates the model omitted are appended in their
    prompt order (counted in the report, so position-bias / omission rates are measurable).
    """
    cand_set = set(candidates)
    report = {"parsed": obj is not None, "n_unknown": 0, "n_dup": 0, "n_missing_appended": 0}
    if isinstance(obj, list):
        obj = {"ranking": obj}
    raw = obj.get("ranking", []) if isinstance(obj, dict) else []
    seen, ranking = set(), []
    for x in raw if isinstance(raw, list) else []:
        v = _to_int(x)
        if v is None or v not in cand_set:
            report["n_unknown"] += 1
            continue
        if v in seen:
            report["n_dup"] += 1
            continue
        seen.add(v)
        ranking.append(v)
    for c in candidates:
        if c not in seen:
            ranking.append(c)
            report["n_missing_appended"] += 1
    reasons = []
    for r in (obj.get("reasons", []) if isinstance(obj, dict) else []) or []:
        if not isinstance(r, dict):
            continue
        it = _to_int(r.get("item"))
        if it is None or it not in cand_set:
            continue
        cl = r.get("claims", [])
        cl = [v for v in (_to_int(c) for c in (cl if isinstance(cl, list) else [cl])) if v is not None]
        reasons.append({"item": it, "claims": cl, "reason": str(r.get("reason", ""))[:300]})
    report["parsed"] = bool(report["parsed"] and isinstance(obj, dict) and isinstance(raw, list) and len(raw) > 0)
    return ranking, reasons, report


def salvage_profile(text: str) -> dict | None:
    """Recover complete claim objects from a truncated/invalid profile output (nothing is invented)."""
    if not text:
        return None
    dec = json.JSONDecoder()
    claims = []
    start = text.find("[")
    i = start + 1 if start >= 0 else 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "statement" in obj:
            claims.append(obj)
        i = j + max(end, 1)
    return {"claims": claims, "_salvaged": True} if claims else None


def salvage_ranking(text: str) -> dict | None:
    """Recover the integer list after "ranking": [ even if the array/object is truncated."""
    if not text:
        return None
    m = re.search(r'"ranking"\s*:\s*\[([^\]]*)', text)
    if not m:
        return None
    ids = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    out: dict = {"ranking": ids, "_salvaged": True}
    rs = re.search(r'"reasons"\s*:\s*\[', text)
    if rs:
        sal = salvage_profile_like_objects(text[rs.end():])
        if sal:
            out["reasons"] = sal
    return out if ids else None


def salvage_profile_like_objects(text: str) -> list[dict]:
    dec = json.JSONDecoder()
    objs, i = [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + max(end, 1)
        except json.JSONDecodeError:
            i = j + 1
    return objs


def parse_profile_text(text: str, max_claims: int = 8) -> tuple[list[dict], dict]:
    obj = extract_json(text)
    salvaged = False
    if not (isinstance(obj, dict) and isinstance(obj.get("claims"), list)):
        obj = salvage_profile(text)
        salvaged = obj is not None
    claims, rep = validate_profile(obj, max_claims)
    rep["salvaged"] = salvaged
    return claims, rep


def parse_ranking_text(text: str, candidates: list[int]) -> tuple[list[int], list[dict], dict]:
    obj = extract_json(text)
    salvaged = False
    if not (isinstance(obj, dict) and isinstance(obj.get("ranking"), list) and obj.get("ranking")):
        obj = salvage_ranking(text)
        salvaged = obj is not None
    ranking, reasons, rep = validate_ranking(obj, candidates)
    rep["salvaged"] = salvaged
    return ranking, reasons, rep
