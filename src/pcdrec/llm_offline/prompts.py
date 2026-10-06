"""Prompt templates (profile extraction, teacher ranking). Item ids are the dataset's integer ids."""

from __future__ import annotations

import hashlib
import json

PROFILE_SYSTEM = (
    "You are an analyst of e-commerce purchase histories. You extract a user's stable product "
    "preferences, and every preference must cite the purchased items that support it. Output JSON only."
)

PROFILE_USER = """Purchase history of one user (oldest first). Each line is [item_id] title | categories | brand.
{history}

Write at most {max_claims} preference claims about this user. Rules:
- each claim: "aspect" (short label, e.g. "category", "brand", "skin type", "price", "scent"),
  "statement" (one short sentence about what the user likes or avoids),
  "polarity" ("+" likes / "-" avoids), "evidence" (list of item_id numbers from the history above that support it).
- cite only item_ids that appear in the history; do not invent items.
Answer with exactly one compact JSON object on a single line (no markdown, no extra text) of the form
{{"claims": [{{"aspect": "...", "statement": "...", "polarity": "+", "evidence": [123, 456]}}]}}"""

RANK_SYSTEM = (
    "You rank candidate products for a user according to the user's preference profile. Output JSON only."
)

RANK_USER = """User preference profile (claim number: polarity statement):
{profile}

Candidate products (in random order). Each line is [item_id] title | categories | brand.
{candidates}

Rank ALL {n_cand} candidates from most to least likely to be the user's next purchase, using the profile.
{reason_rule}Answer with exactly one compact JSON object on a single line (no markdown, no extra text) of the form
{example}"""

REASON_RULE = ("For the top {k} items of your ranking also give a reason that cites the claim numbers it relies on.\n")


def render_items(item_ids: list[int], item_text: dict, max_chars: int = 160) -> str:
    lines = []
    for i in item_ids:
        t = (item_text.get(int(i), "") or "").replace("\n", " ")
        lines.append(f"[{int(i)}] {t[:max_chars]}")
    return "\n".join(lines)


def profile_messages(history: list[int], item_text: dict, max_claims: int = 8) -> list[dict]:
    return [
        {"role": "system", "content": PROFILE_SYSTEM},
        {"role": "user", "content": PROFILE_USER.format(history=render_items(history, item_text), max_claims=max_claims)},
    ]


def render_profile(claims: list[dict]) -> str:
    if not claims:
        return "(no claims)"
    return "\n".join(f"{k}: {c['polarity']} {c['statement']}" for k, c in enumerate(claims))


def rank_messages(claims: list[dict], candidates: list[int], item_text: dict, reasons_top_k: int = 0) -> list[dict]:
    if reasons_top_k > 0:
        reason_rule = REASON_RULE.format(k=reasons_top_k)
        example = ('{"ranking": [id1, id2, ...], "reasons": [{"item": id1, "claims": [0, 2], '
                   '"reason": "..."}]}')
    else:
        reason_rule = ""
        example = '{"ranking": [id1, id2, ...]}'
    return [
        {"role": "system", "content": RANK_SYSTEM},
        {"role": "user", "content": RANK_USER.format(profile=render_profile(claims),
                                                     candidates=render_items(candidates, item_text),
                                                     n_cand=len(candidates), reason_rule=reason_rule,
                                                     example=example)},
    ]


def messages_hash(messages: list[dict], model: str, decoding: dict) -> str:
    blob = json.dumps({"m": messages, "model": model, "dec": decoding}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
