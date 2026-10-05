"""Item text field helpers for offline LLM prompts (title + categories + brand)."""

from __future__ import annotations

from typing import Any, Mapping

DEFAULT_TEXT_FIELDS = ("title", "categories", "brand")


def build_item_text(
    row: Mapping[str, Any],
    fields: tuple[str, ...] | list[str] = DEFAULT_TEXT_FIELDS,
) -> str:
    """Concatenate available text fields; skip empty / NaN values."""
    parts: list[str] = []
    for f in fields:
        key = f.upper() if f.upper() in row else f
        val = row.get(key, row.get(f, ""))
        if val is None:
            continue
        s = str(val).strip()
        if not s or s.lower() == "nan":
            continue
        parts.append(s)
    return " | ".join(parts)


def item_text_table(meta_df, fields: tuple[str, ...] | list[str] = DEFAULT_TEXT_FIELDS) -> dict[int, str]:
    """Map item_id -> text string from item_meta dataframe."""
    out: dict[int, str] = {}
    cols_upper = {c.upper(): c for c in meta_df.columns}
    item_col = cols_upper.get("ITEM", "ITEM")
    for _, row in meta_df.iterrows():
        iid = int(row[item_col])
        mapping = {c: row[c] for c in meta_df.columns}
        # also expose lower-case aliases
        for c in meta_df.columns:
            mapping[c.lower()] = row[c]
        out[iid] = build_item_text(mapping, fields)
    return out
