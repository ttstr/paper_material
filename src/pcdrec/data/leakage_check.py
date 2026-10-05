"""Zero-leakage checks for offline LLM inputs and temporal holdout prefixes.

Rules (method plan §1.3 / §7.5):
- LLM inputs must never contain valid/test target item ids.
- Profile prefixes use train only; optional holdout_h temporal leave-out on train suffix.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def assert_no_target_ids_in_llm_input(
    llm_item_ids: Iterable[int],
    forbidden_target_ids: Iterable[int],
    *,
    context: str = "llm_input",
) -> None:
    """Raise AssertionError if any forbidden valid/test target id appears in LLM input path."""
    forbidden = set(int(x) for x in forbidden_target_ids)
    present = set(int(x) for x in llm_item_ids) & forbidden
    if present:
        raise AssertionError(
            f"Leakage in {context}: valid/test target item ids appear in LLM input: {sorted(present)[:20]}"
        )


def train_prefix_for_profile(
    train_item_ids: Sequence[int],
    train_timestamps: Sequence[float] | None = None,
    holdout_h: int = 3,
) -> list[int]:
    """Return train-only prefix for LLM profile extraction.

    Drops the last ``holdout_h`` train interactions (temporal leave-out boundary).
    If sequence is shorter than holdout_h + 1, keep at least the first item when possible.
    """
    items = list(train_item_ids)
    if holdout_h <= 0:
        return items
    if len(items) <= holdout_h:
        return items[: max(1, len(items) - 1)] if len(items) > 1 else items
    return items[:-holdout_h]


def assert_prefix_respects_holdout(
    full_train_ids: Sequence[int],
    prefix_ids: Sequence[int],
    holdout_h: int = 3,
) -> None:
    """Assert profile prefix equals train[:-holdout_h] (or shorter-seq fallback)."""
    expected = train_prefix_for_profile(full_train_ids, holdout_h=holdout_h)
    if list(prefix_ids) != expected:
        raise AssertionError(
            f"Profile prefix violates holdout_h={holdout_h}: got {list(prefix_ids)} expected {expected}"
        )


def collect_valid_test_targets(valid_by_user: dict, test_by_user: dict) -> set[int]:
    """Union of valid/test target item ids across users."""
    out: set[int] = set()
    for d in (valid_by_user, test_by_user):
        for v in d.values():
            if isinstance(v, (list, tuple)):
                out.update(int(x) for x in v)
            else:
                out.add(int(v))
    return out
