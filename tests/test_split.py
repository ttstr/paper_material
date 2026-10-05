"""Data split loading and invariant tests against user Beauty LOO."""

from pathlib import Path

import pytest

from pcdrec.data.load_splits import load_and_process, load_processed

DATA_CFG = "/workspace/pcdrec/configs/data/beauty.yaml"
PROCESSED = Path("/workspace/pcdrec/data/processed/beauty")


@pytest.fixture(scope="module")
def bundle():
    if not (PROCESSED / "beauty_loo.pkl").exists():
        return load_and_process(DATA_CFG, write_processed=True)
    return load_processed(PROCESSED)


def test_manifest_and_counts(bundle):
    assert all(v["ok"] for v in bundle["manifest_report"].values())
    assert bundle["n_users"] == 22363
    assert bundle["stats"]["n_train_interactions"] == 153776
    assert bundle["stats"]["n_valid"] == 22363
    assert bundle["stats"]["n_test"] == 22363
    assert bundle["n_items"] >= 12101


def test_one_target_per_user(bundle):
    assert len(bundle["valid_target"]) == bundle["n_users"]
    assert len(bundle["test_target"]) == bundle["n_users"]
    assert set(bundle["train_seq"]) == set(bundle["valid_target"]) == set(bundle["test_target"])


def test_train_seq_nonempty(bundle):
    for u, seq in bundle["train_seq"].items():
        assert len(seq) >= 1
        # timestamps non-decreasing within user
        ts = [e["timestamp"] for e in seq]
        assert ts == sorted(ts)


def test_item_text_fields(bundle):
    assert len(bundle["item_text"]) >= 12101
    # sample a few non-empty
    nonempty = sum(1 for t in bundle["item_text"].values() if t.strip())
    assert nonempty > 10000
