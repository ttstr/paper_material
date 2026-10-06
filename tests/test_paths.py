from pathlib import Path

from pcdrec.paths import REPO_ROOT, apply_data_env


def test_relative_paths_resolve_to_repo(monkeypatch):
    monkeypatch.delenv("PCDREC_DATA_DIR", raising=False)
    monkeypatch.delenv("PCDREC_PROCESSED_DIR", raising=False)
    cfg = apply_data_env({"raw_dir": "data/raw/amazon-beauty", "processed_dir": "data/processed/beauty"})
    assert Path(cfg["raw_dir"]) == REPO_ROOT / "data/raw/amazon-beauty"
    assert Path(cfg["manifest"]) == REPO_ROOT / "data/raw/amazon-beauty/manifest.json"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PCDREC_DATA_DIR", str(tmp_path))
    cfg = apply_data_env({"raw_dir": "data/raw/amazon-beauty", "manifest": "x/manifest.json"})
    assert cfg["raw_dir"] == str(tmp_path)
    assert cfg["manifest"] == str(tmp_path / "manifest.json")


def test_no_absolute_paths_in_configs():
    for y in (REPO_ROOT / "configs").rglob("*.yaml"):
        txt = y.read_text()
        assert "/workspace/" not in txt and ":\\\\" not in txt, y
