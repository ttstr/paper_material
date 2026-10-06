"""Repository-relative path resolution.

All configs store paths relative to the repository root. Absolute paths are
still accepted. Environment overrides:

- ``PCDREC_DATA_DIR``      -> directory with train/valid/test/item_meta.txt + manifest.json
                              (default: ``<repo>/data/raw/amazon-beauty``)
- ``PCDREC_PROCESSED_DIR`` -> processed cache directory (default: ``<repo>/data/processed/beauty``)
- ``PCDREC_RESULTS_DIR``   -> results directory (default: ``<repo>/results``)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | os.PathLike | None, base: Path | None = None) -> Path | None:
    """Expand ``~``/env vars; make relative paths relative to the repo root."""
    if path is None:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not p.is_absolute():
        p = (base or REPO_ROOT) / p
    return p


def apply_data_env(cfg: dict) -> dict:
    """Resolve data-config paths and apply environment overrides (returns a copy)."""
    cfg = dict(cfg)
    env_raw = os.environ.get("PCDREC_DATA_DIR")
    if env_raw:
        cfg["raw_dir"] = env_raw
        cfg["manifest"] = str(Path(env_raw) / "manifest.json")
    cfg["raw_dir"] = str(resolve(cfg.get("raw_dir", "data/raw/amazon-beauty")))
    cfg["manifest"] = str(resolve(cfg.get("manifest") or Path(cfg["raw_dir"]) / "manifest.json"))
    env_proc = os.environ.get("PCDREC_PROCESSED_DIR")
    cfg["processed_dir"] = str(resolve(env_proc or cfg.get("processed_dir", "data/processed/beauty")))
    return cfg


def results_dir(default: str | None = None) -> Path:
    return resolve(os.environ.get("PCDREC_RESULTS_DIR") or default or "results")
