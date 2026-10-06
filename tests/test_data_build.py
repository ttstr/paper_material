"""Data build helpers: CRLF reproduction and reference hashes match the shipped loader manifest."""

import importlib.util
import json
from pathlib import Path

from pcdrec.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "build_beauty", REPO_ROOT / "scripts/data_build/build_beauty_from_recboard.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def test_crlf_keeps_quoted_newlines():
    lf = b'ITEM\tTITLE\n0\t"a\nb"\n1\t"say ""hi"""\n'
    out = bb.to_crlf(lf)
    assert out == b'ITEM\tTITLE\r\n0\t"a\nb"\r\n1\t"say ""hi"""\r\n'
    assert bb.to_lf(out) == lf.replace(b'"a\nb"', b'"a\nb"')


def test_reference_hashes_cover_all_files():
    for mode in ("crlf", "lf"):
        assert set(bb.REFERENCE_SHA256[mode]) == set(bb.FILE_MAP.values())


def test_reference_matches_local_manifest_if_present():
    from pcdrec.data.load_splits import load_data_config

    cfg = load_data_config(REPO_ROOT / "configs/data/beauty.yaml")
    mp = Path(cfg["manifest"])
    if not mp.exists():
        import pytest
        pytest.skip("no local data")
    man = json.loads(mp.read_text())
    for name, sha in bb.REFERENCE_SHA256["crlf"].items():
        assert man[name]["sha256"] == sha
