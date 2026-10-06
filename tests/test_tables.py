import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("main_table", ROOT / "scripts" / "main_table.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def test_ndcg10_matches_definition():
    r = np.array([1, 2, 10, 11, 5000])
    np.testing.assert_allclose(mt.ndcg10(r), [1.0, 1 / np.log2(3), 1 / np.log2(11), 0.0, 0.0])


def test_holm_monotone_and_capped():
    out = mt.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(out["a"] - 0.03) < 1e-12          # 3 * 0.01
    assert abs(out["c"] - 0.06) < 1e-12          # 2 * 0.03
    assert abs(out["b"] - 0.06) < 1e-12          # max(0.06, 1 * 0.04)
    assert mt.holm({"x": 0.9, "y": 0.8})["x"] == 1.0 or mt.holm({"x": 0.9, "y": 0.8})["x"] <= 1.0


def test_mean_std_sample():
    m, s = mt.ms([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(s - 1.0) < 1e-12
