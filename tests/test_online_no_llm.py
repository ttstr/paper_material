"""Exported online module must not import LLM stacks; param count = SASRec only."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

from pcdrec.export_online import export
from pcdrec.models.sasrec import SASRec

from pcdrec.paths import REPO_ROOT as REPO
CKPT_CANDIDATES = [
    REPO / "results/checkpoints/sasrec_subset1000_best.pt",
    REPO / "results/checkpoints/sasrec_subset500_best.pt",
    REPO / "results/checkpoints/sasrec_best.pt",
]


def _find_ckpt() -> Path | None:
    for p in CKPT_CANDIDATES:
        if p.exists():
            return p
    # any best ckpt
    d = REPO / "results/checkpoints"
    if d.exists():
        pts = sorted(d.glob("*_best.pt"))
        if pts:
            return pts[0]
    return None


FORBIDDEN = {"vllm"}
# transformers generative classes that must not appear for online path
FORBIDDEN_ATTRS = {"AutoModelForCausalLM", "pipeline", "vllm"}


def _collect_imports(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for a in node.names:
                names.add(a.name)
    return names


def test_online_export_no_llm_imports_and_param_match():
    ckpt = _find_ckpt()
    if ckpt is None:
        pytest.skip("No SASRec checkpoint yet; run smoke train first")

    out_dir = REPO / "results/online_export_test"
    meta = export(str(ckpt), str(out_dir))

    infer_py = out_dir / "online_infer.py"
    assert infer_py.exists()
    imports = _collect_imports(infer_py)
    assert "vllm" not in imports
    assert "transformers" not in imports
    text = infer_py.read_text(encoding="utf-8")
    for bad in FORBIDDEN_ATTRS:
        assert bad not in text

    # param count equals plain SASRec
    ref = SASRec(
        n_items=meta["n_items"],
        hidden_size=meta["hidden_size"],
        n_layers=meta["n_layers"],
        n_heads=meta["n_heads"],
        inner_size=meta["inner_size"],
        max_seq_length=meta["max_seq_length"],
    )
    assert meta["n_params"] == ref.num_parameters()
    assert meta["online_llm_calls"] == 0
    assert meta["has_aux_heads"] is False

    # loadable
    loaded = torch.load(out_dir / "sasrec.pt", map_location="cpu", weights_only=False)
    ref.load_state_dict(loaded["model_state"])


def test_student_export_drops_aux_heads(tmp_path):
    from pcdrec.distill.student import PCDRecStudent
    from pcdrec.export_online import export_student

    torch.manual_seed(0)
    bb = SASRec(n_items=40, hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=6)
    st = PCDRecStudent(bb, claim_dim=12)
    ck = tmp_path / "student.pt"
    torch.save({"student_state": st.state_dict(), "online_state": st.online_state_dict(), "n_items": 40, "max_len": 6,
                "model_cfg": {"hidden_size": 8, "n_layers": 1, "n_heads": 2, "inner_size": 16}}, ck)
    meta = export_student(str(ck), str(tmp_path / "exp"))
    assert meta["n_params"] == bb.num_parameters() < sum(p.numel() for p in st.parameters())
    assert set(meta["dropped_training_only_keys"]) == {"g", "pool"}
    saved = torch.load(tmp_path / "exp/sasrec.pt", weights_only=False)["model_state"]
    assert all(not k.startswith(("g.", "pool.", "inject.")) for k in saved)
    imports = _collect_imports(tmp_path / "exp/online_infer.py")
    assert not imports & {"vllm", "transformers", "sentence_transformers", "openai"}
    assert "llm_offline" not in (tmp_path / "exp/online_infer.py").read_text()


def test_export_module_has_no_llm_imports():
    import pcdrec.export_online as eo

    imports = _collect_imports(Path(eo.__file__))
    assert not imports & {"vllm", "transformers", "sentence_transformers", "openai", "llm_offline", "verify"}
