"""Distillation losses: S-DPO align (with/without reference), ListKL, BPR variant, L_pref, aux pipeline."""

import math

import torch

from pcdrec.distill.student import PCDRecStudent
from pcdrec.distill.train_student import aux_losses, collate_aux
from pcdrec.losses import AttnPool, bpr_distill_loss, listkl_loss, pref_contrast_loss, sdpo_align_loss
from pcdrec.models.sasrec import SASRec


def test_sdpo_reference_zero_margin_value():
    s_pos, s_neg = torch.tensor([1.0]), torch.tensor([[0.5, 0.2, 0.1]])
    w = torch.ones(1)
    l = sdpo_align_loss(s_pos, s_neg, s_pos.clone(), s_neg.clone(), w)   # Delta = 0 everywhere
    assert math.isclose(l.item(), math.log(1 + 3), rel_tol=1e-6)
    better = sdpo_align_loss(s_pos + 1.0, s_neg, s_pos, s_neg, w)
    assert better < l
    no_ref = sdpo_align_loss(s_pos, s_neg, None, None, w)
    assert torch.isfinite(no_ref)


def test_weights_zero_user_ignored():
    s_pos, s_neg = torch.tensor([1.0, -5.0]), torch.tensor([[0.0], [5.0]])
    l1 = sdpo_align_loss(s_pos[:1], s_neg[:1], None, None, torch.ones(1))
    l2 = sdpo_align_loss(s_pos, s_neg, None, None, torch.tensor([1.0, 0.0]))
    assert torch.allclose(l1, l2)
    assert torch.isfinite(bpr_distill_loss(s_pos, s_neg, torch.ones(2)))


def test_listkl_zero_at_teacher():
    q = torch.tensor([[0.7, 0.2, 0.1]])
    s = torch.log(q)
    assert listkl_loss(s, q, torch.ones(1)).abs().item() < 1e-6
    assert listkl_loss(torch.zeros(1, 3), q, torch.ones(1)).item() > 0


def test_pref_contrast_prefers_alignment():
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    hard = torch.nn.functional.normalize(torch.randn(4, 2, 8), dim=-1)
    w = torch.ones(4)
    aligned = pref_contrast_loss(z, z, hard, torch.ones(4, 2, dtype=torch.bool), w, 0.1)
    rand = pref_contrast_loss(torch.nn.functional.normalize(torch.randn(4, 8), dim=-1), z, hard,
                              torch.ones(4, 2, dtype=torch.bool), w, 0.1)
    assert aligned < rand
    pool = AttnPool(8)
    out = pool(torch.randn(2, 3, 8), torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


def _sig(u, n_items=30, D=8):
    g = torch.Generator().manual_seed(u)
    return {"context": [1, 2, 3 + u], "cands": torch.arange(5) + u, "q": torch.softmax(torch.randn(5, generator=g), 0),
            "pos": int(u), "negs": torch.tensor([10, 11, 12]), "w": 0.8,
            "claims": torch.nn.functional.normalize(torch.randn(3, D, generator=g), dim=-1),
            "hard": [torch.randn(2, D, generator=g), torch.randn(4, D, generator=g)]}


def test_aux_losses_backward_all_variants():
    torch.manual_seed(0)
    bb = SASRec(n_items=30, hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=6)
    ref = SASRec(n_items=30, hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=6)
    st = PCDRecStudent(bb, claim_dim=8)
    sig = {u: _sig(u) for u in range(3)}
    batch = collate_aux([0, 1, 2], sig, bb.pad_id, 6, 8)
    for at in ("sdpo", "listkl", "bpr"):
        for use_ref in (True, False):
            cfg = {"loss": {"lambda1": 1, "lambda2": 1, "align_type": at, "use_ref": use_ref, "beta": 1, "tau": 0.1},
                   "neg": {"in_batch": True}}
            out = aux_losses(st, ref, batch, cfg)
            loss = out["align"] + out["pref"]
            st.zero_grad()
            loss.backward()
            assert torch.isfinite(loss)
            assert bb.item_emb.weight.grad is not None
    assert all(p.grad is None for p in ref.parameters())
