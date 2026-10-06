"""Shape / semantics tests for GRU4Rec, BERT4Rec, SASRec-T, MF-BPR and the shared evaluator."""

import numpy as np
import torch

from pcdrec.metrics.ranking import ranks_from_score_matrix, ranks_from_scores
from pcdrec.models.bert4rec import BERT4Rec
from pcdrec.models.gru4rec import GRU4Rec
from pcdrec.models.mf_bpr import MFBPR
from pcdrec.models.sasrec import SASRec
from pcdrec.models.sasrec_t import SASRecT


def test_vectorised_ranks_match_reference():
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, 5, (20, 30), generator=g).float()  # many ties
    t = torch.randint(0, 30, (20,), generator=g)
    vec = ranks_from_score_matrix(s, t).numpy()
    ref = [ranks_from_scores(s[i].numpy(), int(t[i])) for i in range(20)]
    assert list(vec) == ref


def test_gru4rec_padding_invariance():
    torch.manual_seed(0)
    m = GRU4Rec(n_items=50, hidden_size=8, gru_hidden=8, dropout=0.0, max_seq_length=6).eval()
    pad = m.pad_id
    a = torch.tensor([[pad, pad, pad, 3, 4, 5]])
    b = torch.tensor([[pad, 3, 4, 5]])
    la, lb = m.predict_logits(a), m.predict_logits(b)
    assert torch.allclose(la, lb, atol=1e-6)
    h = m(a)
    assert torch.all(h[0, :3] == 0)


def test_bert4rec_mask_and_eval_shapes():
    torch.manual_seed(0)
    m = BERT4Rec(n_items=40, hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=5)
    seq = torch.tensor([[m.pad_id, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    inp, tgt = m.mask_batch(seq)
    assert (inp == m.mask_id).sum(1).min() >= 1
    assert torch.all((tgt == -100) | (inp == m.mask_id))
    ap = m.append_mask(seq)
    assert ap[0, -1] == m.mask_id and ap[0, -2] == 4 and ap.shape == seq.shape
    assert m.eval().predict_logits(seq).shape == (2, 40)


def test_sasrec_t_extra_params_only_projection():
    kw = dict(hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=5)
    base = SASRec(n_items=30, **kw)
    t = SASRecT(n_items=30, text_emb=np.random.randn(30, 12).astype(np.float32), **kw)
    assert t.num_parameters() - base.num_parameters() == 12 * 8
    seq = torch.tensor([[t.pad_id, 1, 2, 3, 4]])
    assert t.eval().predict_logits(seq).shape == (1, 30)
    assert torch.all(t.embed_items(torch.tensor([t.pad_id])) == 0)


def test_mf_bpr_scores_and_loss():
    m = MFBPR(n_users=5, n_items=7, hidden_size=4)
    s = m.score(torch.tensor([0, 3]), None)
    assert s.shape == (2, 7)
    loss = m.bpr_loss(torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), l2=1e-4)
    loss.backward()
    assert m.user_emb.weight.grad is not None
