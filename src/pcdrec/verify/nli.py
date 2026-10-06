"""Small NLI cross-encoder (default cross-encoder/nli-deberta-v3-xsmall) -> P(entailment)."""

from __future__ import annotations

import numpy as np

DEFAULT_NLI = "cross-encoder/nli-deberta-v3-xsmall"


class NLIScorer:
    def __init__(self, model_name: str = DEFAULT_NLI, batch_size: int = 32, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, v in id2label.items() if v.startswith("entail"))
        self.batch_size, self.max_length = batch_size, max_length
        self.model_name = model_name
        self._cache: dict[tuple[str, str], float] = {}

    def entail(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs of (premise, hypothesis) -> entailment probabilities."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self._cache]
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s : s + self.batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            with self.torch.no_grad():
                prob = self.torch.softmax(self.model(**enc).logits, -1)[:, self.entail_idx].tolist()
            for p, v in zip(chunk, prob):
                self._cache[p] = float(v)
        return np.asarray([self._cache[p] for p in pairs], dtype=np.float64)


class LexicalEntail:
    """Deterministic stand-in for unit tests only (token-overlap proxy); never used for reported numbers."""

    model_name = "lexical-overlap-test-stub"

    def entail(self, pairs):
        out = []
        for p, h in pairs:
            pt, ht = set(p.lower().split()), set(h.lower().split())
            out.append(len(pt & ht) / max(len(ht), 1))
        return np.asarray(out, dtype=np.float64)
