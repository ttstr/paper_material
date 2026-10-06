"""PCS aggregation (plan §2.3): PCS_u = G^alpha * T^beta * R^gamma (switchable components, A2),
w_u = sigmoid((PCS_u - theta) / T_w), hard filter G_u < g_min -> no teacher signal (w_u = 0).
With pcs.enabled = false every user gets w_u = 1 (Naive-Distill, A1)."""

from __future__ import annotations

import math


def pcs_score(G: float, T: float, R: float, cfg: dict) -> float:
    s = 1.0
    if cfg.get("use_G", True):
        s *= max(G, 0.0) ** float(cfg.get("alpha", 1.0))
    if cfg.get("use_T", True):
        s *= max(T, 0.0) ** float(cfg.get("beta", 1.0))
    if cfg.get("use_R", True):
        s *= max(R, 0.0) ** float(cfg.get("gamma", 1.0))
    return float(s)


def user_weight(G: float, T: float, R: float, cfg: dict) -> tuple[float, float]:
    """Return (PCS_u, w_u)."""
    if not cfg.get("enabled", True):
        return float("nan"), 1.0
    p = pcs_score(G, T, R, cfg)
    if cfg.get("use_G", True) and G < float(cfg.get("g_min", 0.0)):
        return p, 0.0
    w = 1.0 / (1.0 + math.exp(-(p - float(cfg.get("theta", 0.25))) / float(cfg.get("T_w", 0.1))))
    return p, float(w)
