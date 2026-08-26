"""VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje.

Opricovic and Tzeng, *Compromise solution by MCDM methods: A comparative
analysis of VIKOR and TOPSIS*, European Journal of Operational Research
156(2), 2004.
"""

from __future__ import annotations

import numpy as np

from ..normalization import normalize_weights
from .base import Method, ScoringContext


def vikor(data: np.ndarray, weights: np.ndarray, v: float = 0.5) -> np.ndarray:
    """Compromise ranking, returned **negated** so higher is better.

    VIKOR's native ``Q`` measures regret, so the smallest ``Q`` is the best
    option. Every method in this package reads higher-is-better, so ``-Q`` is
    returned. Scores are therefore in ``[-1, 0]``.

    ``S`` is group utility (the weighted sum of normalised regret) and ``R`` is
    individual regret (its maximum). ``v`` trades one against the other:
    ``v = 1`` maximises group utility alone, ``v = 0`` minimises the worst
    single criterion, and the customary ``0.5`` weighs them equally.
    """
    n_alt = data.shape[0]

    w = normalize_weights(weights)
    if w is None:
        return np.zeros(n_alt)

    f_best = np.max(data, axis=0)
    f_worst = np.min(data, axis=0)

    denom = f_best - f_worst
    denom = np.where(denom == 0, 1.0, denom)

    regret_matrix = w * (f_best - data) / denom

    S = np.sum(regret_matrix, axis=1)
    R = np.max(regret_matrix, axis=1)

    S_best, S_worst = np.min(S), np.max(S)
    R_best, R_worst = np.min(R), np.max(R)

    s_denom = S_worst - S_best if S_worst != S_best else 1.0
    r_denom = R_worst - R_best if R_worst != R_best else 1.0

    Q = v * (S - S_best) / s_denom + (1 - v) * (R - R_best) / r_denom

    return (-Q).astype(float)


class Vikor(Method):
    """VIKOR as a registered method.

    Accepts ``v`` through ``rank(..., v=...)``: the weight given to group
    utility against worst-case regret.
    """

    name = "vikor"
    summary = "Compromise ranking between group utility and worst-case regret."
    citation = "Opricovic and Tzeng (2004)"

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return vikor(ctx.data, ctx.weights, **ctx.opts)
