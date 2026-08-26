"""ELECTRE I — ELimination Et Choix Traduisant la REalite.

Roy, *Classement et choix en presence de points de vue multiples: la methode
ELECTRE*, RIRO 8, 1968.
"""

from __future__ import annotations

import numpy as np

from .base import Method, ScoringContext


def electre(
    data: np.ndarray, weights: np.ndarray, normalization: str = "vector"
) -> np.ndarray:
    """Net outranking flow — how many options ``i`` outranks, minus how many
    outrank ``i``. Higher is better; scores are whole numbers.

    Option ``i`` outranks ``j`` when the criteria favouring ``i`` carry enough
    weight (concordance) *and* no criterion is against it too strongly
    (discordance). Both thresholds are the mean of their matrix, the usual
    choice when the analyst has not set them by hand.

    The scores are counts, so ties are common on small problems — three options
    can easily each outrank exactly one other. A tie here means ELECTRE genuinely
    does not separate them, not that information was lost.
    """
    n_alt = data.shape[0]

    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.zeros(n_alt)
    w = weights / w_sum

    from ..normalization import normalize as _normalize

    v = _normalize(data, normalization) * w

    concordance = np.zeros((n_alt, n_alt))
    discordance = np.zeros((n_alt, n_alt))

    for i in range(n_alt):
        for j in range(n_alt):
            if i == j:
                continue
            conc_mask = v[i] >= v[j]
            concordance[i, j] = np.sum(w[conc_mask])

            disc_diffs = v[j] - v[i]
            disc_diffs = np.where(disc_diffs < 0, 0.0, disc_diffs)
            max_diff = np.max(np.abs(v[i] - v[j]))
            discordance[i, j] = 0.0 if max_diff == 0 else np.max(disc_diffs) / max_diff

    pairs = n_alt * (n_alt - 1)
    c_threshold = np.sum(concordance) / pairs if n_alt > 1 else 0.0
    d_threshold = np.sum(discordance) / pairs if n_alt > 1 else 1.0

    outranking = np.zeros((n_alt, n_alt))
    for i in range(n_alt):
        for j in range(n_alt):
            if (
                i != j
                and concordance[i, j] >= c_threshold
                and discordance[i, j] <= d_threshold
            ):
                outranking[i, j] = 1

    net_flows = np.sum(outranking, axis=1) - np.sum(outranking, axis=0)
    return net_flows.astype(float)


class Electre(Method):
    """ELECTRE I as a registered method."""

    name = "electre"
    summary = "ELECTRE I outranking; net concordance/discordance flows."
    citation = "Roy (1968)"

    normalization = "vector"

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return electre(ctx.data, ctx.weights, ctx.normalization)
