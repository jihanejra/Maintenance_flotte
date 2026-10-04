"""Métriques : qualité des courbes de survie, qualité de classement, coût du challenge."""

from __future__ import annotations

import numpy as np

from cockpit.decision.cost_rule import challenge_cost, cost_matrix
from cockpit.eda import auc


def c_index(duration, event, risk) -> float:
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(event.astype(bool), duration, risk)[0])


def integrated_brier(train_d, train_e, test_d, test_e, S, times) -> float:
    """Score de Brier intégré (pondéré par l'inverse de la censure) sur la grille `times`."""
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv

    tr = Surv.from_arrays(train_e.astype(bool), np.maximum(train_d, 0.1))
    te_mask = test_d < train_d.max()
    te = Surv.from_arrays(test_e[te_mask].astype(bool), np.maximum(test_d[te_mask], 0.1))
    t = np.asarray(times, float)
    t = t[(t > np.maximum(test_d[te_mask], 0.1).min()) & (t < test_d[te_mask].max())]
    idx = [list(times).index(x) for x in t]
    return float(integrated_brier_score(tr, te, S[te_mask][:, idx], t))


def binary_auc(y_class, risk) -> float:
    return auc(np.asarray(risk, float), np.asarray(y_class) >= 1)


def bootstrap_cost(y_true, y_pred, n_boot=2000, seed=0) -> dict:
    """Coût total et intervalle de confiance à 95 % par rééchantillonnage des camions."""
    C = cost_matrix()
    per = C[y_true.astype(int), y_pred.astype(int)]
    rng = np.random.default_rng(seed)
    boots = per[rng.integers(0, len(per), (n_boot, len(per)))].sum(axis=1)
    return {"cost": float(per.sum()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "cost_per_vehicle": float(per.mean())}


def confusion(y_true, y_pred) -> np.ndarray:
    m = np.zeros((5, 5), dtype=int)
    np.add.at(m, (y_true.astype(int), y_pred.astype(int)), 1)
    return m


def reliability(prob, y, n_bins=10) -> dict:
    """Courbe de fiabilité par quantiles de probabilité prédite."""
    prob, y = np.asarray(prob), np.asarray(y, float)
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, prob, "right") - 1, 0, len(edges) - 2)
    pred = np.array([prob[b == i].mean() for i in range(len(edges) - 1)])
    obs = np.array([y[b == i].mean() for i in range(len(edges) - 1)])
    n = np.array([(b == i).sum() for i in range(len(edges) - 1)])
    return {"pred": pred.tolist(), "obs": obs.tolist(), "n": n.tolist()}


__all__ = ["c_index", "integrated_brier", "binary_auc", "bootstrap_cost", "confusion",
           "reliability", "challenge_cost"]
