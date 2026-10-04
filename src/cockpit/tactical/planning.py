"""Décisions tactiques : stock de pièces du mois et effectifs des prochaines semaines.

Chaque camion i a une probabilité p_i de remplacement du composant X sur la période (lue sur
sa courbe de survie calibrée). Le nombre de remplacements N = somme de Bernoulli indépendantes
suit une loi de Poisson-binomiale, calculée exactement par convolutions successives. On en
tire une distribution, pas seulement une moyenne : c'est elle qui fixe le stock et les effectifs
pour un niveau de service donné.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cockpit.config import load_config
from cockpit.economics import unit_costs


def poisson_binomial(p: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Loi exacte du nombre de succès de Bernoulli indépendantes de probabilités p."""
    p = np.clip(np.asarray(p, float), 0, 1)
    kmax = kmax or int(min(len(p), p.sum() + 12 * np.sqrt(p.sum() + 1) + 20))
    pmf = np.zeros(kmax + 1)
    pmf[0] = 1.0
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi
        pmf[0] *= 1 - pi
    return pmf / pmf.sum()


def period_probabilities(fleet: pl.DataFrame, start: int, length: int) -> np.ndarray:
    """P(panne entre start et start + length pas de temps | pas de panne avant start)."""
    s0 = np.ones(fleet.height) if start == 0 else fleet[f"S_{start}"].to_numpy()
    s1 = fleet[f"S_{start + length}"].to_numpy()
    return np.clip(1 - s1 / np.maximum(s0, 1e-9), 0, 1)


def spare_parts_plan(fleet: pl.DataFrame, overrides: dict | None = None) -> dict:
    """Niveau de stock (« order-up-to ») qui minimise coût de possession + coût de rupture.

    Demande pendant la période couverte = révision (4 semaines) + délai de livraison. Critère
    du marchand de journaux : stock optimal = plus petit s tel que F(s) ≥ cu / (cu + co), avec
    cu le coût d'une pièce manquante et co le coût de possession d'une pièce sur la période.
    """
    cfg = {**load_config("costs")["parts"], **(overrides or {})}
    eco = load_config("costs")["economics"]
    costs = unit_costs()
    horizon = cfg["review_period_weeks"] + cfg["lead_time_weeks"]
    p = period_probabilities(fleet, 0, horizon)
    pmf = poisson_binomial(p)
    cdf = np.cumsum(pmf)
    co = eco["part_cost"] * cfg["holding_rate_per_year"] * horizon / 52
    cu = cfg["stockout_extra_days"] * costs.downtime_day + cfg["express_shipping"]
    ratio = cu / (cu + co)
    s_opt = int(np.searchsorted(cdf, ratio))
    k = np.arange(len(pmf))
    table = []
    for s in range(max(0, s_opt - 6), min(len(pmf), s_opt + 7)):
        shortage = float((np.maximum(k - s, 0) * pmf).sum())
        leftover = float((np.maximum(s - k, 0) * pmf).sum())
        table.append({"stock": s, "service": float(cdf[s]), "ruptures_attendues": shortage,
                      "cout_attendu": cu * shortage + co * leftover})
    return {"horizon_weeks": horizon, "expected_demand": float(p.sum()),
            "pmf": pmf.tolist(), "critical_ratio": ratio, "cost_under": cu, "cost_over": co,
            "optimal_stock": s_opt, "table": table,
            "p90": int(np.searchsorted(cdf, 0.9)), "p99": int(np.searchsorted(cdf, 0.99))}


def staffing_plan(fleet: pl.DataFrame, weeks: int = 12, alarms_per_week: float = 0.0,
                  overrides: dict | None = None) -> pl.DataFrame:
    """Techniciens nécessaires par semaine pour le niveau de service visé.

    Charge = remplacements (pannes ou interventions préventives, loi Poisson-binomiale semaine
    par semaine) + inspections déclenchées par les alertes du modèle (fausses alertes incluses).
    """
    cfg = {**load_config("costs")["staffing"], **(overrides or {})}
    hours = load_config("costs")["workshop"]["intervention_hours"]
    rows = []
    for w in range(weeks):
        pmf = poisson_binomial(period_probabilities(fleet, w, 1))
        cdf = np.cumsum(pmf)
        q = int(np.searchsorted(cdf, cfg["service_level"]))
        load = (q + alarms_per_week) * hours
        rows.append({"semaine": w + 1, "remplacements_attendus": float((np.arange(len(pmf))
                     * pmf).sum()), f"remplacements_q{int(100 * cfg['service_level'])}": q,
                     "heures": load,
                     "techniciens": int(np.ceil(load / cfg["hours_per_technician_week"]))})
    return pl.DataFrame(rows)
