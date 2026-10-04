"""Décision opérationnelle : quels camions faire entrer à l'atelier cette semaine, et quel jour.

Variables : x[i, d] = 1 si le camion i est planifié le jour d (5 jours ouvrés).

Gain d'une intervention sur le camion i le jour d :
    (q_i − p1_i · d / 5) · (C_panne − C_remplacement + C_inspection) − C_inspection,
où q_i est la probabilité de panne dans la fenêtre de risque (6 pas de temps par défaut) et
p1_i · d / 5 la probabilité que le camion tombe en panne avant le jour prévu (p1_i = probabilité
de panne dans la semaine) : une panne survenue avant l'intervention n'est pas évitée.

Contraintes : chaque camion au plus une fois ; créneaux par jour (baies) ; heures de
techniciens qualifiés par jour ; pièces en stock sur la semaine ; disponibilité du camion
(camions engagés sur des tournées). Résolu avec HiGHS via `scipy.optimize.milp`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from cockpit.config import load_config
from cockpit.economics import UnitCosts, unit_costs

DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]


@dataclass
class WorkshopParams:
    bays: int
    slots_per_bay_per_day: int
    certified_technicians: int
    technician_hours_per_day: float
    intervention_hours: float
    parts_in_stock: int
    truck_availability: float
    risk_horizon: int
    max_candidates: int = 300

    @classmethod
    def from_config(cls, **overrides) -> WorkshopParams:
        return cls(**{**load_config("costs")["workshop"], **overrides})


def availability(vehicle_ids: np.ndarray, p: float, seed: int = 0) -> np.ndarray:
    """Disponibilité simulée des camions par jour (en réalité : planning des tournées)."""
    rng = np.random.default_rng(seed)
    return rng.random((len(vehicle_ids), len(DAYS))) < p


def plan_week(fleet: pl.DataFrame, params: WorkshopParams | None = None,
              costs: UnitCosts | None = None, seed: int = 0) -> dict:
    """`fleet` : une ligne par camion avec `vehicle_id` et les survies `S_1` … `S_52`."""
    params = params or WorkshopParams.from_config()
    costs = costs or unit_costs()
    q = 1 - fleet[f"S_{params.risk_horizon}"].to_numpy()
    p1 = 1 - fleet["S_1"].to_numpy()
    k = costs.breakdown - costs.replacement + costs.inspection   # valeur d'une panne évitée
    base = q * k - costs.inspection

    # Candidats : interventions rentables, les plus rentables d'abord (taille du problème bornée)
    cand = np.flatnonzero(base > 0)
    cand = cand[np.argsort(-base[cand])][: params.max_candidates]
    n, D = len(cand), len(DAYS)
    summary = {"n_fleet": fleet.height, "n_profitable": int((base > 0).sum()),
               "threshold": costs.threshold}
    if n == 0:
        return {"schedule": pl.DataFrame(), **summary, "expected_saving": 0.0, "status": "vide"}

    avail = availability(fleet["vehicle_id"].to_numpy()[cand], params.truck_availability, seed)
    gain = base[cand, None] - p1[cand, None] * (np.arange(D)[None, :] / D) * k
    c = -(gain * avail).ravel()                       # milp minimise
    ub = avail.ravel().astype(float)                  # camion indisponible : x = 0

    rows = n + 2 * D + 1
    A = lil_matrix((rows, n * D))
    lb_c, ub_c = np.full(rows, -np.inf), np.zeros(rows)
    for i in range(n):                                # au plus une intervention par camion
        A[i, i * D:(i + 1) * D] = 1
        ub_c[i] = 1
    for d in range(D):
        cols = np.arange(n) * D + d
        A[n + d, cols] = 1                            # créneaux de baies
        ub_c[n + d] = params.bays * params.slots_per_bay_per_day
        A[n + D + d, cols] = params.intervention_hours  # heures de techniciens qualifiés
        ub_c[n + D + d] = params.certified_technicians * params.technician_hours_per_day
    A[rows - 1, :] = 1                                # pièces disponibles sur la semaine
    ub_c[rows - 1] = params.parts_in_stock

    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb_c, ub_c),
               integrality=np.ones(n * D), bounds=Bounds(0, ub),
               options={"time_limit": 30})
    x = np.round(res.x).reshape(n, D).astype(bool) if res.x is not None else np.zeros((n, D), bool)
    i_sel, d_sel = np.nonzero(x)
    sched = pl.DataFrame({
        "vehicle_id": fleet["vehicle_id"].to_numpy()[cand[i_sel]],
        "jour": [DAYS[d] for d in d_sel],
        "jour_idx": d_sel,
        "proba_panne_fenetre": q[cand[i_sel]],
        "proba_panne_semaine": p1[cand[i_sel]],
        "gain_attendu": gain[i_sel, d_sel],
    }).sort(["jour_idx", "gain_attendu"], descending=[False, True])
    used_slots = x.sum(axis=0)
    return {
        "schedule": sched, **summary, "status": res.message,
        "n_candidates": n, "n_scheduled": int(x.sum()),
        "expected_saving": float(gain[x].sum()),
        "slots_used_per_day": used_slots.tolist(),
        "slots_per_day": params.bays * params.slots_per_bay_per_day,
        "unserved_profitable": int((base > 0).sum() - x.sum()),
        "binding": _binding(x, params),
    }


def _binding(x: np.ndarray, p: WorkshopParams) -> list[str]:
    """Contraintes saturées : ce qui limite le planning (utile pour dimensionner)."""
    out = []
    if x.sum() >= p.parts_in_stock:
        out.append("pièces en stock")
    if (x.sum(axis=0) >= p.bays * p.slots_per_bay_per_day).any():
        out.append("créneaux de baies")
    hours = x.sum(axis=0) * p.intervention_hours
    if (hours > p.certified_technicians * p.technician_hours_per_day - p.intervention_hours).any():
        out.append("heures de techniciens")
    return out
