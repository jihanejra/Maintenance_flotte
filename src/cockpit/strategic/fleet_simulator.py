"""Décision stratégique : simulateur Monte-Carlo de la flotte sur plusieurs années.

Modèle de panne : loi de Weibull à fragilité gamma, ajustée par maximum de vraisemblance sur
les durées d'entraînement (réparations et censures). Chaque camion a sa fragilité z (usage plus
ou moins intense) ; le composant X tombe en panne selon S_i(a) = exp(-z (a / λ)^k), où a est
l'âge du composant. Après un remplacement (panne ou préventif), le composant repart à neuf.

Politiques comparées :
- corrective : on répare à la panne ;
- préventive à âge fixe : remplacement quand le composant atteint l'âge T ;
- prédictive : chaque semaine, le modèle réévalue la flotte et lève des alertes. Un camion dont
  la panne survient dans la fenêtre de risque est détecté avec la probabilité mesurée sur la
  validation (rappel), un camion sain déclenche une fausse alerte avec le taux mesuré lui aussi.
  Une alerte entraîne une inspection, puis un remplacement si la panne était imminente. Les
  alertes non servies faute de créneau ne s'accumulent pas : elles sont réévaluées la semaine
  suivante. Un camion inspecté sans anomalie n'est pas réalerté avant son prochain relevé
  (`cooldown` semaines), faute d'information nouvelle.
L'atelier a une capacité hebdomadaire : les pannes passent en priorité, les interventions
planifiées attendent s'il n'y a plus de créneau (et le camion peut tomber en panne entre-temps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize

from cockpit.config import load_config, path
from cockpit.economics import UnitCosts, unit_costs
from cockpit.ingest.load import load_readouts, load_tte

# --- Loi de panne ------------------------------------------------------------------------------

@dataclass
class WeibullGamma:
    k: float          # forme (> 1 : usure)
    lam: float        # échelle (pas de temps)
    theta: float      # variance de la fragilité

    def survival(self, t):
        return (1 + self.theta * (np.asarray(t) / self.lam) ** self.k) ** (-1 / self.theta)

    def sample_frailty(self, n, rng):
        return rng.gamma(1 / self.theta, self.theta, n)

    def residual(self, age, z, rng):
        """Temps restant avant panne d'un composant d'âge `age`, conditionnellement à sa survie."""
        u = rng.random(len(age))
        return (age ** self.k - self.lam ** self.k * np.log(u) / z) ** (1 / self.k) - age


def fit_failure_law() -> WeibullGamma:
    tte = load_tte()
    t = tte["length_of_study_time_step"].to_numpy().astype(float)
    e = tte["in_study_repair"].to_numpy().astype(float)

    def nll(par):
        k, lam, th = np.exp(par)
        x = (t / lam) ** k
        log_s = -np.log1p(th * x) / th
        log_h = np.log(k / lam) + (k - 1) * np.log(t / lam) - np.log1p(th * x)
        return -(e * log_h + log_s).sum()

    res = minimize(nll, x0=np.log([2.0, 600.0, 0.5]), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    k, lam, th = np.exp(res.x)
    return WeibullGamma(float(k), float(lam), float(th))


# --- Paramètres de la politique prédictive (issus de la phase 2) -------------------------------

def predictive_rates(threshold: float, horizon: int = 6) -> dict:
    """Rappel et taux de fausses alertes du modèle au seuil économique, sur la validation.

    Rappel : part des camions de classe 4 (panne dans les 6 pas de temps) dont la probabilité
    de panne à 6 pas dépasse le seuil. Fausses alertes : même proportion chez les camions de
    classe 0, ramenée à la semaine en divisant par l'écart moyen entre relevés (une nouvelle
    information n'arrive qu'à chaque relevé).
    """
    fr = pl.read_parquet(path("processed") / "fleet_risk_validation.parquet")
    q = 1 - fr[f"S_{horizon}"].to_numpy()
    y = fr["class_label"].to_numpy()
    recall = float((q[y == 4] > threshold).mean())
    fpr = float((q[y == 0] > threshold).mean())
    keys = load_readouts("train", columns=["vehicle_id", "time_step"])
    gap = float(keys.select(pl.col("time_step").diff().over("vehicle_id")).drop_nulls()
                .mean().item())
    return {"recall": recall, "fpr_per_check": fpr, "weekly_false_alarm": fpr / gap,
            "readout_gap": gap, "cooldown": int(round(gap)), "threshold": threshold}


# --- Simulation --------------------------------------------------------------------------------

@dataclass
class Scenario:
    policy: str                      # corrective | preventive | predictive | oracle
    bays: int = 3
    fleet_age_shift: float = 0.0     # vieillissement (+) ou renouvellement (-) de la flotte
    interval: int = 200              # âge de remplacement (préventive)
    recall: float = 0.5
    weekly_false_alarm: float = 0.01
    window: int = 6
    cooldown: int = 5                # semaines sans nouvelle alerte après une inspection négative
    label: str = ""
    extra: dict = field(default_factory=dict)


def study_lengths() -> np.ndarray:
    return load_tte()["length_of_study_time_step"].to_numpy().astype(float)


def initial_ages(n: int, shift: float, rng, lengths: np.ndarray | None = None) -> np.ndarray:
    """Âges de départ tirés dans la distribution des durées de suivi observées, décalés."""
    t = study_lengths() if lengths is None else lengths
    a = rng.choice(t, n) * rng.random(n)            # camion observé au hasard de son suivi
    return np.clip(a + shift, 0, None)


def simulate(sc: Scenario, law: WeibullGamma, costs: UnitCosts, n: int, weeks: int,
             seed: int, lengths: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    cfg = load_config("costs")
    slots = sc.bays * cfg["workshop"]["slots_per_bay_per_day"] * cfg["time"][
        "working_days_per_week"]
    age = initial_ages(n, sc.fleet_age_shift, rng, lengths)
    z = law.sample_frailty(n, rng)
    t_fail = law.residual(age, z, rng)               # temps restant avant panne
    queued = np.zeros(n, bool)
    cool = np.zeros(n)
    backlog_breakdowns = 0
    tot = {"breakdowns": 0, "replacements": 0, "inspections": 0, "downtime_days": 0.0,
           "cost": 0.0, "slots_used": 0, "waiting_breakdown_weeks": 0}
    p_detect = 1 - (1 - sc.recall) ** (1 / sc.window)  # détection hebdomadaire équivalente

    for _ in range(weeks):
        fails = t_fail < 1
        nf = int(fails.sum())
        tot["breakdowns"] += nf
        tot["cost"] += nf * costs.breakdown
        tot["downtime_days"] += nf * cfg["economics"]["downtime_days_breakdown"]
        # Les pannes passent en priorité ; au-delà de la capacité, le camion attend
        demand = nf + backlog_breakdowns
        served_b = min(demand, slots)
        backlog_breakdowns = demand - served_b
        tot["waiting_breakdown_weeks"] += backlog_breakdowns
        tot["downtime_days"] += backlog_breakdowns * 7
        tot["cost"] += backlog_breakdowns * 7 * costs.downtime_day
        free = slots - served_b
        tot["slots_used"] += served_b
        if nf:
            idx = np.flatnonzero(fails)
            age[idx] = 0
            t_fail[idx] = law.residual(age[idx], z[idx], rng) + t_fail[idx]
            queued[idx] = False

        # Interventions planifiées selon la politique
        if sc.policy == "preventive":
            queued |= age >= sc.interval
        elif sc.policy in ("predictive", "oracle"):
            at_risk = t_fail < sc.window
            if sc.policy == "oracle":
                alarm = at_risk
            else:
                u = rng.random(n)
                alarm = np.where(at_risk, u < p_detect, u < sc.weekly_false_alarm)
            queued = alarm & (cool <= 0)          # réévaluation hebdomadaire
        todo = np.flatnonzero(queued)
        if len(todo) and free > 0:
            # Priorité aux plus urgents : âge (préventif) ou ordre aléatoire (alertes)
            order = todo[np.argsort(-age[todo])] if sc.policy == "preventive" \
                else rng.permutation(todo)
            done = order[:free]
            tot["slots_used"] += len(done)
            queued[done] = False
            if sc.policy == "preventive":
                replace = done
            else:
                imminent = t_fail[done] < sc.window
                replace = done[imminent]
                insp = len(done) - len(replace)
                cool[done[~imminent]] = sc.cooldown
                tot["inspections"] += insp
                tot["cost"] += insp * costs.inspection
            tot["replacements"] += len(replace)
            tot["cost"] += len(replace) * costs.replacement
            eco = cfg["economics"]
            tot["downtime_days"] += (len(replace) * eco["downtime_days_replacement"]
                                     + (len(done) - len(replace)) * eco["downtime_days_inspection"])
            age[replace] = 0
            t_fail[replace] = law.residual(age[replace], z[replace], rng)

        age += 1
        t_fail -= 1
        cool -= 1

    years = weeks / 52
    return {**tot, "slots_capacity": slots * weeks,
            "cost_per_truck_year": tot["cost"] / (n * years),
            "breakdowns_per_100_truck_years": 100 * tot["breakdowns"] / (n * years),
            "downtime_days_per_truck_year": tot["downtime_days"] / (n * years),
            "utilization": tot["slots_used"] / (slots * weeks)}


def run_scenarios(scenarios: list[Scenario], n: int | None = None, years: int | None = None,
                  replications: int | None = None, costs: UnitCosts | None = None,
                  law: WeibullGamma | None = None,
                  lengths: np.ndarray | None = None) -> pl.DataFrame:
    cfg = load_config("costs")["strategy"]
    n, years = n or cfg["fleet_size"], years or cfg["years"]
    reps = replications or cfg["replications"]
    costs, law = costs or unit_costs(), law or fit_failure_law()
    lengths = study_lengths() if lengths is None else lengths
    rows = []
    for sc in scenarios:
        runs = [simulate(sc, law, costs, n, 52 * years, cfg["seed"] + r, lengths)
                for r in range(reps)]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        sd = float(np.std([r["cost_per_truck_year"] for r in runs]))
        rows.append({"scenario": sc.label or sc.policy, "policy": sc.policy, "bays": sc.bays,
                     "fleet_age_shift": sc.fleet_age_shift, "interval": sc.interval,
                     **agg, "cost_per_truck_year_sd": sd})
    return pl.DataFrame(rows)
