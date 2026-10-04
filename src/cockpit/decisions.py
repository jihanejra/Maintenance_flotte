"""Phase 3 : les trois niveaux de décision, à partir des courbes de survie calibrées.

La flotte « actuelle » est celle des 5 045 camions du jeu de test, observés à leur dernier relevé.
Usage : `python -m cockpit.decisions` (après `python -m cockpit.survival.train`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cockpit.config import PROJECT_ROOT, load_config, path
from cockpit.economics import UnitCosts, unit_costs
from cockpit.eda import BLUE, GREY, ORANGE, RED, _save
from cockpit.operational.workshop_milp import DAYS, WorkshopParams, plan_week
from cockpit.strategic.fleet_simulator import (
    Scenario,
    fit_failure_law,
    predictive_rates,
    run_scenarios,
    study_lengths,
)
from cockpit.tactical.planning import spare_parts_plan, staffing_plan

log = logging.getLogger(__name__)
POLICY_LABELS = {"corrective": "corrective", "preventive": "préventive (âge fixe)",
                 "predictive": "prédictive (modèle)", "oracle": "prédictive parfaite"}


def load_fleet(split: str = "test") -> pl.DataFrame:
    return pl.read_parquet(path("processed") / f"fleet_risk_{split}.parquet")


def recost(df: pl.DataFrame, c: UnitCosts) -> pl.Series:
    """Coût par camion-an recalculé depuis les compteurs simulés, pour d'autres coûts unitaires."""
    cfg = load_config("costs")["strategy"]
    total = (df["breakdowns"] * c.breakdown + df["replacements"] * c.replacement
             + df["inspections"] * c.inspection
             + df["waiting_breakdown_weeks"] * 7 * c.downtime_day)
    return total / (cfg["fleet_size"] * cfg["years"])


# --- Niveau opérationnel -----------------------------------------------------------------------

def operational(fleet: pl.DataFrame, out: dict) -> None:
    plan = plan_week(fleet)
    sched = plan.pop("schedule")
    sched.write_csv(PROJECT_ROOT / "reports" / "workshop_plan_week.csv")
    out["operational"] = {**plan, "schedule_head": sched.head(15).to_dicts()}

    # Sensibilité : économie attendue selon le nombre de baies
    sens = []
    for bays in (1, 2, 3, 4, 5, 6):
        p = plan_week(fleet, WorkshopParams.from_config(bays=bays, parts_in_stock=10_000,
                                                        certified_technicians=bays))
        sens.append({"bays": bays, "scheduled": p.get("n_scheduled", 0),
                     "saving": p["expected_saving"]})
    out["operational"]["bays_sensitivity"] = sens

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    if sched.height:
        for d in range(len(DAYS)):
            s = sched.filter(pl.col("jour_idx") == d)
            ax[0].scatter([d] * s.height, 100 * s["proba_panne_fenetre"], color=BLUE, s=25)
        ax[0].set_xticks(range(len(DAYS)), DAYS)
    ax[0].axhline(100 * plan["threshold"], color=RED, ls="--", lw=1, label="seuil de rentabilité")
    ax[0].set(title=f"Planning de la semaine : {plan.get('n_scheduled', 0)} camions",
              ylabel="P(panne dans 6 pas) %")
    ax[0].legend(fontsize=8)
    ax[1].plot([s["bays"] for s in sens], [s["saving"] / 1000 for s in sens], "o-", color=ORANGE)
    ax[1].set(title="Économie attendue de la semaine selon la capacité", xlabel="baies",
              ylabel="k€")
    out.setdefault("figures", {})["operational"] = _save(fig, "31_workshop_plan")


# --- Niveau tactique ---------------------------------------------------------------------------

def tactical(fleet: pl.DataFrame, out: dict, alarms_per_week: float) -> None:
    parts = spare_parts_plan(fleet)
    staff = staffing_plan(fleet, weeks=12, alarms_per_week=alarms_per_week)
    staff.write_csv(PROJECT_ROOT / "reports" / "staffing_plan.csv")
    out["tactical"] = {"parts": {k: v for k, v in parts.items() if k != "pmf"},
                       "staffing": staff.to_dicts(), "alarms_per_week": alarms_per_week}

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    pmf = np.array(parts["pmf"])
    k = np.arange(len(pmf))
    m = pmf > 1e-5
    ax[0].bar(k[m], pmf[m], color=[ORANGE if x == parts["optimal_stock"] else BLUE for x in k[m]])
    ax[0].axvline(parts["expected_demand"], color="black", ls=":", lw=1, label="demande moyenne")
    ax[0].set(title=f"Remplacements sur {parts['horizon_weeks']} semaines : stock optimal "
              f"{parts['optimal_stock']}", xlabel="nombre de pièces", ylabel="probabilité")
    ax[0].legend(fontsize=8)
    ax[1].bar(staff["semaine"], staff["techniciens"], color=BLUE)
    ax[1].set(title="Techniciens qualifiés nécessaires (niveau de service 95 %)",
              xlabel="semaine", ylabel="techniciens")
    out["figures"]["tactical"] = _save(fig, "32_parts_staffing")


# --- Niveau stratégique ------------------------------------------------------------------------

def strategic(out: dict) -> pl.DataFrame:
    cfg = load_config("costs")["strategy"]
    costs = unit_costs()
    law = fit_failure_law()
    rates = predictive_rates(costs.threshold)
    out["strategic"] = {"failure_law": law.__dict__, "predictive_rates": rates}
    # Entrées de simulation autonomes (API, tableau de bord, Docker sans données brutes)
    lengths = study_lengths()
    sample = np.random.default_rng(0).choice(lengths, 3000, replace=False)
    (PROJECT_ROOT / "reports" / "sim_inputs.json").write_text(json.dumps(
        {"failure_law": law.__dict__, "predictive_rates": rates,
         "study_lengths_sample": sample.tolist()}), encoding="utf-8")
    log.info("loi de panne %s ; taux du modèle %s", law, rates)

    # Âge de remplacement préventif : le meilleur intervalle sur la configuration actuelle
    grid = run_scenarios([Scenario("preventive", interval=T, label=f"T={T}")
                          for T in cfg["preventive_intervals"]], replications=4, law=law)
    best_T = int(grid.sort("cost_per_truck_year")["interval"][0])
    out["strategic"]["preventive_grid"] = grid.select("interval", "cost_per_truck_year",
                                                      "breakdowns_per_100_truck_years").to_dicts()

    base = {"recall": rates["recall"], "weekly_false_alarm": rates["weekly_false_alarm"],
            "cooldown": rates["cooldown"]}
    scenarios = []
    for shift, age_label in ((0, "actuelle"), (100, "vieillissante"), (-100, "renouvelée")):
        for bays in (3, 4, 5):
            for pol in ("corrective", "preventive", "predictive", "oracle"):
                scenarios.append(Scenario(pol, bays=bays, fleet_age_shift=shift, interval=best_T,
                                          label=f"{pol}|{bays}|{age_label}", **base))
    res = run_scenarios(scenarios, law=law, costs=costs)
    res = res.with_columns(pl.col("scenario").str.split("|").list.get(2).alias("fleet"))
    res.write_csv(PROJECT_ROOT / "reports" / "strategic_scenarios.csv")
    out["strategic"]["best_interval"] = best_T
    out["strategic"]["scenarios"] = res.drop("scenario").to_dicts()

    # Rentabilité de la maintenance prédictive (flotte actuelle, 3 baies)
    cur = res.filter((pl.col("fleet") == "actuelle") & (pl.col("bays") == 3))
    c = {r["policy"]: r["cost_per_truck_year"] for r in cur.to_dicts()}
    n = cfg["fleet_size"]
    run_cost = cfg["predictive_cost_per_truck_year"]
    ref = min(c["corrective"], c["preventive"])
    net = (ref - c["predictive"] - run_cost) * n
    out["strategic"]["business_case"] = {
        "reference_policy": "corrective" if c["corrective"] <= c["preventive"] else "preventive",
        "cost_per_truck_year": c, "annual_net_saving": net,
        "payback_months": cfg["predictive_setup_cost"] / net * 12 if net > 0 else None,
        "oracle_net_saving": (ref - c["oracle"] - run_cost) * n,
    }
    # Baie supplémentaire (politique prédictive, flotte actuelle)
    pred = res.filter((pl.col("fleet") == "actuelle") & (pl.col("policy") == "predictive"))
    pc = dict(zip(pred["bays"].to_list(), pred["cost_per_truck_year"].to_list(), strict=True))
    gain_bay = (pc[3] - pc[4]) * n - cfg["extra_bay_cost_per_year"]
    out["strategic"]["extra_bay"] = {
        "cost_per_truck_year_by_bays": pc, "annual_net_gain_4_bays": gain_bay,
        "payback_months": cfg["extra_bay_capex"] / gain_bay * 12 if gain_bay > 0 else None,
    }

    # Seuil d'alerte : le seuil économique (camion par camion) ignore la capacité de l'atelier ;
    # on teste des seuils plus élevés (moins de fausses alertes, moins de rappel) par simulation.
    curve = []
    for thr in (costs.threshold, 0.075, 0.10, 0.15, 0.20, 0.30):
        r = predictive_rates(thr)
        sim = run_scenarios([Scenario("predictive", bays=3, recall=r["recall"],
                                      weekly_false_alarm=r["weekly_false_alarm"],
                                      cooldown=r["cooldown"])], law=law, costs=costs,
                            replications=4)
        curve.append({"threshold": thr, "recall": r["recall"],
                      "weekly_false_alarm": r["weekly_false_alarm"],
                      "cost_per_truck_year": float(sim["cost_per_truck_year"][0]),
                      "net_saving_per_truck_year": float(ref - sim["cost_per_truck_year"][0]
                                                         - run_cost)})
    best_thr = max(curve, key=lambda x: x["net_saving_per_truck_year"])
    out["strategic"]["threshold_curve"] = curve
    out["strategic"]["best_threshold"] = best_thr
    bnet = best_thr["net_saving_per_truck_year"] * n
    out["strategic"]["business_case"]["best_threshold_annual_net_saving"] = bnet
    out["strategic"]["business_case"]["best_threshold_payback_months"] = (
        cfg["predictive_setup_cost"] / bnet * 12 if bnet > 0 else None)

    # Performance de modèle nécessaire : grille rappel x fausses alertes (flotte actuelle, 3 baies)
    recalls = [0.1, 0.25, 0.5, 0.75, 1.0]
    fas = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
    grid_sc = [Scenario("predictive", bays=3, recall=r, weekly_false_alarm=f,
                        cooldown=rates["cooldown"], label=f"{r}|{f}")
               for r in recalls for f in fas]
    g = run_scenarios(grid_sc, law=law, costs=costs, replications=3)
    net = (ref - g["cost_per_truck_year"] - run_cost).to_numpy().reshape(len(recalls), len(fas))
    out["strategic"]["performance_grid"] = {"recalls": recalls, "weekly_false_alarms": fas,
                                            "net_saving_per_truck_year": net.tolist()}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = np.abs(net).max()
    im = ax.imshow(net, cmap="RdYlGn", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    ax.grid(False)
    for i in range(len(recalls)):
        for j in range(len(fas)):
            ax.text(j, i, f"{net[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(fas)), [f"{100 * f:.2g} %" for f in fas])
    ax.set_yticks(range(len(recalls)), [f"{100 * r:.0f} %" for r in recalls])
    xm = np.interp(rates["weekly_false_alarm"], fas, range(len(fas)))
    ym = np.interp(rates["recall"], recalls, range(len(recalls)))
    for k, pt in enumerate(curve):
        ax.scatter([np.interp(pt["weekly_false_alarm"], fas, range(len(fas)))],
                   [np.interp(pt["recall"], recalls, range(len(recalls)))], marker="o", s=30,
                   color="black", label="modèle, autres seuils" if k == 1 else None)
    ax.scatter([xm], [ym], marker="*", s=250, color="black", label="modèle, seuil économique")
    ax.set(xlabel="fausses alertes par camion et par semaine", ylabel="rappel (pannes détectées)",
           title="Économie nette du prédictif (€/camion/an, programme inclus)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.colorbar(im)
    out["figures"]["performance_grid"] = _save(fig, "36_required_performance")

    # Sensibilité au coût d'une panne (comportements simulés fixes, coûts recalculés)
    sens = []
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        cc = replace(costs, breakdown=costs.breakdown * mult)
        row = {"multiplier": mult, "breakdown_cost": cc.breakdown}
        for pol in ("corrective", "preventive", "predictive", "oracle"):
            row[pol] = float(recost(cur.filter(pl.col("policy") == pol), cc)[0])
        row["predictive_net_vs_best"] = min(row["corrective"], row["preventive"]) - \
            row["predictive"] - run_cost
        sens.append(row)
    out["strategic"]["breakdown_cost_sensitivity"] = sens
    strategic_figures(res, sens, out, run_cost)
    return res


def strategic_figures(res, sens, out, run_cost):
    cols = {"corrective": GREY, "preventive": BLUE, "predictive": ORANGE, "oracle": RED}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), sharey=False)
    for a, fleet in zip(ax, ("actuelle", "vieillissante", "renouvelée"), strict=True):
        sub = res.filter(pl.col("fleet") == fleet)
        x = np.arange(3)
        for j, pol in enumerate(cols):
            s = sub.filter(pl.col("policy") == pol).sort("bays")
            extra = run_cost if pol in ("predictive", "oracle") else 0
            a.bar(x + (j - 1.5) * 0.2, s["cost_per_truck_year"] + extra, 0.2, color=cols[pol],
                  label=POLICY_LABELS[pol])
        a.set_xticks(x, ["3 baies", "4 baies", "5 baies"])
        a.set_title(f"Flotte {fleet}")
    ax[0].set_ylabel("€ par camion et par an")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Coût total de maintenance du composant X (coût du programme prédictif inclus)")
    out["figures"]["strategic"] = _save(fig, "33_strategic_scenarios")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cur = res.filter((pl.col("fleet") == "actuelle") & (pl.col("bays") == 3))
    metrics = [("breakdowns_per_100_truck_years", "pannes / 100 camions-an"),
               ("downtime_days_per_truck_year", "jours d'immobilisation / camion-an")]
    for a, (m, lab) in zip(ax, metrics, strict=True):
        a.bar([POLICY_LABELS[p] for p in cur["policy"]], cur[m],
              color=[cols[p] for p in cur["policy"]])
        a.set_title(lab)
        a.tick_params(axis="x", labelsize=8, rotation=15)
    out["figures"]["strategic_ops"] = _save(fig, "34_strategic_operations")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bc = [s["breakdown_cost"] / 1000 for s in sens]
    for pol in cols:
        extra = run_cost if pol in ("predictive", "oracle") else 0
        ax.plot(bc, [s[pol] + extra for s in sens], "o-", color=cols[pol],
                label=POLICY_LABELS[pol])
    ax.set(title="Sensibilité au coût d'une panne (flotte actuelle, 3 baies)",
           xlabel="coût complet d'une panne (k€)", ylabel="€ par camion et par an")
    ax.legend(fontsize=8)
    out["figures"]["sensitivity"] = _save(fig, "35_breakdown_cost_sensitivity")


def run() -> dict:
    scfg = load_config("costs")["strategy"]
    out: dict = {"unit_costs": unit_costs().__dict__, "threshold": unit_costs().threshold,
                 "strategy_fleet": scfg["fleet_size"], "strategy_reps": scfg["replications"],
                 "setup_cost": scfg["predictive_setup_cost"]}
    fleet = load_fleet("test")
    operational(fleet, out)
    res = strategic(out)
    alarms = float(res.filter((pl.col("policy") == "predictive") & (pl.col("bays") == 3)
                              & (pl.col("fleet") == "actuelle"))["inspections"][0]
                   / (52 * load_config("costs")["strategy"]["years"]))
    tactical(fleet, out, alarms_per_week=alarms)
    p = PROJECT_ROOT / "reports" / "decision_results.json"
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("résultats : %s", p)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
