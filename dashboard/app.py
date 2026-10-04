"""Tableau de bord du cockpit de maintenance : un onglet par niveau de décision.

Lancement : `streamlit run dashboard/app.py`
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

from cockpit.config import PROJECT_ROOT, load_config, path
from cockpit.economics import unit_costs
from cockpit.operational.workshop_milp import WorkshopParams, plan_week
from cockpit.strategic.fleet_simulator import Scenario, WeibullGamma, run_scenarios
from cockpit.tactical.planning import spare_parts_plan, staffing_plan

st.set_page_config(page_title="Cockpit maintenance flotte", page_icon="🚚", layout="wide")
REPORTS = PROJECT_ROOT / "reports"
POLICY = {"corrective": "Corrective", "preventive": "Préventive (âge fixe)",
          "predictive": "Prédictive (modèle)", "oracle": "Prédictive parfaite"}


@st.cache_data
def load_fleet() -> pl.DataFrame:
    return pl.read_parquet(path("processed") / "fleet_risk_test.parquet")


@st.cache_data
def load_json(name: str) -> dict:
    f = REPORTS / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@st.cache_data
def load_scenarios() -> pl.DataFrame | None:
    f = REPORTS / "strategic_scenarios.csv"
    return pl.read_csv(f) if f.exists() else None


fleet = load_fleet()
eco_cfg = load_config("costs")["economics"]
ws_cfg = load_config("costs")["workshop"]

# --- Barre latérale : hypothèses économiques communes -----------------------------------------
st.sidebar.header("Hypothèses de coûts (€)")
st.sidebar.caption("Absentes des données : à remplacer par les coûts réels de la flotte.")
overrides = {
    "customer_penalty": st.sidebar.number_input("Pénalités client par panne", 0, 20000,
                                                eco_cfg["customer_penalty"], 250),
    "towing_cost": st.sidebar.number_input("Remorquage", 0, 10000, eco_cfg["towing_cost"], 100),
    "downtime_cost_per_day": st.sidebar.number_input("Immobilisation par jour", 0, 5000,
                                                     eco_cfg["downtime_cost_per_day"], 50),
    "inspection_cost": st.sidebar.number_input("Inspection (fausse alerte)", 0, 2000,
                                               eco_cfg["inspection_cost"], 25),
    "part_cost": st.sidebar.number_input("Pièce (composant X)", 0, 10000,
                                         eco_cfg["part_cost"], 100),
}
costs = unit_costs(overrides)
st.sidebar.metric("Coût complet d'une panne", f"{costs.breakdown:,.0f} €".replace(",", " "))
st.sidebar.metric("Seuil de rentabilité d'une intervention", f"{100 * costs.threshold:.1f} %",
                  help="Probabilité de panne dans la fenêtre de risque au-delà de laquelle "
                       "intervenir coûte moins cher qu'attendre la panne.")

st.title("Cockpit de maintenance : flotte SCANIA (composant X)")
st.caption(f"{fleet.height:,} camions · courbes de survie calibrées issues de la phase 2 · "
           "1 pas de temps ≈ 1 semaine (hypothèse)".replace(",", " "))
tab_op, tab_tac, tab_strat, tab_model = st.tabs(
    ["Opérationnel : la semaine", "Tactique : le mois", "Stratégique : 5 ans", "Modèle"])

# --- Opérationnel -------------------------------------------------------------------------------
with tab_op:
    c1, c2, c3, c4 = st.columns(4)
    bays = c1.slider("Baies", 0, 10, ws_cfg["bays"])
    techs = c2.slider("Techniciens qualifiés", 0, 10, ws_cfg["certified_technicians"])
    stock = c3.slider("Pièces en stock", 0, 60, ws_cfg["parts_in_stock"])
    avail = c4.slider("Disponibilité des camions", 0.1, 1.0, ws_cfg["truck_availability"], 0.05)
    plan = plan_week(fleet, WorkshopParams.from_config(
        bays=bays, certified_technicians=techs, parts_in_stock=stock, truck_availability=avail),
        costs=costs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Camions au-dessus du seuil", plan["n_profitable"])
    m2.metric("Camions planifiés", plan.get("n_scheduled", 0))
    m3.metric("Économie attendue", f"{plan['expected_saving']:,.0f} €".replace(",", " "))
    m4.metric("Contraintes saturées", ", ".join(plan.get("binding", [])) or "aucune")
    sched = plan["schedule"]
    if sched.height:
        st.dataframe(sched.drop("jour_idx").with_columns(
            (pl.col("proba_panne_fenetre") * 100).round(1).alias("P(panne 6 pas) %"),
            (pl.col("proba_panne_semaine") * 100).round(2).alias("P(panne semaine) %"),
            pl.col("gain_attendu").round(0)).drop("proba_panne_fenetre", "proba_panne_semaine"),
            use_container_width=True, hide_index=True)
    st.subheader("Courbe de survie d'un camion")
    top = fleet.with_columns((1 - pl.col("S_6")).alias("p6")).sort("p6", descending=True)
    vid = st.selectbox("Camion (triés par risque décroissant)", top["vehicle_id"].to_list()[:300])
    r = fleet.filter(pl.col("vehicle_id") == vid).row(0, named=True)
    curve = pl.DataFrame({"pas de temps": list(range(1, 53)),
                          "P(pas de panne)": [r[f"S_{h}"] for h in range(1, 53)]})
    st.altair_chart(alt.Chart(curve.to_pandas()).mark_line(point=True).encode(
        x="pas de temps", y=alt.Y("P(pas de panne)", scale=alt.Scale(zero=False))),
        use_container_width=True)

# --- Tactique -----------------------------------------------------------------------------------
with tab_tac:
    p_cfg = load_config("costs")["parts"]
    c1, c2, c3 = st.columns(3)
    lead = c1.slider("Délai de réapprovisionnement (semaines)", 0, 12, p_cfg["lead_time_weeks"])
    review = c2.slider("Période de commande (semaines)", 1, 12, p_cfg["review_period_weeks"])
    hold = c3.slider("Coût de possession annuel (%)", 5, 60,
                     int(100 * p_cfg["holding_rate_per_year"])) / 100
    parts = spare_parts_plan(fleet, {"lead_time_weeks": lead, "review_period_weeks": review,
                                     "holding_rate_per_year": hold})
    m1, m2, m3 = st.columns(3)
    m1.metric("Demande attendue sur la période", f"{parts['expected_demand']:.1f}")
    m2.metric("Stock recommandé", parts["optimal_stock"],
              help=f"Ratio critique {parts['critical_ratio']:.2f} (coût de rupture / "
                   "(rupture + possession))")
    m3.metric("Stock pour 99 % de service", parts["p99"])
    pmf = np.array(parts["pmf"])
    k = np.flatnonzero(pmf > 1e-5)
    st.altair_chart(alt.Chart(pl.DataFrame({"pièces": k, "probabilité": pmf[k],
                                            "recommandé": k == parts["optimal_stock"]})
                              .to_pandas()).mark_bar().encode(
        x="pièces:O", y="probabilité", color=alt.Color("recommandé", legend=None,
                                                       scale=alt.Scale(range=["#2563eb",
                                                                              "#ea580c"]))),
        use_container_width=True)
    st.subheader("Effectifs des 12 prochaines semaines")
    alarms = load_json("decision_results.json").get("tactical", {}).get("alarms_per_week", 0.0)
    level = st.slider("Niveau de service visé", 0.80, 0.99, 0.95, 0.01)
    staff = staffing_plan(fleet, 12, alarms, {"service_level": level})
    st.altair_chart(alt.Chart(staff.to_pandas()).mark_bar().encode(
        x="semaine:O", y="techniciens", tooltip=list(staff.columns)), use_container_width=True)

# --- Stratégique --------------------------------------------------------------------------------
with tab_strat:
    sc = load_scenarios()
    dec = load_json("decision_results.json").get("strategic", {})
    strat_cfg = load_config("costs")["strategy"]
    if sc is None:
        st.warning("Lancer `python -m cockpit.decisions` pour générer les scénarios.")
    else:
        fleet_kind = st.radio("Flotte", ["actuelle", "vieillissante", "renouvelée"],
                              horizontal=True)
        run_cost = st.slider("Coût du programme prédictif (€/camion/an)", 0, 300,
                             strat_cfg["predictive_cost_per_truck_year"], 10)
        sub = sc.filter(pl.col("fleet") == fleet_kind)
        truck_years = strat_cfg["fleet_size"] * strat_cfg["years"]
        sub = sub.with_columns(((pl.col("breakdowns") * costs.breakdown
                                 + pl.col("replacements") * costs.replacement
                                 + pl.col("inspections") * costs.inspection
                                 + pl.col("waiting_breakdown_weeks") * 7 * costs.downtime_day)
                                / truck_years
                                + pl.when(pl.col("policy").is_in(["predictive", "oracle"]))
                                .then(run_cost).otherwise(0)).alias("€/camion/an"),
                               pl.col("policy").replace_strict(POLICY).alias("politique"))
        st.altair_chart(alt.Chart(sub.to_pandas()).mark_bar().encode(
            x=alt.X("politique:N", title=None), y="€/camion/an", color="politique:N",
            column=alt.Column("bays:O", title="baies"),
            tooltip=["politique", "bays", "€/camion/an", "breakdowns_per_100_truck_years",
                     "downtime_days_per_truck_year", "utilization"]))
        cur = {r["policy"]: r["€/camion/an"] for r in sub.filter(pl.col("bays") == 3).to_dicts()}
        ref = min(cur["corrective"], cur["preventive"])
        net = (ref - cur["predictive"]) * strat_cfg["fleet_size"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Économie nette annuelle du prédictif", f"{net:,.0f} €".replace(",", " "))
        m2.metric("Délai de rentabilité",
                  f"{12 * strat_cfg['predictive_setup_cost'] / net:.0f} mois" if net > 0
                  else "non rentable")
        m3.metric("Potentiel d'un modèle parfait",
                  f"{(ref - cur['oracle']) * strat_cfg['fleet_size']:,.0f} €/an".replace(",", " "))
        st.caption("Coûts recalculés avec les hypothèses de la barre latérale à partir des "
                   "événements simulés (pannes, remplacements, inspections).")

        st.subheader("Simulation à la demande")
        sim = load_json("sim_inputs.json")
        rates = sim.get("predictive_rates", {})
        c1, c2, c3 = st.columns(3)
        recall = c1.slider("Rappel du modèle (pannes détectées)", 0.0, 1.0,
                           float(rates.get("recall", 0.5)), 0.05)
        fa = c2.slider("Fausses alertes par camion et par semaine (%)", 0.0, 2.0,
                       100 * float(rates.get("weekly_false_alarm", 0.005)), 0.05) / 100
        sim_bays = c3.slider("Baies (simulation)", 1, 8, 3)
        if st.button("Simuler 5 ans (flotte de 2 000 camions, 3 tirages)") and sim:
            law = WeibullGamma(**sim["failure_law"])
            lengths = np.asarray(sim["study_lengths_sample"])
            shift = {"actuelle": 0, "vieillissante": 100, "renouvelée": -100}[fleet_kind]
            res = run_scenarios(
                [Scenario(p, bays=sim_bays, fleet_age_shift=shift, recall=recall,
                          weekly_false_alarm=fa, interval=dec.get("best_interval", 200))
                 for p in ("corrective", "preventive", "predictive")],
                n=2000, replications=3, costs=costs, law=law, lengths=lengths)
            res = res.with_columns(pl.col("policy").replace_strict(POLICY),
                                   (pl.col("cost_per_truck_year")
                                    + pl.when(pl.col("policy") == "predictive")
                                    .then(run_cost).otherwise(0)).alias("€/camion/an"))
            st.dataframe(res.select("policy", "€/camion/an", "breakdowns_per_100_truck_years",
                                    "downtime_days_per_truck_year", "utilization"),
                         hide_index=True, use_container_width=True)

# --- Modèle -------------------------------------------------------------------------------------
with tab_model:
    mr = load_json("model_results.json")
    if mr:
        rows = [{"modèle": k, "coût validation": v["validation"]["cost_min"],
                 "coût test": v["test"]["cost"], "IC 95 % bas": v["test"]["ci_low"],
                 "IC 95 % haut": v["test"]["ci_high"], "AUC test": round(v["test"]["auc"], 3),
                 "C-index (CV)": round(v["metrics"]["cv_cindex_mean"], 3)}
                for k, v in mr["models"].items()]
        rows += [{"modèle": k, "coût validation": v["validation"], "coût test": v["test"]["cost"]}
                 for k, v in mr["baselines"].items()]
        st.dataframe(pl.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Modèle retenu : {mr['best_model']} · courbes de survie de la flotte : "
                   f"{mr['best_survival_model']}")
        for key in ("costs", "decision", "calibration"):
            f = mr.get("figures", {}).get(key)
            if f and (PROJECT_ROOT / f).exists():
                st.image(str(PROJECT_ROOT / f))
