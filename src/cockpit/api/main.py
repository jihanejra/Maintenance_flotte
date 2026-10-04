"""API du cockpit de maintenance : risque par camion, planning atelier, plan de pièces.

Les courbes de survie de la flotte sont calculées en lot (phase 2) et lues au démarrage ; les
plans d'atelier et de pièces sont recalculés à la demande, avec des hypothèses modifiables.
Lancement : `uvicorn cockpit.api.main:app --reload` puis http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from cockpit import __version__
from cockpit.config import path
from cockpit.economics import unit_costs
from cockpit.operational.workshop_milp import WorkshopParams, plan_week
from cockpit.tactical.planning import spare_parts_plan

app = FastAPI(title="Cockpit de maintenance de flotte", version=__version__,
              description="Risque de panne du composant X, planning atelier et stock de pièces.")
HORIZONS = list(range(1, 53))
CLASS_LABELS = ["> 48 pas", "48-24 pas", "24-12 pas", "12-6 pas", "< 6 pas"]


@lru_cache
def fleet() -> pl.DataFrame:
    f = path("processed") / "fleet_risk_test.parquet"
    if not f.exists():
        raise RuntimeError("Lancer d'abord `python -m cockpit.survival.train`.")
    return pl.read_parquet(f)


class RiskResponse(BaseModel):
    vehicle_id: int
    age_time_step: float
    survival: dict[int, float]
    failure_prob_6: float
    class_probabilities: dict[str, float]
    challenge_decision: int
    recommended_action: str
    profitability_threshold: float


class PlannedJob(BaseModel):
    vehicle_id: int
    jour: str
    proba_panne_fenetre: float
    gain_attendu: float


class WorkshopPlanResponse(BaseModel):
    n_scheduled: int
    n_profitable: int
    expected_saving_eur: float
    slots_used_per_day: list[int]
    binding_constraints: list[str]
    schedule: list[PlannedJob]


class PartsPlanResponse(BaseModel):
    horizon_weeks: int
    expected_demand: float
    optimal_stock: int
    critical_ratio: float
    stock_for_90pct: int
    stock_for_99pct: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vehicles": fleet().height, "version": __version__}


@app.get("/fleet/summary")
def fleet_summary(top: int = Query(10, ge=1, le=100)) -> dict:
    f = fleet().with_columns((1 - pl.col("S_6")).alias("p6"))
    thr = unit_costs().threshold
    return {
        "vehicles": f.height,
        "above_threshold": int((f["p6"] > thr).sum()),
        "threshold": thr,
        "expected_failures_next_4_steps": float((1 - f["S_4"]).sum()),
        "top_risk": f.sort("p6", descending=True).head(top)
        .select("vehicle_id", "p6", "t_cut").to_dicts(),
    }


@app.get("/risk/{vehicle_id}", response_model=RiskResponse)
def risk(vehicle_id: int) -> RiskResponse:
    row = fleet().filter(pl.col("vehicle_id") == vehicle_id)
    if row.height == 0:
        raise HTTPException(404, f"camion {vehicle_id} inconnu")
    r = row.row(0, named=True)
    thr = unit_costs().threshold
    p6 = 1 - r["S_6"]
    return RiskResponse(
        vehicle_id=vehicle_id, age_time_step=r["t_cut"],
        survival={h: round(r[f"S_{h}"], 5) for h in HORIZONS},
        failure_prob_6=p6,
        class_probabilities={CLASS_LABELS[k]: r[f"p_class{k}"] for k in range(5)},
        challenge_decision=r["decision"],
        recommended_action="planifier une intervention" if p6 > thr else "aucune action",
        profitability_threshold=thr,
    )


@app.get("/workshop-plan", response_model=WorkshopPlanResponse)
def workshop_plan(bays: int | None = Query(None, ge=0, le=20),
                  parts_in_stock: int | None = Query(None, ge=0),
                  certified_technicians: int | None = Query(None, ge=0),
                  seed: int = 0) -> WorkshopPlanResponse:
    overrides = {k: v for k, v in {"bays": bays, "parts_in_stock": parts_in_stock,
                                   "certified_technicians": certified_technicians}.items()
                 if v is not None}
    p = plan_week(fleet(), WorkshopParams.from_config(**overrides), seed=seed)
    sched = p["schedule"]
    jobs = [] if sched.height == 0 else [
        PlannedJob(**{k: r[k] for k in PlannedJob.model_fields}) for r in sched.to_dicts()]
    return WorkshopPlanResponse(
        n_scheduled=p.get("n_scheduled", 0), n_profitable=p["n_profitable"],
        expected_saving_eur=p["expected_saving"],
        slots_used_per_day=p.get("slots_used_per_day", [0] * 5),
        binding_constraints=p.get("binding", []), schedule=jobs)


@app.get("/parts-plan", response_model=PartsPlanResponse)
def parts_plan(lead_time_weeks: int | None = Query(None, ge=0, le=26),
               review_period_weeks: int | None = Query(None, ge=1, le=26)) -> PartsPlanResponse:
    overrides = {k: v for k, v in {"lead_time_weeks": lead_time_weeks,
                                   "review_period_weeks": review_period_weeks}.items()
                 if v is not None}
    p = spare_parts_plan(fleet(), overrides)
    return PartsPlanResponse(horizon_weeks=p["horizon_weeks"],
                             expected_demand=p["expected_demand"],
                             optimal_stock=p["optimal_stock"], critical_ratio=p["critical_ratio"],
                             stock_for_90pct=p["p90"], stock_for_99pct=p["p99"])
