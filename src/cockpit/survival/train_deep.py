"""Deep learning : GRU de survie sur les séquences de relevés, puis ensemble avec LightGBM.

Même protocole que `train.py` (plis groupés par camion, calibration isotonique hors
échantillon, plusieurs graines, recalage a priori, décision au coût minimal, choix sur la
validation, test évalué une fois). L'ensemble moyenne les probabilités de classes naturelles du
GRU et de LightGBM avec un poids fixé à l'avance (50/50), sans réglage sur la validation.
Les résultats sont ajoutés à `reports/model_results.json`.
Usage : `python -m cockpit.survival.train_deep` (après `python -m cockpit.survival.train`).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import polars as pl

from cockpit.config import PROJECT_ROOT
from cockpit.decision.cost_rule import challenge_cost, class_prior, decide, most_likely, prior_shift
from cockpit.survival.calibration import HORIZONS, SurvivalCalibrator
from cockpit.survival.dataset import load_all
from cockpit.survival.deep import GRUSurvival
from cockpit.survival.metrics import (
    binary_auc,
    bootstrap_cost,
    c_index,
    confusion,
    integrated_brier,
    reliability,
)
from cockpit.survival.models import survival_to_classes
from cockpit.survival.sequences import build_all
from cockpit.survival.train import BIN_IDX, MODELS_DIR, figures, fit_predict, write_fleet_risk

log = logging.getLogger(__name__)
GRU_SEEDS = [0, 1, 2]
LGBM_SEEDS = [0, 1, 2, 3, 4]
ENSEMBLE_WEIGHT = 0.5


def evaluate(p_oof, pv_list, pt_list, yv, yt, prior_val, metrics):
    """Même schéma de résultats que `train.py`."""
    prior_model = p_oof.mean(axis=0)
    pv_nat, pt_nat = np.mean(pv_list, axis=0), np.mean(pt_list, axis=0)
    pv, pt = prior_shift(pv_nat, prior_model, prior_val), prior_shift(pt_nat, prior_model,
                                                                      prior_val)
    seed_costs = [challenge_cost(yt, decide(prior_shift(p, prior_model, prior_val)))
                  for p in pt_list]
    return {
        "metrics": metrics,
        "validation": {"auc": binary_auc(yv, 1 - pv[:, 0]),
                       "cost_most_likely": challenge_cost(yv, most_likely(pv_nat)),
                       "cost_min_no_prior": challenge_cost(yv, decide(pv_nat)),
                       "cost_min": challenge_cost(yv, decide(pv))},
        "test": {"auc": binary_auc(yt, 1 - pt[:, 0]),
                 "cost_most_likely": challenge_cost(yt, most_likely(pt_nat)),
                 "cost_min_no_prior": challenge_cost(yt, decide(pt_nat)),
                 **bootstrap_cost(yt, decide(pt)), "seed_costs": seed_costs,
                 "confusion": confusion(yt, decide(pt)).tolist()},
        "prior_model": prior_model.tolist(),
    }


def run(quick: bool = False) -> dict:
    import mlflow
    import torch

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("fleet-maintenance-phase2")

    tr, va, te, w = load_all()
    stores = build_all()
    spec = [i for i, n in enumerate(tr.feature_names) if n.startswith("Spec_")]
    Z = {k: ds.X[:, spec].astype(np.float32) for k, ds in (("tr", tr), ("va", va), ("te", te))}
    m = tr.meta
    vid, cut = m["vehicle_id"].to_numpy(), m["cut_idx"].to_numpy()
    d, e = m["duration"].to_numpy().astype(float), m["event"].to_numpy().astype(int)
    fold = m["fold"].to_numpy()
    lab = m["class_label"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    known = ~np.isnan(lab)
    yv = va.meta["class_label"].to_numpy().astype(int)
    yt = te.meta["class_label"].to_numpy().astype(int)
    prior_val = class_prior(yv)
    vv, cv_ = va.meta["vehicle_id"].to_numpy(), va.meta["cut_idx"].to_numpy()
    vt, ct = te.meta["vehicle_id"].to_numpy(), te.meta["cut_idx"].to_numpy()
    kw = {"epochs": 2} if quick else {}
    folds = [0, 1] if quick else range(5)

    # --- GRU : validation croisée groupée par camion -----------------------------------------
    t0 = time.time()
    oof = np.zeros((len(d), len(HORIZONS)))
    cidx, epochs = [], []
    for k in folds:
        fit, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        g = GRUSurvival(seed=0, **kw).fit(stores["train"], vid[fit], cut[fit], Z["tr"][fit],
                                          d[fit], e[fit], w[fit])
        oof[hold] = g.survival(stores["train"], vid[hold], cut[hold], Z["tr"][hold])
        cidx.append(c_index(d[hold], e[hold], 1 - oof[hold, HORIZONS.index(48)]))
        epochs.append(len(g.history_))
        log.info("gru pli %d : C-index %.3f (%d époques)", k, cidx[-1], epochs[-1])
    done = np.isin(fold, list(folds))
    calib = SurvivalCalibrator().fit(oof[done], d[done], e[done], w[done])
    oof_cal = calib.transform(oof)
    h0 = fold == 0
    j48 = HORIZONS.index(48)
    k48 = done & ((e == 1) | (d > 48))
    y48 = (e == 1) & (d <= 48)
    gru_metrics = {
        "cv_cindex_mean": float(np.mean(cidx)), "cv_cindex_std": float(np.std(cidx)),
        "cv_ibs_raw": integrated_brier(d[~h0], e[~h0], d[h0], e[h0], oof[h0], HORIZONS),
        "cv_ibs_calibrated": integrated_brier(d[~h0], e[~h0], d[h0], e[h0], oof_cal[h0],
                                              HORIZONS),
        "cv_auc48_known": binary_auc(lab[known & done], 1 - oof_cal[known & done, j48]),
        "mean_epochs": float(np.mean(epochs)),
    }
    p_oof_gru = survival_to_classes(oof_cal[:, BIN_IDX])
    Sv, St, finals = [], [], []
    for seed in GRU_SEEDS[:1] if quick else GRU_SEEDS:
        g = GRUSurvival(seed=seed, **kw).fit(stores["train"], vid, cut, Z["tr"], d, e, w)
        finals.append(g)
        Sv.append(g.survival(stores["validation"], vv, cv_, Z["va"]))
        St.append(g.survival(stores["test"], vt, ct, Z["te"]))
    pv_gru = [survival_to_classes(calib.transform(s)[:, BIN_IDX]) for s in Sv]
    pt_gru = [survival_to_classes(calib.transform(s)[:, BIN_IDX]) for s in St]
    res_gru = evaluate(p_oof_gru[done], pv_gru, pt_gru, yv, yt, prior_val, gru_metrics)
    res_gru["seconds"] = round(time.time() - t0)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(finals[0].net_.state_dict(), MODELS_DIR / "gru_seq.pt")
    log.info("gru : coût validation %.0f, test %.0f", res_gru["validation"]["cost_min"],
             res_gru["test"]["cost"])

    # --- LightGBM (mêmes graines que train.py) et ensemble ------------------------------------
    t1 = time.time()
    oof_l = np.zeros((len(d), 5))
    for k in folds:
        fit, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        _, (p,) = fit_predict("lgbm_clf", 0, tr.X[fit], d[fit], e[fit], lab[fit], w[fit],
                              [tr.X[hold]])
        oof_l[hold] = p
    pv_l, pt_l = [], []
    for seed in LGBM_SEEDS[:1] if quick else LGBM_SEEDS:
        _, (pv, pt) = fit_predict("lgbm_clf", seed, tr.X, d, e, lab, w, [va.X, te.X])
        pv_l.append(pv)
        pt_l.append(pt)
    a = ENSEMBLE_WEIGHT
    p_oof_ens = a * p_oof_gru + (1 - a) * oof_l
    n_pairs = min(len(pv_gru), len(pv_l))
    pv_ens = [a * pv_gru[i] + (1 - a) * pv_l[i] for i in range(n_pairs)]
    pt_ens = [a * pt_gru[i] + (1 - a) * pt_l[i] for i in range(n_pairs)]
    ens_metrics = {"cv_cindex_mean": c_index(d[done], e[done], 1 - p_oof_ens[done, 0]),
                   "cv_cindex_std": 0.0,
                   "cv_auc48_known": binary_auc(lab[known & done], 1 - p_oof_ens[known & done, 0]),
                   "weight_gru": a}
    res_ens = evaluate(p_oof_ens[done], pv_ens, pt_ens, yv, yt, prior_val, ens_metrics)
    # Évaluation principale : moyenne de toutes les graines de chaque modèle
    res_ens_full = evaluate(p_oof_ens[done], [a * np.mean(pv_gru, 0) + (1 - a) * np.mean(pv_l, 0)],
                            [a * np.mean(pt_gru, 0) + (1 - a) * np.mean(pt_l, 0)], yv, yt,
                            prior_val, ens_metrics)
    res_ens_full["test"]["seed_costs"] = res_ens["test"]["seed_costs"]
    res_ens_full["seconds"] = round(time.time() - t1)
    log.info("ensemble : coût validation %.0f, test %.0f",
             res_ens_full["validation"]["cost_min"], res_ens_full["test"]["cost"])

    if quick:
        print(json.dumps({"gru": res_gru["validation"], "ens": res_ens_full["validation"],
                          "seconds": [res_gru["seconds"], res_ens_full["seconds"]]}, indent=1))
        return {}

    for name, res, params in (("gru_seq", res_gru, {"length": 32, "hidden": 64,
                                                    "seeds": GRU_SEEDS}),
                              ("ens_gru_lgbm", res_ens_full, {"weight_gru": a})):
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, **params})
            mlflow.log_metrics({**{k: v for k, v in res["metrics"].items()},
                                "val_cost": res["validation"]["cost_min"],
                                "test_cost": res["test"]["cost"], "test_auc": res["test"]["auc"]})

    # --- Fusion avec les résultats existants ----------------------------------------------------
    out = PROJECT_ROOT / "reports" / "model_results.json"
    results = json.loads(out.read_text(encoding="utf-8"))
    results["models"]["gru_seq"] = res_gru
    results["models"]["ens_gru_lgbm"] = res_ens_full
    results.setdefault("reliability_oof", {})["gru_seq"] = {
        "raw": reliability(1 - oof[k48, j48], y48[k48]),
        "calibrated": reliability(1 - oof_cal[k48, j48], y48[k48]),
    }
    results["best_model"] = min(results["models"],
                                key=lambda n: results["models"][n]["validation"]["cost_min"])
    surv = ("cox", "xgb_aft", "rsf", "gru_seq")
    best_surv = min((n for n in surv if n in results["models"]),
                    key=lambda n: results["models"][n]["validation"]["cost_min"])
    results["fleet_curves_updated"] = best_surv != results["best_survival_model"]
    if best_surv == "gru_seq":
        # Le GRU devient la source des courbes de survie de la flotte (phase 3 à relancer)
        for split, meta, S in (("test", te.meta, np.mean(St, 0)),
                               ("validation", va.meta, np.mean(Sv, 0))):
            write_fleet_risk(split, meta, S, calib, np.array(res_gru["prior_model"]), prior_val)
    results["best_survival_model"] = best_surv
    figures(results, {}, results["best_model"], yt)
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("meilleur modèle (validation) : %s ; courbes de flotte : %s",
             results["best_model"], best_surv)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(quick="--quick" in sys.argv)
