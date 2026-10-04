"""Phase 2 : entraînement, calibration, décision au coût minimal et évaluation.

Protocole :
1. Validation croisée à 5 plis groupés par camion sur l'entraînement : prédictions hors
   échantillon (OOF), C-index, score de Brier intégré.
2. Calibration isotonique des courbes de survie sur les prédictions OOF (probabilités
   « naturelles », utilisées par les décisions de la phase 3).
3. Modèles finaux entraînés sur tout l'entraînement avec plusieurs graines (variabilité).
4. Décision au coût minimal, après recalage des probabilités sur la répartition des classes de
   la validation (règle de Bayes). Choix du modèle sur la validation, évaluation finale unique
   sur le test avec intervalle de confiance par bootstrap.
Chaque modèle est tracé dans MLflow (`mlflow.db` ; interface :
`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Usage : `python -m cockpit.survival.train`
"""

from __future__ import annotations

import json
import logging
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cockpit.config import PROJECT_ROOT, path
from cockpit.decision.cost_rule import (
    challenge_cost,
    class_prior,
    decide,
    most_likely,
    prior_shift,
)
from cockpit.eda import BLUE, GREY, ORANGE, RED, _save
from cockpit.survival.calibration import HORIZONS, SurvivalCalibrator
from cockpit.survival.dataset import load_all
from cockpit.survival.metrics import (
    binary_auc,
    bootstrap_cost,
    c_index,
    confusion,
    integrated_brier,
    reliability,
)
from cockpit.survival.models import (
    SURVIVAL_MODELS,
    LGBMClassifierModel,
    survival_to_classes,
)

log = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"
SEEDS = {"cox": [0], "xgb_aft": [0, 1, 2, 3, 4], "rsf": [0, 1], "lgbm_clf": [0, 1, 2, 3, 4]}
BIN_IDX = [HORIZONS.index(h) for h in (6, 12, 24, 48)]
FLEET_HORIZONS = list(range(1, 53))
MODEL_LABELS = {"cox": "Cox", "xgb_aft": "XGBoost AFT", "rsf": "Forêt de survie",
                "lgbm_clf": "LightGBM 5 classes", "gru_seq": "GRU (séquences)",
                "ens_gru_lgbm": "Ensemble GRU + LightGBM"}


def make_model(name: str, seed: int):
    if name == "lgbm_clf":
        return LGBMClassifierModel(seed=seed)
    return SURVIVAL_MODELS[name](seed=seed)


def fit_predict(name, seed, X_fit, d, e, lab, w, X_pred_list):
    """Entraîne un modèle et renvoie, pour chaque matrice, S aux HORIZONS (ou les classes)."""
    m = make_model(name, seed)
    if name == "lgbm_clf":
        known = ~np.isnan(lab)
        m.fit(X_fit[known], lab[known].astype(int), w[known])
        return m, [m.class_proba(X) for X in X_pred_list]
    m.fit(X_fit, d, e, w)
    return m, [m.survival(X, HORIZONS) for X in X_pred_list]


def to_classes(name, out, calibrator=None):
    if name == "lgbm_clf":
        return out
    S = calibrator.transform(out) if calibrator is not None else out
    return survival_to_classes(S[:, BIN_IDX])


def run() -> dict:
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("fleet-maintenance-phase2")
    MODELS_DIR.mkdir(exist_ok=True)

    tr, va, te, w = load_all()
    d = tr.meta["duration"].to_numpy().astype(float)
    e = tr.meta["event"].to_numpy().astype(int)
    fold = tr.meta["fold"].to_numpy()
    lab = tr.meta["class_label"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    yv = va.meta["class_label"].to_numpy().astype(int)
    yt = te.meta["class_label"].to_numpy().astype(int)
    prior_val = class_prior(yv)
    known = ~np.isnan(lab)

    results: dict = {"baselines": {}, "models": {}}
    for name, cls in (("toujours sain (0)", 0), ("toujours panne imminente (4)", 4)):
        results["baselines"][name] = {
            "validation": challenge_cost(yv, np.full_like(yv, cls)),
            "test": bootstrap_cost(yt, np.full_like(yt, cls)),
        }

    store = {}
    for name in ("cox", "xgb_aft", "rsf", "lgbm_clf"):
        t0 = time.time()
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "seeds": SEEDS[name], "n_train": len(d),
                               "n_features": tr.X.shape[1]})
            # 1. Validation croisée groupée par camion
            oof = None
            fold_cidx = []
            for k in range(5):
                fit_idx, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                _, (pred,) = fit_predict(name, 0, tr.X[fit_idx], d[fit_idx], e[fit_idx],
                                         lab[fit_idx], w[fit_idx], [tr.X[hold]])
                if oof is None:
                    oof = np.zeros((len(d), pred.shape[1]))
                oof[hold] = pred
                risk = 1 - pred[:, 0] if name == "lgbm_clf" else 1 - pred[:, HORIZONS.index(48)]
                fold_cidx.append(c_index(d[hold], e[hold], risk))
                log.info("%s pli %d : C-index %.3f", name, k, fold_cidx[-1])

            # 2. Calibration isotonique sur les prédictions hors échantillon
            calib = None
            metrics = {"cv_cindex_mean": float(np.mean(fold_cidx)),
                       "cv_cindex_std": float(np.std(fold_cidx))}
            if name != "lgbm_clf":
                calib = SurvivalCalibrator().fit(oof, d, e, w)
                oof_cal = calib.transform(oof)
                hold0 = fold == 0
                metrics["cv_ibs_raw"] = integrated_brier(d[~hold0], e[~hold0], d[hold0],
                                                         e[hold0], oof[hold0], HORIZONS)
                metrics["cv_ibs_calibrated"] = integrated_brier(
                    d[~hold0], e[~hold0], d[hold0], e[hold0], oof_cal[hold0], HORIZONS)
                y48 = (e == 1) & (d <= 48)
                k48 = (e == 1) | (d > 48)
                j48 = HORIZONS.index(48)
                results.setdefault("reliability_oof", {})[name] = {
                    "raw": reliability(1 - oof[k48, j48], y48[k48]),
                    "calibrated": reliability(1 - oof_cal[k48, j48], y48[k48]),
                }
            p_oof = to_classes(name, oof, calib)
            metrics["cv_auc48_known"] = binary_auc(lab[known], 1 - p_oof[known, 0])
            prior_model = p_oof.mean(axis=0)      # répartition implicite du modèle (Saerens)

            # 3. Modèles finaux, plusieurs graines
            preds_v, preds_t, seed_costs = [], [], []
            final_models = []
            for seed in SEEDS[name]:
                m, (pv, pt) = fit_predict(name, seed, tr.X, d, e, lab, w, [va.X, te.X])
                final_models.append(m)
                preds_v.append(pv)
                preds_t.append(pt)
                ct = prior_shift(to_classes(name, pt, calib), prior_model, prior_val)
                seed_costs.append(challenge_cost(yt, decide(ct)))
            out_v, out_t = np.mean(preds_v, axis=0), np.mean(preds_t, axis=0)
            pv_nat, pt_nat = to_classes(name, out_v, calib), to_classes(name, out_t, calib)
            pv, pt = prior_shift(pv_nat, prior_model, prior_val), prior_shift(pt_nat, prior_model,
                                                                              prior_val)
            # 4. Trois règles de décision, pour mesurer l'apport de chaque étape
            res = {
                "metrics": metrics,
                "validation": {
                    "auc": binary_auc(yv, 1 - pv[:, 0]),
                    "cost_most_likely": challenge_cost(yv, most_likely(pv_nat)),
                    "cost_min_no_prior": challenge_cost(yv, decide(pv_nat)),
                    "cost_min": challenge_cost(yv, decide(pv)),
                },
                "test": {
                    "auc": binary_auc(yt, 1 - pt[:, 0]),
                    "cost_most_likely": challenge_cost(yt, most_likely(pt_nat)),
                    "cost_min_no_prior": challenge_cost(yt, decide(pt_nat)),
                    **bootstrap_cost(yt, decide(pt)),
                    "seed_costs": seed_costs,
                    "confusion": confusion(yt, decide(pt)).tolist(),
                },
                "prior_model": prior_model.tolist(),
                "seconds": round(time.time() - t0),
            }
            results["models"][name] = res
            mlflow.log_metrics({**metrics, "val_cost": res["validation"]["cost_min"],
                                "val_auc": res["validation"]["auc"],
                                "test_cost": res["test"]["cost"], "test_auc": res["test"]["auc"]})
            store[name] = {"models": final_models, "calib": calib, "prior_model": prior_model,
                           "pt": pt, "out_t": out_t, "out_v": out_v}
            log.info("%s : coût validation %.0f, test %.0f (%ds)", name,
                     res["validation"]["cost_min"], res["test"]["cost"], res["seconds"])

    best = min(results["models"], key=lambda n: results["models"][n]["validation"]["cost_min"])
    best_surv = min((n for n in results["models"] if n != "lgbm_clf"),
                    key=lambda n: results["models"][n]["validation"]["cost_min"])
    results["best_model"], results["best_survival_model"] = best, best_surv
    results["prior_validation"] = prior_val.tolist()
    results["feature_names"] = tr.feature_names

    save_artifacts(store, best, best_surv, tr, va, te, prior_val)
    importance(store, tr.feature_names, results)
    figures(results, store, best, yt)
    out = PROJECT_ROOT / "reports" / "model_results.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("meilleur modèle (validation) : %s ; résultats : %s", best, out)
    return results


def save_artifacts(store, best, best_surv, tr, va, te, prior_val):
    """Pipeline de prédiction et risque de la flotte « actuelle » (camions du jeu de test)."""
    b = store[best_surv]
    bundle = {"model_name": best_surv, "models": b["models"], "calibrator": b["calib"],
              "prior_model": b["prior_model"], "prior_eval": prior_val,
              "feature_names": tr.feature_names}
    if best_surv != "rsf":            # la forêt pèse plusieurs centaines de Mo
        joblib.dump(bundle, MODELS_DIR / "survival_pipeline.joblib", compress=3)
    for split, ds, out in (("test", te, b["out_t"]), ("validation", va, b["out_v"])):
        write_fleet_risk(split, ds.meta, out, b["calib"], b["prior_model"], prior_val)


def write_fleet_risk(split, meta, S_raw, calib, prior_model, prior_val):
    """Courbes de survie calibrées de la flotte (1 à 52 pas) et décision du challenge."""
    S_cal = calib.transform(S_raw)
    S_grid = calib.survival_at(S_cal, FLEET_HORIZONS)
    p_eval = prior_shift(survival_to_classes(S_cal[:, BIN_IDX]), prior_model, prior_val)
    df = meta.select("vehicle_id", "t_cut", "class_label").with_columns(
        pl.Series("decision", decide(p_eval)),
        *[pl.Series(f"p_class{k}", p_eval[:, k]) for k in range(5)],
        *[pl.Series(f"S_{h}", S_grid[:, i]) for i, h in enumerate(FLEET_HORIZONS)],
    )
    df.write_parquet(path("processed") / f"fleet_risk_{split}.parquet")


def importance(store, names, results):
    if "xgb_aft" not in store:
        return
    gains = {}
    for m in store["xgb_aft"]["models"]:
        for k, v in m.booster_.get_score(importance_type="gain").items():
            gains[k] = gains.get(k, 0) + v
    top = sorted(gains.items(), key=lambda kv: -kv[1])[:20]
    results["xgb_importance"] = [{"feature": names[int(k[1:])], "gain": v} for k, v in top]


def figures(results, store, best, yt):
    figs = results.setdefault("figures", {})
    labels = MODEL_LABELS
    # Comparaison des coûts (test, IC 95 %)
    rows = [(k, v["test"]["cost"], v["test"]["ci_low"], v["test"]["ci_high"], GREY)
            for k, v in results["baselines"].items()]
    rows += [(labels[k], v["test"]["cost"], v["test"]["ci_low"], v["test"]["ci_high"],
              ORANGE if k == best else BLUE) for k, v in results["models"].items()]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    for i, (_n, c, lo, hi, col) in enumerate(rows):
        ax.barh(i, c, color=col)
        ax.errorbar(c, i, xerr=[[c - lo], [hi - c]], color="black", capsize=3, lw=1)
        ax.text(hi + 400, i, f"{c:,.0f}".replace(",", " "), va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(title="Coût officiel sur le jeu de test (IC 95 % par bootstrap)", xlabel="coût total")
    figs["costs"] = _save(fig, "21_model_costs")

    # Apport de chaque étape de la décision
    fig, ax = plt.subplots(figsize=(1.6 * len(results["models"]) + 2, 3.6))
    steps = [("cost_most_likely", "classe la plus probable"),
             ("cost_min_no_prior", "coût minimal"), ("cost", "coût minimal + recalage a priori")]
    x = np.arange(len(results["models"]))
    for j, (key, lab) in enumerate(steps):
        ax.bar(x + (j - 1) * 0.27, [v["test"][key] for v in results["models"].values()], 0.27,
               label=lab, color=[GREY, BLUE, ORANGE][j])
    ax.axhline(results["baselines"]["toujours panne imminente (4)"]["test"]["cost"], color=RED,
               ls="--", lw=1, label="baseline « toujours 4 »")
    ax.set_xticks(x, [labels[k] for k in results["models"]], fontsize=8, rotation=10)
    ax.set(title="Coût de test selon la règle de décision", ylabel="coût total")
    ax.legend(fontsize=8)
    figs["decision"] = _save(fig, "22_decision_rules")

    # Calibration OOF
    rel = results.get("reliability_oof", {})
    if rel:
        fig, axes = plt.subplots(1, len(rel), figsize=(4 * len(rel), 3.6), squeeze=False)
        for ax, (n, r) in zip(axes[0], rel.items(), strict=True):
            ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
            ax.plot(r["raw"]["pred"], r["raw"]["obs"], "o-", color=GREY, label="brut")
            ax.plot(r["calibrated"]["pred"], r["calibrated"]["obs"], "o-", color=ORANGE,
                    label="calibré (isotonique)")
            m = max(max(r["raw"]["pred"]), max(r["raw"]["obs"])) * 1.1
            ax.set(xlim=(0, m), ylim=(0, m), title=labels[n], xlabel="probabilité prédite")
        axes[0, 0].set_ylabel("fréquence observée")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Fiabilité de P(réparation dans les 48 pas), hors échantillon")
        figs["calibration"] = _save(fig, "23_calibration")

    # Matrice de confusion du meilleur modèle
    cm = np.array(results["models"][best]["test"]["confusion"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(np.log1p(cm), cmap="Blues")
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 3 else "black")
    ax.set(xticks=range(5), yticks=range(5), xlabel="classe prédite", ylabel="vraie classe",
           title=f"Test : {labels[best]}")
    figs["confusion"] = _save(fig, "24_confusion")

    # Variabilité selon la graine
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(results["models"]) + 1.2))
    for i, (_n, v) in enumerate(results["models"].items()):
        sc = v["test"]["seed_costs"]
        ax.scatter(sc, [i] * len(sc), color=BLUE, s=30)
        ax.scatter([v["test"]["cost"]], [i], color=ORANGE, marker="D", s=40)
    ax.set_yticks(range(len(results["models"])), [labels[k] for k in results["models"]])
    ax.set(title="Coût de test par graine (bleu) et moyenne des graines (losange)",
           xlabel="coût total")
    figs["seeds"] = _save(fig, "25_seed_variability")

    # Importance des variables (XGBoost AFT)
    if "xgb_importance" in results:
        imp = results["xgb_importance"][:15]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh([r["feature"] for r in imp][::-1], [r["gain"] for r in imp][::-1], color=BLUE)
        ax.set(title="XGBoost AFT : importance (gain cumulé sur les graines)", xlabel="gain")
        figs["importance"] = _save(fig, "26_importance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
