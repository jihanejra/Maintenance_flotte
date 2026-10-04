"""Modèles de risque de panne, derrière une interface commune.

Chaque modèle de survie expose `survival(X, times)` : probabilité de ne pas être réparé
`t` pas de temps après l'instant de prédiction. On en déduit les probabilités des 5 classes du
challenge. Le classifieur LightGBM sert de comparaison directe avec le challenge.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

BINS = (6, 12, 24, 48)
MIN_DURATION = 0.1   # une coupure au dernier relevé d'un camion réparé peut avoir une durée nulle


def survival_to_classes(S: np.ndarray) -> np.ndarray:
    """S évaluée en (6, 12, 24, 48) -> probabilités des classes 0..4 (colonnes)."""
    s6, s12, s24, s48 = (S[:, i] for i in range(4))
    p = np.column_stack([s48, s24 - s48, s12 - s24, s6 - s12, 1 - s6])
    p = np.clip(p, 0, None)
    return p / p.sum(axis=1, keepdims=True)


class SurvivalModel:
    name = "base"

    def fit(self, X, duration, event, weight=None):
        raise NotImplementedError

    def survival(self, X, times) -> np.ndarray:
        raise NotImplementedError

    def class_proba(self, X) -> np.ndarray:
        return survival_to_classes(self.survival(X, BINS))

    def risk(self, X) -> np.ndarray:
        """Score de risque (plus grand = panne plus probable dans les 48 pas de temps)."""
        return 1 - self.survival(X, (48,))[:, 0]


# --- Cox ---------------------------------------------------------------------------------------

class CoxModel(SurvivalModel):
    """Cox à pénalité ridge, sur variables imputées, écrêtées, standardisées et dédoublonnées."""

    name = "cox"

    def __init__(self, alpha: float = 10.0, corr_threshold: float = 0.95, seed: int = 0):
        self.alpha, self.corr_threshold, self.seed = alpha, corr_threshold, seed

    def _prep_fit(self, X):
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0, self.med_)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.lo_, self.hi_ = np.percentile(Xi, [1, 99], axis=0)
        Xi = np.clip(Xi, self.lo_, self.hi_)
        self.mu_, self.sd_ = Xi.mean(axis=0), Xi.std(axis=0)
        keep = self.sd_ > 1e-8
        # Dédoublonnage glouton : on écarte une variable trop corrélée à une variable déjà gardée
        rng = np.random.default_rng(self.seed)
        sample = Xi[rng.choice(len(Xi), min(20000, len(Xi)), replace=False)]
        corr = np.corrcoef(sample[:, keep].T)
        idx = np.flatnonzero(keep)
        kept: list[int] = []
        for j in range(len(idx)):
            if all(abs(corr[j, k]) < self.corr_threshold for k in kept):
                kept.append(j)
        self.cols_ = idx[kept]
        return self._prep(X)

    def _prep(self, X):
        Xi = np.where(np.isnan(X), self.med_, X)
        Xi = np.clip(Xi, self.lo_, self.hi_)
        return ((Xi - self.mu_) / np.where(self.sd_ > 1e-8, self.sd_, 1))[:, self.cols_]

    def fit(self, X, duration, event, weight=None):
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        Z = self._prep_fit(X)
        y = _surv_array(duration, event)
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha, ties="breslow", n_iter=200)
        self.model_.fit(Z, y)
        return self

    def survival(self, X, times):
        lin = self._prep(X) @ self.model_.coef_
        step = self.model_.cum_baseline_hazard_
        pos = np.searchsorted(step.x, np.asarray(times, float), "right") - 1
        H0 = np.where(pos >= 0, step.y[np.clip(pos, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(lin), H0))


# --- Forêt aléatoire de survie -----------------------------------------------------------------

class RSFModel(SurvivalModel):
    """Random Survival Forest (scikit-survival), entraînée sur un sous-échantillon.

    Les durées sont arrondies à l'entier supérieur : la forêt stocke une courbe par feuille sur
    la grille des temps observés, et la réduire à ~500 valeurs limite fortement la mémoire.
    """

    name = "rsf"

    def __init__(self, n_estimators=60, min_samples_leaf=50, max_samples=20000, seed=0):
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.max_samples, self.seed = max_samples, seed

    def fit(self, X, duration, event, weight=None):
        from sksurv.ensemble import RandomSurvivalForest

        rng = np.random.default_rng(self.seed)
        n = len(X)
        if n > self.max_samples:
            p = None if weight is None else weight / weight.sum()
            idx = rng.choice(n, self.max_samples, replace=False, p=p)
        else:
            idx = np.arange(n)
        self.med_ = np.nan_to_num(np.nanmedian(X, axis=0))
        Xi = np.where(np.isnan(X[idx]), self.med_, X[idx])
        y = _surv_array(np.ceil(np.maximum(duration[idx], 1)), event[idx])
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt", n_jobs=2, random_state=self.seed,
        ).fit(Xi, y)
        return self

    def survival(self, X, times):
        out = []
        grid = self.model_.unique_times_
        pos = np.clip(np.searchsorted(grid, np.asarray(times, float), "right") - 1, 0, None)
        for start in range(0, len(X), 2000):     # par paquets : limite la mémoire
            Xi = np.where(np.isnan(X[start:start + 2000]), self.med_, X[start:start + 2000])
            S = self.model_.predict_survival_function(Xi, return_array=True)
            St = S[:, pos]
            St[:, np.asarray(times) < grid[0]] = 1.0
            out.append(St)
        return np.vstack(out)


# --- XGBoost AFT -------------------------------------------------------------------------------

class XGBAFTModel(SurvivalModel):
    """XGBoost en mode survie paramétrique (Accelerated Failure Time, loi log-normale).

    Le modèle prédit mu(x) = E[log T | x] ; S(t | x) = 1 - Phi((log t - mu(x)) / sigma).
    Les camions censurés donnent un intervalle [durée, +inf[.
    """

    name = "xgb_aft"

    def __init__(self, sigma=1.0, rounds=500, learning_rate=0.05, max_depth=5, seed=0):
        self.sigma, self.rounds, self.lr, self.max_depth, self.seed = (
            sigma, rounds, learning_rate, max_depth, seed)

    def params(self):
        return {"objective": "survival:aft", "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal", "aft_loss_distribution_scale": self.sigma,
                "tree_method": "hist", "learning_rate": self.lr, "max_depth": self.max_depth,
                "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.5,
                "lambda": 5.0, "seed": self.seed, "nthread": 4, "verbosity": 0}

    def fit(self, X, duration, event, weight=None):
        import xgboost as xgb

        d = np.maximum(duration, MIN_DURATION)
        dm = xgb.DMatrix(X, weight=weight, missing=np.nan)
        dm.set_float_info("label_lower_bound", d)
        dm.set_float_info("label_upper_bound", np.where(event == 1, d, np.inf))
        self.booster_ = xgb.train(self.params(), dm, num_boost_round=self.rounds)
        return self

    def mu(self, X):
        import xgboost as xgb

        return self.booster_.predict(xgb.DMatrix(X, missing=np.nan), output_margin=True)

    def survival(self, X, times):
        z = (np.log(np.asarray(times, float))[None, :] - self.mu(X)[:, None]) / self.sigma
        return norm.sf(z)


# --- Classifieur multiclasse -------------------------------------------------------------------

class LGBMClassifierModel:
    """LightGBM sur les 5 classes (échantillons à classe connue uniquement)."""

    name = "lgbm_clf"

    def __init__(self, rounds=400, seed=0):
        self.rounds, self.seed = rounds, seed

    def fit(self, X, label, weight=None):
        import lightgbm as lgb

        params = {"objective": "multiclass", "num_class": 5, "learning_rate": 0.05,
                  "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.5,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
                  "seed": self.seed, "verbose": -1, "num_threads": 4}
        self.model_ = lgb.train(params, lgb.Dataset(X, label, weight=weight),
                                num_boost_round=self.rounds)
        return self

    def class_proba(self, X):
        return self.model_.predict(X)

    def risk(self, X):
        return 1 - self.class_proba(X)[:, 0]


def _surv_array(duration, event):
    from sksurv.util import Surv

    return Surv.from_arrays(event=event.astype(bool), time=np.maximum(duration, MIN_DURATION))


SURVIVAL_MODELS = {"cox": CoxModel, "rsf": RSFModel, "xgb_aft": XGBAFTModel}
