"""Matrices de modélisation à partir des tables `features_{split}.parquet`.

- Variables numériques telles quelles (les modèles d'arbres gèrent les NaN) ;
- spécifications encodées en indicatrices (catégories fréquentes, le reste regroupé) ;
- poids d'échantillon qui corrigent l'écart d'âge restant entre coupures d'entraînement et
  coupures de validation (conclusion de l'analyse exploratoire).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from cockpit.config import load_config, path
from cockpit.features.sampling import age_density_weights

ID_COLS = ["vehicle_id", "cut_idx", "t_cut", "duration", "event", "class_label", "fold"]
SPEC_COLS = [f"Spec_{i}" for i in range(8)]
MIN_CATEGORY_SIZE = 200


@dataclass
class Dataset:
    X: np.ndarray            # (n, p) float32, NaN autorisés
    feature_names: list[str]
    meta: pl.DataFrame       # identifiants et cibles


def load_split(split: str) -> pl.DataFrame:
    return pl.read_parquet(path("processed") / f"features_{split}.parquet")


def spec_categories(train: pl.DataFrame) -> dict[str, list[str]]:
    """Catégories gardées pour chaque spécification (au moins 200 camions à l'entraînement)."""
    veh = train.unique("vehicle_id")
    return {
        c: sorted(veh.group_by(c).len().filter(pl.col("len") >= MIN_CATEGORY_SIZE)[c].to_list())
        for c in SPEC_COLS
    }


def numeric_features(df: pl.DataFrame) -> list[str]:
    return [c for c, t in df.schema.items()
            if c not in ID_COLS and c not in SPEC_COLS and t.is_numeric()]


def to_matrix(df: pl.DataFrame, num_cols: list[str], cats: dict[str, list[str]]) -> Dataset:
    num = df.select(pl.col(num_cols).cast(pl.Float32).fill_null(np.nan)).to_numpy()
    onehot, names = [], []
    for c, levels in cats.items():
        for lv in levels:
            onehot.append((df[c] == lv).cast(pl.Float32).to_numpy())
            names.append(f"{c}={lv}")
    X = np.column_stack([num, *onehot]).astype(np.float32) if onehot else num
    meta = df.select([c for c in ID_COLS if c in df.columns])
    return Dataset(X=X, feature_names=num_cols + names, meta=meta)


def age_weights(train: pl.DataFrame, validation: pl.DataFrame) -> np.ndarray:
    """Poids w(âge) qui rapprochent la distribution d'âge des coupures de celle de la validation."""
    width = load_config("data")["sampling"]["age_bin_width"]
    edges, w = age_density_weights(train["t_cut"].to_numpy(), validation["t_cut"].to_numpy(),
                                   width)
    b = np.clip(np.searchsorted(edges, train["t_cut"].to_numpy(), "right") - 1, 0, len(w) - 1)
    wt = np.clip(w[b], 0.2, 5.0)        # bornes : éviter qu'une poignée de lignes domine
    return (wt / wt.mean()).astype(np.float32)


def load_all() -> tuple[Dataset, Dataset, Dataset, np.ndarray]:
    tr, va, te = load_split("train"), load_split("validation"), load_split("test")
    cats = spec_categories(tr)
    cols = numeric_features(tr)
    return to_matrix(tr, cols, cats), to_matrix(va, cols, cats), to_matrix(te, cols, cats), \
        age_weights(tr, va)
