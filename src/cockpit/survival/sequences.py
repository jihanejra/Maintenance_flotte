"""Séquences de relevés pour le modèle récurrent (GRU).

Chaque relevé devient un vecteur d'entrée qui décrit l'état du camion à cet instant :
- temps : âge (time_step) et écart depuis le relevé précédent (relevés irréguliers) ;
- compteurs : niveau (log) et rythme depuis le relevé précédent (log de la variation / écart) ;
- histogrammes : profil cumulé (part du temps dans chaque classe) et rythme de remplissage.
Les valeurs manquantes sont propagées vers l'avant dans chaque camion (passé uniquement). Toutes
les entrées sont déjà sur une échelle logarithmique ou de proportion : on les stocke brutes en
float16 (~300 Mo pour l'entraînement, construit par paquets de camions pour limiter la mémoire)
et on les standardise à la volée, avec les statistiques de l'entraînement, au moment de découper
les fenêtres de relevés.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from cockpit.config import path
from cockpit.ingest.load import variable_groups


@dataclass
class SequenceStore:
    X: np.ndarray            # (lignes, F) float16 brut, trié par camion puis temps
    vehicle_ids: np.ndarray  # identifiants uniques triés
    starts: np.ndarray       # position de la première ligne de chaque camion
    feature_names: list[str]
    mu: np.ndarray | None = None   # statistiques de standardisation (entraînement)
    sd: np.ndarray | None = None

    def rows_for(self, vehicle_id: np.ndarray, cut_idx: np.ndarray, length: int):
        """Indices des `length` derniers relevés jusqu'à la coupure (et masque de remplissage)."""
        pos = np.searchsorted(self.vehicle_ids, vehicle_id)
        s = self.starts[pos]
        end = s + cut_idx.astype(np.int64)
        rows = end[:, None] - (length - 1 - np.arange(length))[None, :]
        valid = rows >= s[:, None]
        return np.where(valid, rows, 0), valid

    def window(self, vehicle_id, cut_idx, length) -> np.ndarray:
        """(n, length, F + 1) : relevés remplis à gauche par des zéros, plus un canal de masque."""
        rows, valid = self.rows_for(vehicle_id, cut_idx, length)
        x = (self.X[rows].astype(np.float32) - self.mu) / self.sd
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8, 8) * valid[..., None]
        return np.concatenate([x, valid[..., None]], axis=-1).astype(np.float32)


def step_features(ro: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    counters, hists = variable_groups(ro.columns)
    measures = [c for c in ro.columns if c not in ("vehicle_id", "time_step")]
    ro = ro.with_columns(pl.col(measures).forward_fill().over("vehicle_id"))
    dt = pl.col("time_step").diff().over("vehicle_id").fill_null(0)
    exprs = [(pl.col("time_step") / 100).alias("age"), (dt + 1).log().alias("log_gap")]
    for c in counters:
        name = c.removesuffix("_0")
        exprs.append((pl.col(c).clip(0) + 1).log().alias(f"cnt_{name}_log"))
        rate = (pl.col(c).diff().over("vehicle_id").clip(0) / pl.when(dt > 0).then(dt)
                .otherwise(None))
        exprs.append((rate.fill_null(0) + 1).log().alias(f"cnt_{name}_rate"))
    for v, cols in hists.items():
        tot = pl.sum_horizontal(cols)
        for c in cols:
            exprs.append((pl.col(c) / pl.when(tot > 0).then(tot).otherwise(None))
                         .alias(f"h{c}_share"))
        rate = tot.diff().over("vehicle_id").clip(0) / pl.when(dt > 0).then(dt).otherwise(None)
        exprs.append((rate.fill_null(0) + 1).log().alias(f"h{v}_rate"))
    out = ro.select("vehicle_id", *exprs)
    return out, out.columns[1:]


def build_store(split: str, stats: tuple[np.ndarray, np.ndarray] | None = None,
                chunk_vehicles: int = 4000) -> SequenceStore:
    """Construit (ou relit depuis le cache) les séquences d'un jeu, par paquets de camions."""
    cache = path("processed") / f"seq_{split}.npz"
    if not cache.exists():
        src = path("interim") / f"readouts_{split}.parquet"
        ids = np.sort(pl.scan_parquet(src).select("vehicle_id").unique().collect()
                      .to_series().to_numpy())
        parts, vids, names = [], [], None
        for k in range(0, len(ids), chunk_vehicles):
            chunk = ids[k:k + chunk_vehicles]
            ro = (pl.scan_parquet(src).filter(pl.col("vehicle_id").is_in(chunk.tolist()))
                  .collect().sort("vehicle_id", "time_step"))
            df, names = step_features(ro)
            parts.append(df.select(names).to_numpy().astype(np.float16))
            vids.append(df["vehicle_id"].to_numpy())
            del ro, df
        X, vid = np.concatenate(parts), np.concatenate(vids)
        uniq, starts = np.unique(vid, return_index=True)
        np.savez(cache, X=X, vehicle_ids=uniq, starts=starts, names=np.array(names))
    z = np.load(cache, allow_pickle=False)
    store = SequenceStore(z["X"], z["vehicle_ids"], z["starts"], [str(n) for n in z["names"]])
    if stats is None:                 # statistiques calculées sur l'entraînement uniquement
        sample = store.X[np.random.default_rng(0).choice(len(store.X),
                                                         min(300_000, len(store.X)),
                                                         replace=False)].astype(np.float32)
        mu, sd = np.nanmean(sample, axis=0), np.nanstd(sample, axis=0)
        stats = (np.nan_to_num(mu), np.where(sd > 1e-6, np.nan_to_num(sd, nan=1.0), 1.0))
    store.mu, store.sd = stats
    return store


def build_all() -> dict[str, SequenceStore]:
    train = build_store("train")
    stats = (train.mu, train.sd)
    return {"train": train, "validation": build_store("validation", stats),
            "test": build_store("test", stats)}
