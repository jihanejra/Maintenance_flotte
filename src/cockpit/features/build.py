"""Variables explicatives calculées à un instant de coupure, sans fuite.

Pour une coupure au relevé de rang i d'un camion, seules les lignes 0..i sont utilisées :
- la première ligne (rythme moyen depuis le début du suivi),
- la ligne i - k (rythme récent sur les k derniers relevés),
- la ligne i (état au moment de la prédiction).
Les compteurs et les histogrammes sont cumulatifs : on travaille sur des différences divisées
par le temps écoulé (rythmes) et sur des parts (profils d'utilisation), pas sur les valeurs
brutes. Les valeurs manquantes sont propagées vers l'avant à l'intérieur de chaque camion
(dernière valeur connue), ce qui n'utilise que le passé.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cockpit.ingest.load import variable_groups


def _row_index(readouts: pl.DataFrame, cuts: pl.DataFrame, k: int) -> dict[str, np.ndarray]:
    """Positions globales (dans `readouts`, trié par camion puis temps) des lignes utiles."""
    vid = readouts["vehicle_id"].to_numpy()
    uniq, start = np.unique(vid, return_index=True)
    pos = np.searchsorted(uniq, cuts["vehicle_id"].to_numpy())
    if not np.array_equal(uniq[pos], cuts["vehicle_id"].to_numpy()):
        raise KeyError("coupure sur un camion absent des relevés")
    s = start[pos]
    i = cuts["cut_idx"].to_numpy().astype(np.int64)
    return {
        "first": s,
        "prev": s + np.maximum(i - 1, 0),
        "back": s + np.maximum(i - k, 0),
        "last": s + i,
    }


def _rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def _ffill(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(pl.col(cols).forward_fill().over("vehicle_id")).to_numpy()


def meta_features(readouts: pl.DataFrame, idx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t = readouts["time_step"].to_numpy()
    n = (idx["last"] - idx["first"] + 1).astype(np.float32)
    return {
        "age": t[idx["last"]],
        "n_readouts": n,
        "mean_gap": _rate(t[idx["last"]] - t[idx["first"]], n - 1),
        "last_gap": (t[idx["last"]] - t[idx["prev"]]).astype(np.float32),
    }


def counter_features(
    readouts: pl.DataFrame, idx: dict[str, np.ndarray], counters: list[str]
) -> dict[str, np.ndarray]:
    t = readouts["time_step"].to_numpy()
    x = _ffill(readouts, counters)
    dt_all = t[idx["last"]] - t[idx["first"]]
    dt_rec = t[idx["last"]] - t[idx["back"]]
    out = {}
    for j, c in enumerate(counters):
        name = c.removesuffix("_0")
        last, first, back = x[idx["last"], j], x[idx["first"], j], x[idx["back"], j]
        out[f"cnt_{name}_last"] = last
        out[f"cnt_{name}_rate_all"] = _rate(last - first, dt_all)
        rec = _rate(last - back, dt_rec)
        out[f"cnt_{name}_rate_recent"] = np.where(rec < 0, np.nan, rec).astype(np.float32)
        # Accélération de l'usure : rythme récent rapporté au rythme moyen.
        out[f"cnt_{name}_accel"] = _rate(out[f"cnt_{name}_rate_recent"],
                                         out[f"cnt_{name}_rate_all"])
    return out


def _profile_stats(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Moyenne et écart-type de l'indice de classe, parts des 20 % de classes basses/hautes."""
    nb = shares.shape[1]
    b = np.arange(nb, dtype=np.float32)
    mean = np.nansum(shares * b, axis=1)
    std = np.sqrt(np.nansum(shares * (b - mean[:, None]) ** 2, axis=1))
    tail = max(1, round(0.2 * nb))
    return mean, std, np.nansum(shares[:, :tail], axis=1), np.nansum(shares[:, -tail:], axis=1)


def histogram_features(
    readouts: pl.DataFrame, idx: dict[str, np.ndarray], var: str, cols: list[str]
) -> dict[str, np.ndarray]:
    t = readouts["time_step"].to_numpy()
    h = _ffill(readouts, cols)
    last, back = h[idx["last"]], h[idx["back"]]
    tot_last = last.sum(axis=1)
    recent = last - back
    # Rares cas où une classe redescend après une période manquante : l'accroissement récent
    # n'est alors pas interprétable, on le marque manquant plutôt que de garder une part négative.
    recent[(recent < 0).any(axis=1)] = np.nan
    tot_recent = recent.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sh_all = np.where(tot_last[:, None] > 0, last / tot_last[:, None], np.nan)
        sh_rec = np.where(tot_recent[:, None] > 0, recent / tot_recent[:, None], np.nan)

    out = {f"hist_{var}_b{b}_share": sh_all[:, b].astype(np.float32) for b in range(len(cols))}
    m_all, s_all, lo_all, hi_all = _profile_stats(sh_all)
    m_rec, s_rec, lo_rec, hi_rec = _profile_stats(sh_rec)
    no_recent = ~(tot_recent > 0)
    for arr in (m_rec, s_rec, lo_rec, hi_rec):
        arr[no_recent] = np.nan
    out |= {
        f"hist_{var}_total_rate_all": _rate(tot_last - h[idx["first"]].sum(axis=1),
                                            t[idx["last"]] - t[idx["first"]]),
        f"hist_{var}_total_rate_recent": _rate(tot_recent, t[idx["last"]] - t[idx["back"]]),
        f"hist_{var}_mean_bin": m_all.astype(np.float32),
        f"hist_{var}_std_bin": s_all.astype(np.float32),
        f"hist_{var}_low_share": lo_all.astype(np.float32),
        f"hist_{var}_high_share": hi_all.astype(np.float32),
        f"hist_{var}_recent_mean_bin": m_rec.astype(np.float32),
        f"hist_{var}_recent_std_bin": s_rec.astype(np.float32),
        f"hist_{var}_recent_high_share": hi_rec.astype(np.float32),
        f"hist_{var}_mean_bin_drift": (m_rec - m_all).astype(np.float32),
    }
    return out


def build_features(readouts: pl.DataFrame, cuts: pl.DataFrame, k: int) -> pl.DataFrame:
    """Une ligne par coupure. `readouts` doit être trié par (vehicle_id, time_step).

    `readouts` peut ne contenir qu'une partie des colonnes de mesures : cela permet au pipeline
    de traiter les variables par paquets pour limiter la mémoire.
    """
    idx = _row_index(readouts, cuts, k)
    counters, histograms = variable_groups(readouts.columns)
    feats = meta_features(readouts, idx)
    if counters:
        feats |= counter_features(readouts, idx, counters)
    for var, cols in histograms.items():
        feats |= histogram_features(readouts, idx, var, cols)
    return pl.DataFrame(feats)
