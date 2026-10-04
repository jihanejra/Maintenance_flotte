"""Analyse exploratoire : figures (`reports/figures`) et statistiques (`reports/eda_stats.json`).

Le notebook `notebooks/01_eda.ipynb` appelle ces mêmes fonctions et commente les résultats ;
les documents Word reprennent les statistiques du fichier JSON.
Usage : `python -m cockpit.eda`
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

from cockpit.config import PROJECT_ROOT, load_config, path
from cockpit.features.sampling import draw_cutoffs
from cockpit.ingest.load import (
    label_bins,
    load_labels,
    load_readouts,
    load_specs,
    load_tte,
    variable_groups,
)

log = logging.getLogger(__name__)

BLUE, ORANGE, GREY, RED = "#2563eb", "#ea580c", "#9ca3af", "#dc2626"
CLASS_NAMES = ["0 : > 48", "1 : 48-24", "2 : 24-12", "3 : 12-6", "4 : 6-0"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _save(fig: plt.Figure, name: str) -> str:
    out = path("figures") / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT)).replace("\\", "/")


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimateur de Kaplan-Meier : probabilité de ne pas avoir été réparé à chaque instant."""
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    times = np.unique(d[e == 1])
    at_risk = d.size - np.searchsorted(d, times, side="left")
    deaths = np.array([np.sum((d == t) & (e == 1)) for t in times])
    surv = np.cumprod(1 - deaths / at_risk)
    return np.concatenate([[0], times]), np.concatenate([[1.0], surv])


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """AUC par les rangs (statistique de Mann-Whitney), valeurs manquantes ignorées."""
    m = ~np.isnan(score)
    s, t = score[m], y[m]
    n1, n0 = t.sum(), (~t).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- Analyses ---------------------------------------------------------------------------------

def overview(stats: dict) -> None:
    tte = load_tte()
    per_split = {}
    for split in ("train", "validation", "test"):
        keys = load_readouts(split, columns=["vehicle_id", "time_step"])
        n = keys.group_by("vehicle_id").len()["len"]
        per_split[split] = {"vehicles": keys["vehicle_id"].n_unique(), "readouts": keys.height,
                            "readouts_per_vehicle_median": float(n.median()),
                            "readouts_per_vehicle_p10": float(n.quantile(0.1)),
                            "readouts_per_vehicle_p90": float(n.quantile(0.9))}
    rep = tte.filter(pl.col("in_study_repair") == 1)["length_of_study_time_step"]
    cen = tte.filter(pl.col("in_study_repair") == 0)["length_of_study_time_step"]
    stats["overview"] = per_split
    stats["events"] = {
        "repairs": rep.len(), "censored": cen.len(),
        "repair_rate": rep.len() / tte.height,
        "study_length_repaired_median": float(rep.median()),
        "study_length_censored_median": float(cen.median()),
        "study_length_max": float(tte["length_of_study_time_step"].max()),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    keys = load_readouts("train", columns=["vehicle_id", "time_step"])
    n = keys.group_by("vehicle_id").len()["len"].to_numpy()
    ax[0].hist(n, bins=60, color=BLUE)
    ax[0].set(title="Nombre de relevés par camion (train)", xlabel="relevés", ylabel="camions")
    bins = np.linspace(0, float(tte["length_of_study_time_step"].max()), 50)
    ax[1].hist(cen.to_numpy(), bins=bins, color=GREY, label="sans réparation (censuré)")
    ax[1].hist(rep.to_numpy(), bins=bins, color=ORANGE, label="réparé", alpha=0.9)
    ax[1].set(title="Durée de suivi selon l'issue", xlabel="time_step", ylabel="camions")
    ax[1].legend()
    stats.setdefault("figures", {})["overview"] = _save(fig, "01_overview")


def survival_curves(stats: dict) -> None:
    tte = load_tte().join(load_specs("train"), on="vehicle_id")
    d = tte["length_of_study_time_step"].to_numpy()
    e = tte["in_study_repair"].to_numpy()
    t, s = kaplan_meier(d, e)
    stats["km"] = {f"S({h})": float(s[np.searchsorted(t, h, side="right") - 1])
                   for h in (100, 200, 300, 400, 500)}

    # Segment le plus discriminant parmi les spécifications : écart max du taux de réparation
    rates = {}
    for spec in [c for c in tte.columns if c.startswith("Spec_")]:
        g = tte.group_by(spec).agg(pl.len(), pl.col("in_study_repair").mean().alias("rate"))
        g = g.filter(pl.col("len") >= 300)
        rates[spec] = g.sort("len", descending=True).to_dicts()
    spread = {k: max(r["rate"] for r in v) - min(r["rate"] for r in v)
              for k, v in rates.items() if len(v) > 1}
    best = max(spread, key=spread.get)
    stats["specs"] = {"repair_rate_by_category": rates, "spread": spread, "most_discriminant": best}

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].step(t, s, where="post", color=BLUE)
    ax[0].set(title="Kaplan-Meier : flotte entière", xlabel="time_step",
              ylabel="P(pas encore réparé)", ylim=(0.5, 1.01))
    for cat in [r[best] for r in rates[best][:4]]:
        m = (tte[best] == cat).to_numpy()
        tt, ss = kaplan_meier(d[m], e[m])
        ax[1].step(tt, ss, where="post", label=f"{best} = {cat} (n={m.sum()})")
    ax[1].set(title=f"Kaplan-Meier par {best}", xlabel="time_step", ylim=(0.5, 1.01))
    ax[1].legend(fontsize=8)
    stats["figures"]["survival"] = _save(fig, "02_kaplan_meier")


def readout_regularity(stats: dict) -> None:
    keys = load_readouts("train", columns=["vehicle_id", "time_step"])
    gaps = keys.select(pl.col("time_step").diff().over("vehicle_id")).drop_nulls()["time_step"]
    first = keys.group_by("vehicle_id").agg(pl.col("time_step").min())["time_step"]
    stats["gaps"] = {"median": float(gaps.median()), "p90": float(gaps.quantile(0.9)),
                     "p99": float(gaps.quantile(0.99)), "share_gap_1": float((gaps == 1).mean()),
                     "first_readout_median": float(first.median())}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(np.clip(gaps.to_numpy(), 0, 30), bins=30, color=BLUE)
    ax[0].set(title="Écart entre deux relevés consécutifs (tronqué à 30)",
              xlabel="time_step", ylabel="paires de relevés")
    ax[0].set_yscale("log")
    ax[1].hist(first.to_numpy(), bins=50, color=BLUE)
    ax[1].set(title="time_step du premier relevé", xlabel="time_step", ylabel="camions")
    stats["figures"]["gaps"] = _save(fig, "03_readout_gaps")


def missing_and_monotonicity(stats: dict) -> None:
    q = json.loads((path("processed") / "quality_report.json").read_text(encoding="utf-8"))
    tr = q["train"]
    stats["quality"] = {s: {"rows_with_missing": q[s]["rows_with_missing"], "rows": q[s]["rows"],
                            "counter_decreases": q[s]["counter_decreases"],
                            "histogram_bin_decreases": q[s]["histogram_bin_decreases"],
                            "histogram_missing_rows": q[s]["histogram_missing_rows"],
                            "vehicles_with_missing": q[s]["vehicles_with_missing"],
                            "vehicles": q[s]["vehicles"]}
                        for s in q}
    miss = tr["missing_by_column_top"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(miss)[:12]
    ax[0].barh(names[::-1], [100 * miss[n] / tr["rows"] for n in names[::-1]], color=BLUE)
    ax[0].set(title="Valeurs manquantes (train, % des relevés)", xlabel="%")

    ro = load_readouts("train", columns=["vehicle_id", "time_step", *tr["counters"][:3]])
    ids = ro.group_by("vehicle_id").len().filter(pl.col("len") > 60)["vehicle_id"][:5]
    c = tr["counters"][0]
    for vid in ids:
        v = ro.filter(pl.col("vehicle_id") == vid)
        ax[1].plot(v["time_step"], v[c], marker=".", ms=3, lw=1)
    ax[1].set(title=f"Compteur {c} : trajectoires de 5 camions", xlabel="time_step",
              ylabel="valeur cumulée")
    stats["figures"]["quality"] = _save(fig, "04_missing_counters")


def feature_signal(stats: dict) -> pl.DataFrame:
    """Pouvoir discriminant univarié des variables : AUC pour « réparation dans les 48 pas »."""
    f = pl.read_parquet(path("processed") / "features_train.parquet")
    known = f.filter(pl.col("class_label").is_not_null())
    y = (known["class_label"] >= 1).to_numpy()
    skip = {"vehicle_id", "cut_idx", "t_cut", "duration", "event", "class_label", "fold"}
    num = [c for c, t in known.schema.items() if c not in skip and t.is_numeric()]
    res = []
    for c in num:
        x = known[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
        a = auc(x, y)
        res.append({"feature": c, "auc": a, "strength": abs(a - 0.5),
                    "missing": float(np.isnan(x).mean())})
    res = pl.DataFrame(res).sort("strength", descending=True, nulls_last=True)

    cls = f["class_label"].value_counts().sort("class_label")
    stats["samples"] = {
        "train_cutoffs": f.height, "train_vehicles": f["vehicle_id"].n_unique(),
        "unknown_label": int(f["class_label"].null_count()),
        "class_counts_train_cutoffs": {str(r["class_label"]): r["count"] for r in cls.to_dicts()},
        "positive_rate_known": float(y.mean()), "n_features": len(num),
    }
    stats["top_features"] = res.head(20).to_dicts()

    top = res.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ORANGE if a > 0.5 else BLUE for a in top["auc"]]
    ax.barh(top["feature"][::-1], (top["auc"] - 0.5)[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Top 15 variables : AUC − 0,5 pour « réparation dans les 48 pas »",
           xlabel="AUC − 0,5 (orange : plus élevé avant panne ; bleu : plus faible)")
    stats["figures"]["signal"] = _save(fig, "05_feature_auc")

    # Distributions des 4 meilleures variables : sain vs panne proche
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, c in zip(axes, top["feature"][:4], strict=False):
        x = known[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
        lo, hi = np.nanquantile(x, [0.01, 0.99])
        b = np.linspace(lo, hi, 40)
        ax.hist(x[~y], bins=b, density=True, color=GREY, label="classe 0")
        ax.hist(x[y], bins=b, density=True, color=ORANGE, alpha=0.6, label="classes 1-4")
        ax.set_title(c, fontsize=9)
    axes[0].legend(fontsize=8)
    stats["figures"]["top_dists"] = _save(fig, "06_top_feature_distributions")
    return f


def histogram_profiles(stats: dict, f: pl.DataFrame) -> None:
    known = f.filter(pl.col("class_label").is_not_null())
    _, hists = variable_groups(load_readouts("validation").columns)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    gaps = {}
    for ax, var in zip(axes.ravel(), hists, strict=False):
        cols = [c for c in f.columns if c.startswith(f"hist_{var}_b") and c.endswith("_share")]
        cols.sort(key=lambda c: int(c.split("_b")[1].split("_")[0]))
        sel = known.select("class_label", pl.col(cols).fill_nan(None))
        healthy = sel.filter(pl.col("class_label") == 0).select(cols).mean().row(0)
        soon = sel.filter(pl.col("class_label") >= 1).select(cols).mean().row(0)
        x = np.arange(len(cols))
        ax.bar(x - 0.2, healthy, 0.4, color=GREY, label="classe 0")
        ax.bar(x + 0.2, soon, 0.4, color=ORANGE, label="classes 1-4")
        ax.set_title(f"Histogramme {var} ({len(cols)} classes)", fontsize=9)
        gaps[var] = float(np.abs(np.array(soon) - np.array(healthy)).sum() / 2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Profil d'utilisation moyen (part du temps par classe) : sain vs panne proche")
    stats["histogram_profile_tv_distance"] = gaps
    stats["figures"]["profiles"] = _save(fig, "07_histogram_profiles")


def label_distributions(stats: dict, f: pl.DataFrame) -> None:
    val, test = load_labels("validation"), load_labels("test")
    fv = pl.read_parquet(path("processed") / "features_validation.parquet")
    dist = {}
    for name, lab in [("validation", val), ("test", test)]:
        vc = lab["class_label"].value_counts().sort("class_label")
        dist[name] = {str(r["class_label"]): r["count"] for r in vc.to_dicts()}
    stats["labels"] = dist

    # Comparaison des deux stratégies de tirage des coupures avec la validation officielle
    s = load_config("data")["sampling"]
    keys = load_readouts("train", columns=["vehicle_id", "time_step"])
    uni = draw_cutoffs(keys, load_tte(), label_bins(), s["cutoffs_per_vehicle"],
                       s["min_readouts"], s["seed"])
    fv_lab = fv.select("t_cut", "class_label")
    samples = {"uniforme": uni, "calé sur la validation": f, "validation officielle": fv_lab}
    age_stats = {}
    for name, d in samples.items():
        k = d.filter(pl.col("class_label").is_not_null())
        y = (k["class_label"] >= 1).to_numpy()
        med = k.group_by("class_label").agg(pl.col("t_cut").median()).sort("class_label")
        age_stats[name] = {
            "median_age": float(d["t_cut"].median()),
            "median_age_by_class": {str(c): float(v) for c, v in med.rows()},
            "auc_age": auc(k["t_cut"].to_numpy().astype(float), y),
            "positives": int(y.sum()),
        }
    stats["cutoff_age"] = age_stats

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    b = np.linspace(0, 520, 53)
    for (name, d), col in zip(samples.items(), (GREY, BLUE, ORANGE), strict=True):
        ax[0].hist(d["t_cut"].to_numpy(), bins=b, density=True, histtype="step", lw=2,
                   color=col, label=f"{name} (médiane {d['t_cut'].median():.0f})")
    ax[0].set(title="Âge au moment de la prédiction", xlabel="time_step", ylabel="densité")
    ax[0].legend(fontsize=8)
    x = np.arange(5)
    for off, ((name, a), col) in zip((-0.27, 0, 0.27), zip(age_stats.items(),
                                     (GREY, BLUE, ORANGE), strict=True), strict=True):
        ax[1].bar(x + off, [a["median_age_by_class"].get(str(c), np.nan) for c in x], 0.27,
                  color=col, label=f"{name} (AUC âge {a['auc_age']:.2f})")
    ax[1].set(xticks=x, title="Âge médian à la coupure, par classe", ylabel="time_step")
    ax[1].set_xticklabels(CLASS_NAMES, fontsize=8)
    ax[1].legend(fontsize=8, loc="lower right")
    stats["figures"]["sampling"] = _save(fig, "08_cutoff_sampling")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(5)
    tr = f.filter(pl.col("class_label").is_not_null())["class_label"].value_counts()
    tr = dict(zip(tr["class_label"].to_list(), tr["count"].to_list(), strict=True))
    for off, (name, d, col) in zip(
        (-0.27, 0, 0.27),
        [("train (coupures)", {str(k): v for k, v in tr.items()}, BLUE),
         ("validation", dist["validation"], ORANGE), ("test", dist["test"], GREY)],
        strict=True,
    ):
        tot = sum(d.values())
        ax[0].bar(x + off, [100 * d.get(str(k), 0) / tot for k in x], 0.27, label=name,
                  color=col)
    ax[0].set_yscale("log")
    ax[0].set(xticks=x, title="Répartition des classes (%, échelle log)", ylabel="%")
    ax[0].set_xticklabels(CLASS_NAMES, fontsize=8)
    ax[0].legend(fontsize=8)
    # Pourquoi des classes inconnues : coupures de camions censurés proches de la fin du suivi
    unk = f.filter(pl.col("class_label").is_null())["duration"].to_numpy()
    ax[1].hist(f.filter(pl.col("event") == 0)["duration"].to_numpy(), bins=np.arange(0, 520, 8),
               color=GREY, label="camions censurés : toutes les coupures")
    ax[1].hist(unk, bins=np.arange(0, 520, 8), color=RED, label="dont classe inconnue (≤ 48)")
    ax[1].set(title="Temps restant jusqu'à la fin du suivi (censurés)", xlabel="time_step",
              ylabel="coupures")
    ax[1].legend(fontsize=8)
    stats["figures"]["labels"] = _save(fig, "10_labels")


def counter_correlations(stats: dict) -> None:
    f = pl.read_parquet(path("processed") / "features_train.parquet",
                        columns=None).select(pl.col("^cnt_.*_rate_all$"))
    m = f.fill_nan(None).drop_nulls()
    corr = np.corrcoef(m.to_numpy().T)
    names = [c.removeprefix("cnt_").removesuffix("_rate_all") for c in m.columns]
    stats["counter_rate_corr_max_offdiag"] = float(np.max(np.abs(corr - np.eye(len(names)))))
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Corrélation des rythmes d'usure (compteurs)")
    stats["figures"]["corr"] = _save(fig, "09_counter_correlations")


def run() -> dict:
    cfg = load_config("data")
    stats: dict = {"config": {**cfg["sampling"], **cfg["features"]}}
    overview(stats)
    survival_curves(stats)
    readout_regularity(stats)
    missing_and_monotonicity(stats)
    f = feature_signal(stats)
    histogram_profiles(stats, f)
    label_distributions(stats, f)
    counter_correlations(stats)
    out = PROJECT_ROOT / "reports" / "eda_stats.json"
    out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("statistiques : %s", out)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
