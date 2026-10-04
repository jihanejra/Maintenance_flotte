"""Construction des échantillons (camion, instant de coupure) et de leurs cibles.

En validation et en test, SCANIA a coupé chaque camion à un relevé tiré au hasard et donné la
classe du temps restant avant réparation. Pour que l'entraînement ressemble à l'évaluation, on
fait la même chose sur le jeu d'entraînement : on tire plusieurs coupures par camion et on
calcule, à partir du time-to-event, la durée restante (cible de survie) et la classe (cible du
challenge).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def time_to_class(remaining: np.ndarray, bins: list[int]) -> np.ndarray:
    """Classe officielle : 4 si reste <= 6, 3 si <= 12, 2 si <= 24, 1 si <= 48, sinon 0."""
    edges = np.asarray(bins)
    return (len(edges) - np.searchsorted(edges, remaining, side="left")).astype(np.int8)


def age_density_weights(
    train_ages: np.ndarray, target_ages: np.ndarray, bin_width: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Poids w(âge) = densité cible / densité des relevés d'entraînement, par tranches d'âge.

    Sert à tirer les coupures d'entraînement pour que leur âge ressemble à celui des coupures
    de validation (correction d'un décalage de covariable : seuls les âges de validation sont
    utilisés, jamais ses classes).
    """
    edges = np.arange(0, max(train_ages.max(), target_ages.max()) + bin_width, bin_width)
    p_train, _ = np.histogram(train_ages, edges, density=True)
    p_target, _ = np.histogram(target_ages, edges, density=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(p_train > 0, p_target / p_train, 0.0)
    return edges, w


def draw_cutoffs(
    readouts: pl.DataFrame,
    tte: pl.DataFrame,
    bins: list[int],
    cutoffs_per_vehicle: int,
    min_readouts: int,
    seed: int,
    age_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Tire des coupures parmi les relevés de chaque camion.

    Sans `age_weights`, le tirage est uniforme parmi les relevés. Avec `age_weights`
    (sortie de `age_density_weights`), chaque relevé est tiré avec une probabilité
    proportionnelle au poids de son âge.

    Retourne une ligne par (camion, coupure) avec :
    - `cut_idx` : rang du relevé de coupure (0 = premier relevé du camion) ;
    - `t_cut` : time_step de la coupure ;
    - `duration`, `event` : temps restant jusqu'à la fin d'étude et indicateur de réparation
      (format survie : event = 0 signifie censuré) ;
    - `class_label` : classe du challenge, nulle quand elle est inconnue (camion censuré
      moins de 48 pas de temps après la coupure : on ne sait pas s'il tombera en panne).
    """
    rng = np.random.default_rng(seed)
    steps = readouts.select("vehicle_id", "time_step").with_columns(
        pl.int_range(pl.len()).over("vehicle_id").alias("cut_idx")
    )
    eligible = steps.filter(pl.col("cut_idx") >= min_readouts - 1)

    # Tirage sans remise pondéré (méthode d'Efraimidis-Spirakis) : clé u^(1/w), on garde les
    # `cutoffs_per_vehicle` plus grandes clés de chaque camion. Poids égaux = tirage uniforme.
    u = rng.random(eligible.height)
    if age_weights is None:
        key = u
    else:
        edges, w = age_weights
        b = np.clip(np.searchsorted(edges, eligible["time_step"].to_numpy(), "right") - 1,
                    0, len(w) - 1)
        wt = np.maximum(w[b], 1e-6)  # poids minimal : tout relevé reste tirable
        key = np.log(u) / wt
    eligible = eligible.with_columns(pl.Series("u", -key))
    cuts = (
        eligible.with_columns(pl.col("u").rank("ordinal").over("vehicle_id").alias("r"))
        .filter(pl.col("r") <= cutoffs_per_vehicle)
        .drop("u", "r")
        .rename({"time_step": "t_cut"})
    )
    return attach_targets(cuts, tte, bins).sort("vehicle_id", "cut_idx")


def attach_targets(cuts: pl.DataFrame, tte: pl.DataFrame, bins: list[int]) -> pl.DataFrame:
    out = cuts.join(tte, on="vehicle_id", how="left").with_columns(
        (pl.col("length_of_study_time_step") - pl.col("t_cut")).alias("duration"),
        pl.col("in_study_repair").alias("event"),
    )
    cls = time_to_class(out["duration"].to_numpy(), bins)
    known = (out["event"].to_numpy() == 1) | (out["duration"].to_numpy() > bins[-1])
    return out.with_columns(
        pl.Series("class_label", np.where(known, cls, -1)).cast(pl.Int8).replace(-1, None)
    ).drop("length_of_study_time_step", "in_study_repair")


def last_readout_cutoffs(readouts: pl.DataFrame) -> pl.DataFrame:
    """Validation/test : la coupure est le dernier relevé disponible de chaque camion."""
    return (
        readouts.select("vehicle_id", "time_step")
        .with_columns(pl.int_range(pl.len()).over("vehicle_id").alias("cut_idx"))
        .group_by("vehicle_id")
        .agg(pl.col("cut_idx").last(), pl.col("time_step").last().alias("t_cut"))
        .sort("vehicle_id")
    )


def assign_folds(cuts: pl.DataFrame, n_folds: int, seed: int) -> pl.DataFrame:
    """Plis de validation croisée groupés par camion et stratifiés sur la réparation.

    Toutes les coupures d'un même camion tombent dans le même pli : sinon le modèle verrait
    en validation un camion déjà vu à l'entraînement.
    """
    rng = np.random.default_rng(seed)
    vehicles = cuts.group_by("vehicle_id").agg(pl.col("event").max()).sort("vehicle_id")
    fold = np.empty(vehicles.height, dtype=np.int8)
    for ev in (0, 1):
        idx = np.flatnonzero(vehicles["event"].to_numpy() == ev)
        fold[rng.permutation(idx)] = np.arange(idx.size) % n_folds
    return cuts.join(vehicles.select("vehicle_id", pl.Series("fold", fold)), on="vehicle_id")
