"""Construit les tables de modélisation `data/processed/features_{split}.parquet`.

Une ligne = un camion à un instant de coupure. Colonnes :
identifiants (`vehicle_id`, `cut_idx`, `t_cut`), cibles (`duration`, `event`, `class_label`),
pli de validation croisée (`fold`, entraînement seulement), spécifications et variables.
Usage : `python -m cockpit.features.pipeline`
"""

from __future__ import annotations

import logging

import polars as pl

from cockpit.config import load_config, path
from cockpit.features.build import build_features
from cockpit.features.sampling import (
    age_density_weights,
    assign_folds,
    draw_cutoffs,
    last_readout_cutoffs,
)
from cockpit.ingest.load import (
    SPLITS,
    label_bins,
    load_labels,
    load_readouts,
    load_specs,
    load_tte,
    variable_groups,
)

log = logging.getLogger(__name__)
N_FOLDS = 5


def column_batches(columns: list[str]) -> list[list[str]]:
    """Paquets de colonnes traités séparément : les compteurs, puis un histogramme à la fois."""
    counters, histograms = variable_groups(columns)
    return [counters, *histograms.values()]


def make_cuts(split: str, keys: pl.DataFrame) -> pl.DataFrame:
    cfg = load_config("data")
    if split == "train":
        s = cfg["sampling"]
        weights = None
        if s["strategy"] == "match_validation_age":
            val = last_readout_cutoffs(load_readouts("validation", ["vehicle_id", "time_step"]))
            weights = age_density_weights(keys["time_step"].to_numpy(), val["t_cut"].to_numpy(),
                                          s["age_bin_width"])
        cuts = draw_cutoffs(keys, load_tte(), label_bins(), s["cutoffs_per_vehicle"],
                            s["min_readouts"], s["seed"], age_weights=weights)
        return assign_folds(cuts, N_FOLDS, s["seed"])
    return last_readout_cutoffs(keys).join(load_labels(split), on="vehicle_id", how="left")


def build_split(split: str) -> pl.DataFrame:
    k = load_config("data")["features"]["slope_window"]
    keys = load_readouts(split, columns=["vehicle_id", "time_step"])
    all_cols = pl.read_parquet_schema(path("interim") / f"readouts_{split}.parquet")
    cuts = make_cuts(split, keys)

    parts = [build_features(keys, cuts, k)]  # méta-variables (âge, nombre de relevés...)
    for batch in column_batches(list(all_cols)):
        ro = load_readouts(split, columns=["vehicle_id", "time_step", *batch])
        parts.append(build_features(ro, cuts, k).drop(["age", "n_readouts", "mean_gap",
                                                        "last_gap"]))
        del ro

    specs = load_specs(split)  # catégories gardées en texte : encodage décidé en phase 2
    assert all(p.height == cuts.height for p in parts)
    table = cuts.hstack([s for p in parts for s in p.get_columns()])
    table = table.join(specs, on="vehicle_id", how="left")
    out = path("processed") / f"features_{split}.parquet"
    table.write_parquet(out)
    log.info("%s : %d lignes x %d colonnes -> %s", split, table.height, table.width, out.name)
    return table


def main() -> None:
    for split in SPLITS:
        build_split(split)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
