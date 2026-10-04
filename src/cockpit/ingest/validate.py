"""Contrôles qualité automatiques (pandera) sur les tables converties.

Deux niveaux :
- `validate_*` : règles bloquantes (types, unicité, plages). Une violation lève une erreur.
- `quality_report` : constats chiffrés non bloquants (valeurs manquantes, compteurs qui
  décroissent...), repris dans le rapport d'analyse exploratoire.
Usage : `python -m cockpit.ingest.validate`
"""

from __future__ import annotations

import json
import logging

import pandera.polars as pa
import polars as pl

from cockpit.config import path
from cockpit.ingest.load import (
    SPLITS,
    load_labels,
    load_readouts,
    load_specs,
    load_tte,
    variable_groups,
)

log = logging.getLogger(__name__)


def readout_schema(columns: list[str]) -> pa.DataFrameSchema:
    measures = {
        c: pa.Column(pl.Float32, pa.Check.ge(0), nullable=True)
        for c in columns
        if c not in ("vehicle_id", "time_step")
    }
    return pa.DataFrameSchema(
        {
            "vehicle_id": pa.Column(pl.Int32, nullable=False),
            "time_step": pa.Column(pl.Float32, pa.Check.ge(0), nullable=False),
            **measures,
        },
        unique=["vehicle_id", "time_step"],
    )


TTE_SCHEMA = pa.DataFrameSchema(
    {
        "vehicle_id": pa.Column(pl.Int32, unique=True),
        "length_of_study_time_step": pa.Column(pl.Float32, pa.Check.gt(0)),
        "in_study_repair": pa.Column(pl.Int8, pa.Check.isin([0, 1])),
    }
)

LABEL_SCHEMA = pa.DataFrameSchema(
    {
        "vehicle_id": pa.Column(pl.Int32, unique=True),
        "class_label": pa.Column(pl.Int8, pa.Check.isin([0, 1, 2, 3, 4])),
    }
)


def validate_split(split: str) -> None:
    ro = load_readouts(split)
    readout_schema(ro.columns).validate(ro)

    specs = load_specs(split)
    ids = set(ro["vehicle_id"].unique().to_list())
    if set(specs["vehicle_id"].to_list()) != ids:
        raise ValueError(f"{split} : spécifications et relevés ne couvrent pas les mêmes camions")

    if split == "train":
        tte = TTE_SCHEMA.validate(load_tte())
        if set(tte["vehicle_id"].to_list()) != ids:
            raise ValueError("train : time-to-event et relevés ne couvrent pas les mêmes camions")
        last = ro.group_by("vehicle_id").agg(pl.col("time_step").max().alias("t_last"))
        bad = last.join(tte, on="vehicle_id").filter(
            pl.col("t_last") > pl.col("length_of_study_time_step")
        )
        if bad.height:
            raise ValueError(f"train : {bad.height} camions ont des relevés après la fin d'étude")
    else:
        labels = LABEL_SCHEMA.validate(load_labels(split))
        if set(labels["vehicle_id"].to_list()) != ids:
            raise ValueError(f"{split} : labels et relevés ne couvrent pas les mêmes camions")
    log.info("validé : %s", split)


def quality_report(split: str) -> dict:
    """Constats non bloquants : taille, valeurs manquantes, monotonie des compteurs."""
    ro = load_readouts(split)
    counters, histograms = variable_groups(ro.columns)
    measures = [c for c in ro.columns if c not in ("vehicle_id", "time_step")]

    missing = ro.select(pl.col(measures).null_count()).row(0, named=True)
    rows_with_missing = ro.select(pl.any_horizontal(pl.col(measures).is_null()).sum()).item()

    # Un compteur cumulatif ne devrait jamais décroître d'un relevé au suivant.
    diffs = ro.select(
        "vehicle_id",
        *[pl.col(c).diff().over("vehicle_id").alias(c) for c in counters],
    )
    decreasing = diffs.select([(pl.col(c) < 0).sum().alias(c) for c in counters]).row(
        0, named=True
    )
    # Contrôle classe par classe : sum_horizontal compterait une ligne manquante comme un total nul
    # et ferait apparaître de fausses décroissances.
    hist_decreasing = {
        v: int(ro.select([(pl.col(c).diff().over("vehicle_id") < 0).sum() for c in cols])
               .sum_horizontal().item())
        for v, cols in histograms.items()
    }
    # Décroissances « à travers un trou » : visibles seulement après propagation vers l'avant.
    decreases_across_gaps = {
        v: int(
            ro.select("vehicle_id", pl.col(cols).forward_fill().over("vehicle_id"))
            .select([(pl.col(c).diff().over("vehicle_id") < 0).sum() for c in cols])
            .sum_horizontal().item()
        )
        for v, cols in histograms.items()
    }
    hist_missing_rows = {
        v: int(ro.select(pl.all_horizontal(pl.col(cols).is_null()).sum()).item())
        for v, cols in histograms.items()
    }
    n_steps = ro.group_by("vehicle_id").len()

    return {
        "split": split,
        "rows": ro.height,
        "vehicles": ro["vehicle_id"].n_unique(),
        "columns_measures": len(measures),
        "counters": counters,
        "histograms": {v: len(c) for v, c in histograms.items()},
        "rows_with_missing": int(rows_with_missing),
        "missing_by_column_top": dict(sorted(missing.items(), key=lambda kv: -kv[1])[:15]),
        "counter_decreases": decreasing,
        "histogram_bin_decreases": hist_decreasing,
        "histogram_missing_rows": hist_missing_rows,
        "histogram_decreases_across_gaps": decreases_across_gaps,
        "vehicles_with_missing": int(
            ro.filter(pl.any_horizontal(pl.col(measures).is_null()))["vehicle_id"].n_unique()
        ),
        "readouts_per_vehicle": n_steps["len"].describe().to_dicts(),
    }


def main() -> None:
    reports = {}
    for split in SPLITS:
        validate_split(split)
        reports[split] = quality_report(split)
    out = path("processed") / "quality_report.json"
    out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    log.info("rapport qualité : %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
