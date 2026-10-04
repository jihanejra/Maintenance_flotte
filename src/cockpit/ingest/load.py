"""Conversion des CSV bruts en Parquet typé, puis chargement.

Les relevés d'entraînement font 1,2 Go en CSV (1,1 million de lignes x 107 colonnes). On les
convertit en streaming avec polars, en float32, pour tenir dans une machine à 8 Go de RAM.
Usage : `python -m cockpit.ingest.load`
"""

from __future__ import annotations

import logging
import re

import polars as pl

from cockpit.config import load_config, path

log = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
ID_COLS = ("vehicle_id", "time_step")
HIST_RE = re.compile(r"^(\d+)_(\d+)$")


def readout_schema(columns: list[str]) -> dict[str, pl.DataType]:
    """Identifiant en entier, temps et mesures en float32."""
    return {c: (pl.Int32 if c == "vehicle_id" else pl.Float32) for c in columns}


def convert_split(split: str) -> None:
    raw, interim = path("raw"), path("interim")

    src = raw / f"{split}_operational_readouts.csv"
    header = pl.read_csv(src, n_rows=0).columns
    (
        pl.scan_csv(src, schema_overrides=readout_schema(header))
        .sort(list(ID_COLS))
        .sink_parquet(interim / f"readouts_{split}.parquet")
    )

    specs = pl.read_csv(raw / f"{split}_specifications.csv", infer_schema_length=0)
    specs = specs.with_columns(pl.col("vehicle_id").cast(pl.Int32))
    specs.write_parquet(interim / f"specs_{split}.parquet")

    if split == "train":
        tte = pl.read_csv(raw / "train_tte.csv").with_columns(
            pl.col("vehicle_id").cast(pl.Int32),
            pl.col("length_of_study_time_step").cast(pl.Float32),
            pl.col("in_study_repair").cast(pl.Int8),
        )
        tte.write_parquet(interim / "tte_train.parquet")
    else:
        labels = pl.read_csv(raw / f"{split}_labels.csv").with_columns(
            pl.col("vehicle_id").cast(pl.Int32), pl.col("class_label").cast(pl.Int8)
        )
        labels.write_parquet(interim / f"labels_{split}.parquet")
    log.info("converti : %s", split)


def convert_all() -> None:
    for split in SPLITS:
        convert_split(split)


# --- Chargement -------------------------------------------------------------------------------

def load_readouts(split: str, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path("interim") / f"readouts_{split}.parquet", columns=columns)


def load_specs(split: str) -> pl.DataFrame:
    return pl.read_parquet(path("interim") / f"specs_{split}.parquet")


def load_tte() -> pl.DataFrame:
    return pl.read_parquet(path("interim") / "tte_train.parquet")


def load_labels(split: str) -> pl.DataFrame:
    return pl.read_parquet(path("interim") / f"labels_{split}.parquet")


def variable_groups(columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Sépare les compteurs (une seule colonne `xxx_0`) des histogrammes (`xxx_0..xxx_k`)."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        m = HIST_RE.match(c)
        if m:
            groups.setdefault(m.group(1), []).append(c)
    for cols in groups.values():
        cols.sort(key=lambda c: int(c.split("_")[1]))
    counters = [cols[0] for cols in groups.values() if len(cols) == 1]
    histograms = {v: cols for v, cols in groups.items() if len(cols) > 1}
    return counters, histograms


def label_bins() -> list[int]:
    return load_config("data")["label_bins"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert_all()
