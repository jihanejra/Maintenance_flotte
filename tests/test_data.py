"""Tests sur les vraies tables produites par la phase 1 (ignorés si les données sont absentes)."""

import polars as pl
import pytest

from cockpit.config import PROJECT_ROOT

PROCESSED = PROJECT_ROOT / "data" / "processed"
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (PROCESSED / "features_test.parquet").exists(),
                       reason="lancer d'abord la chaîne de la phase 1"),
]


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return pl.read_parquet(PROCESSED / "features_train.parquet")


def test_one_row_per_vehicle_in_eval_splits():
    for split in ("validation", "test"):
        f = pl.read_parquet(PROCESSED / f"features_{split}.parquet")
        assert f["vehicle_id"].is_unique().all()
        assert f["class_label"].null_count() == 0


def test_train_targets_consistent(train):
    assert (train["duration"] >= 0).all()
    known = train.filter(pl.col("class_label").is_not_null())
    # Classe > 0 seulement pour les camions effectivement réparés
    assert (known.filter(pl.col("class_label") > 0)["event"] == 1).all()
    # Classe inconnue seulement pour des censurés à moins de 48 pas de la fin
    unknown = train.filter(pl.col("class_label").is_null())
    assert (unknown["event"] == 0).all() and (unknown["duration"] <= 48).all()


def test_folds_grouped(train):
    assert (train.group_by("vehicle_id").agg(pl.col("fold").n_unique())["fold"] == 1).all()


def test_same_feature_columns_across_splits(train):
    cols = set(train.columns) - {"duration", "event", "fold"}
    for split in ("validation", "test"):
        assert set(pl.read_parquet_schema(PROCESSED / f"features_{split}.parquet")) == cols
