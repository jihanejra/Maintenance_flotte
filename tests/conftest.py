import numpy as np
import polars as pl
import pytest


@pytest.fixture
def toy_readouts() -> pl.DataFrame:
    """3 camions, relevés irréguliers, 2 compteurs cumulatifs et 1 histogramme à 4 classes."""
    rng = np.random.default_rng(0)
    rows = []
    for vid, n in [(1, 8), (2, 3), (3, 12)]:
        t = np.cumsum(rng.integers(1, 6, n)).astype(np.float32)
        c1 = np.cumsum(rng.random(n) * 10)
        c2 = np.cumsum(rng.random(n) * 3)
        h = np.cumsum(rng.integers(0, 20, (n, 4)), axis=0)
        for j in range(n):
            rows.append({"vehicle_id": vid, "time_step": t[j], "171_0": c1[j], "666_0": c2[j],
                         **{f"167_{b}": float(h[j, b]) for b in range(4)}})
    df = pl.DataFrame(rows).with_columns(
        pl.col("vehicle_id").cast(pl.Int32), pl.exclude("vehicle_id").cast(pl.Float32)
    )
    # Une valeur manquante pour tester la propagation vers l'avant
    return df.with_columns(
        pl.when((pl.col("vehicle_id") == 3) & (pl.int_range(pl.len()) == 15))
        .then(None).otherwise(pl.col("171_0")).alias("171_0")
    )


@pytest.fixture
def toy_tte() -> pl.DataFrame:
    return pl.DataFrame(
        {"vehicle_id": [1, 2, 3], "length_of_study_time_step": [40.0, 100.0, 60.0],
         "in_study_repair": [1, 0, 1]}
    ).with_columns(pl.col("vehicle_id").cast(pl.Int32),
                   pl.col("length_of_study_time_step").cast(pl.Float32),
                   pl.col("in_study_repair").cast(pl.Int8))
