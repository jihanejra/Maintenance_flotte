import numpy as np
import polars as pl
import pytest

from cockpit.features.build import build_features
from cockpit.features.sampling import (
    assign_folds,
    draw_cutoffs,
    last_readout_cutoffs,
    time_to_class,
)

BINS = [6, 12, 24, 48]


def test_time_to_class_boundaries():
    rem = np.array([0, 6, 6.5, 12, 13, 24, 25, 48, 49, 500])
    assert time_to_class(rem, BINS).tolist() == [4, 4, 3, 3, 2, 2, 1, 1, 0, 0]


def test_no_leakage(toy_readouts):
    """Les variables d'une coupure sont identiques si on supprime tous les relevés postérieurs."""
    cuts = pl.DataFrame({"vehicle_id": [1, 3, 3], "cut_idx": [4, 2, 9]}).with_columns(
        pl.col("vehicle_id").cast(pl.Int32)
    )
    full = build_features(toy_readouts, cuts, k=3)
    for row, (vid, i) in enumerate(cuts.iter_rows()):
        past = toy_readouts.filter(
            (pl.col("vehicle_id") == vid) & (pl.int_range(pl.len()).over("vehicle_id") <= i)
        )
        trunc = build_features(past, cuts[row], k=3)
        np.testing.assert_allclose(full[row].to_numpy(), trunc.to_numpy(), equal_nan=True)


def test_rates_and_shares(toy_readouts):
    cuts = last_readout_cutoffs(toy_readouts)
    f = build_features(toy_readouts, cuts, k=3)
    shares = f.select(pl.col("^hist_167_b\\d+_share$")).to_numpy()
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, rtol=1e-5)
    assert (f["cnt_171_rate_all"].drop_nans().drop_nulls() >= 0).all()
    assert f["n_readouts"].to_list() == [8, 3, 12]


def test_missing_value_forward_filled(toy_readouts):
    cuts = pl.DataFrame({"vehicle_id": [3], "cut_idx": [4]}).with_columns(
        pl.col("vehicle_id").cast(pl.Int32)
    )
    f = build_features(toy_readouts, cuts, k=3)
    assert not np.isnan(f["cnt_171_last"][0])


def test_cutoffs_targets(toy_readouts, toy_tte):
    cuts = draw_cutoffs(toy_readouts, toy_tte, BINS, cutoffs_per_vehicle=2, min_readouts=2,
                        seed=1)
    assert cuts.group_by("vehicle_id").len()["len"].max() <= 2
    assert (cuts["cut_idx"] >= 1).all()
    assert (cuts["duration"] >= 0).all()
    # Camion 2 censuré : classe inconnue si la fin d'étude est à moins de 48 pas de temps
    v2 = cuts.filter(pl.col("vehicle_id") == 2)
    assert all((d > 48) == (c is not None) for d, c in v2.select("duration", "class_label")
               .iter_rows())


def test_folds_grouped_by_vehicle(toy_readouts, toy_tte):
    cuts = draw_cutoffs(toy_readouts, toy_tte, BINS, 3, 2, seed=1)
    folded = assign_folds(cuts, n_folds=2, seed=0)
    assert (folded.group_by("vehicle_id").agg(pl.col("fold").n_unique())["fold"] == 1).all()


@pytest.mark.parametrize("k", [1, 5, 50])
def test_window_larger_than_history(toy_readouts, k):
    cuts = last_readout_cutoffs(toy_readouts)
    f = build_features(toy_readouts, cuts, k=k)
    assert f.height == 3
