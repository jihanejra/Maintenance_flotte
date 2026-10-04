"""Tests du modèle récurrent : cibles de survie discrètes, fenêtres sans fuite, apprentissage."""

import numpy as np
import pytest

from cockpit.survival.calibration import HORIZONS
from cockpit.survival.deep import discrete_targets
from cockpit.survival.sequences import SequenceStore


def test_discrete_targets():
    d = np.array([0.5, 7.0, 7.0, 100.0, 100.0])
    e = np.array([1, 1, 0, 1, 0])
    surv, ev = discrete_targets(d, e)
    j7 = HORIZONS.index(8)                      # 7 tombe dans la tranche (6, 8]
    assert ev[0, 0] == 1 and surv[0].sum() == 0
    assert ev[1, j7] == 1 and surv[1].sum() == j7
    assert ev[2].sum() == 0 and surv[2].sum() == HORIZONS.index(6) + 1
    assert ev[3].sum() == 0 and surv[3].sum() == len(HORIZONS)   # réparé après 52 : censuré
    assert surv[4].sum() == len(HORIZONS)


def _toy_store():
    X = np.arange(30, dtype=np.float16).reshape(10, 3)          # camion 1 : 4 lignes, 2 : 6
    return SequenceStore(X, np.array([1, 2]), np.array([0, 4]), ["a", "b", "c"],
                         mu=np.zeros(3), sd=np.ones(3))


def test_window_never_uses_future_rows():
    s = _toy_store()
    rows, valid = s.rows_for(np.array([1, 2, 2]), np.array([2, 0, 5]), 4)
    assert rows[valid].max() <= 9
    for (vid, cut), r, v in zip([(1, 2), (2, 0), (2, 5)], rows, valid, strict=True):
        start = {1: 0, 2: 4}[vid]
        assert set(r[v]) == set(range(max(start, start + cut - 3), start + cut + 1))
    w = s.window(np.array([2]), np.array([0]), 4)
    assert w.shape == (1, 4, 4) and w[0, :3].sum() == 0 and w[0, 3, -1] == 1


def test_gru_learns_on_toy_data():
    pytest.importorskip("torch")
    from cockpit.survival.deep import GRUSurvival

    rng = np.random.default_rng(0)
    n_veh, steps = 300, 8
    risk = rng.random(n_veh)
    X = np.repeat(risk, steps)[:, None].astype(np.float16)
    store = SequenceStore(X, np.arange(n_veh), np.arange(n_veh) * steps, ["r"],
                          mu=np.zeros(1), sd=np.ones(1))
    vid, cut = np.arange(n_veh), np.full(n_veh, steps - 1)
    d = np.where(risk > 0.5, rng.uniform(1, 10, n_veh), 60.0)
    e = (risk > 0.5).astype(int)
    m = GRUSurvival(epochs=15, batch=64, patience=15, lr=1e-2).fit(
        store, vid, cut, np.zeros((n_veh, 1), np.float32), d, e)
    S = m.survival(store, vid, cut, np.zeros((n_veh, 1), np.float32))
    assert (np.diff(S, axis=1) <= 1e-9).all()
    assert S[risk > 0.5, -1].mean() < S[risk <= 0.5, -1].mean() - 0.3
