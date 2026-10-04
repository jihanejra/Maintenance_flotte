"""Tests des briques de décision : coût minimal, calibration, planning, pièces, simulation."""

import itertools

import numpy as np
import polars as pl
import pytest

from cockpit.decision.cost_rule import challenge_cost, cost_matrix, decide, prior_shift
from cockpit.economics import unit_costs
from cockpit.operational.workshop_milp import WorkshopParams, plan_week
from cockpit.strategic.fleet_simulator import Scenario, WeibullGamma, simulate
from cockpit.survival.calibration import SurvivalCalibrator
from cockpit.survival.models import survival_to_classes
from cockpit.tactical.planning import poisson_binomial, spare_parts_plan


def test_cost_rule_alerts_on_small_failure_probability():
    # 3 % de panne imminente : la classe la plus probable est 0, mais l'alerte est moins chère
    p = np.array([[0.97, 0.0, 0.0, 0.0, 0.03]])
    assert p.argmax() == 0
    assert decide(p)[0] > 0


def test_cost_rule_is_bayes_optimal():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), 200)
    C = cost_matrix()
    best = decide(p)
    for j in range(5):
        assert ((p @ C)[np.arange(200), best] <= (p @ C)[:, j] + 1e-12).all()


def test_challenge_cost_matches_matrix():
    y_true, y_pred = np.array([0, 4, 2]), np.array([1, 0, 2])
    assert challenge_cost(y_true, y_pred) == 7 + 500 + 0


def test_prior_shift_identity_and_direction():
    p = np.array([[0.9, 0.025, 0.025, 0.025, 0.025]])
    prior = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    np.testing.assert_allclose(prior_shift(p, prior, prior), p)
    more4 = prior_shift(p, prior, np.array([0.9, 0.01, 0.01, 0.02, 0.06]))
    assert more4[0, 4] > p[0, 4] and np.isclose(more4.sum(), 1)


def test_survival_to_classes_sums_to_one():
    S = np.array([[0.99, 0.98, 0.95, 0.9], [1, 1, 1, 1]])
    p = survival_to_classes(S)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert p[1, 0] == 1 and np.isclose(p[0, 4], 0.01)


def test_calibrator_monotone_in_horizon():
    rng = np.random.default_rng(1)
    n = 3000
    d = rng.exponential(60, n)
    e = (rng.random(n) < 0.6).astype(int)
    raw = np.exp(-np.outer(rng.random(n) * 0.02, np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48,
                                                             52])))
    cal = SurvivalCalibrator().fit(raw, d, e)
    S = cal.transform(raw)
    assert (np.diff(S, axis=1) <= 1e-12).all()
    grid = cal.survival_at(S, range(1, 53))
    assert grid.shape == (n, 52) and (np.diff(grid, axis=1) <= 1e-12).all()


def test_poisson_binomial_matches_enumeration():
    p = np.array([0.1, 0.5, 0.3, 0.8])
    exact = np.zeros(5)
    for outcome in itertools.product([0, 1], repeat=4):
        exact[sum(outcome)] += np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome,
                                                                            strict=True)])
    np.testing.assert_allclose(poisson_binomial(p, kmax=4), exact, atol=1e-12)


@pytest.fixture
def toy_fleet():
    rng = np.random.default_rng(3)
    n = 400
    rate = rng.gamma(0.3, 0.02, n)
    return pl.DataFrame({"vehicle_id": np.arange(n), "t_cut": np.full(n, 100.0),
                         **{f"S_{h}": np.exp(-rate * h) for h in range(1, 53)}})


def test_workshop_plan_respects_constraints(toy_fleet):
    params = WorkshopParams.from_config(bays=2, slots_per_bay_per_day=1, parts_in_stock=7,
                                        truck_availability=0.6)
    plan = plan_week(toy_fleet, params)
    s = plan["schedule"]
    assert s.height <= 7                                       # pièces
    assert s["vehicle_id"].is_unique().all()                   # une fois par camion
    assert (s.group_by("jour").len()["len"] <= 2).all()        # créneaux par jour
    thr = unit_costs().threshold
    assert (s["proba_panne_fenetre"] > thr).all()              # uniquement rentables


def test_spare_parts_stock_covers_critical_ratio(toy_fleet):
    plan = spare_parts_plan(toy_fleet)
    cdf = np.cumsum(plan["pmf"])
    s = plan["optimal_stock"]
    assert cdf[s] >= plan["critical_ratio"] and (s == 0 or cdf[s - 1] < plan["critical_ratio"])


def test_simulator_oracle_beats_corrective():
    law = WeibullGamma(k=3.0, lam=300.0, theta=2.0)
    lengths = np.linspace(100, 400, 200)
    costs = unit_costs()
    corr = simulate(Scenario("corrective"), law, costs, 1000, 104, seed=0, lengths=lengths)
    orac = simulate(Scenario("oracle"), law, costs, 1000, 104, seed=0, lengths=lengths)
    assert orac["breakdowns"] < corr["breakdowns"]
    assert orac["cost"] < corr["cost"]


def test_weibull_gamma_residual_positive():
    law = WeibullGamma(k=2.5, lam=200.0, theta=1.0)
    rng = np.random.default_rng(0)
    r = law.residual(np.full(1000, 50.0), law.sample_frailty(1000, rng), rng)
    assert (r > 0).all()
