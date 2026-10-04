"""Tests de l'API sur la flotte calculée (ignorés si la phase 2 n'a pas tourné)."""

import pytest

from cockpit.config import PROJECT_ROOT

FLEET = PROJECT_ROOT / "data" / "processed" / "fleet_risk_test.parquet"
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not FLEET.exists(), reason="lancer d'abord la phase 2")]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from cockpit.api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["vehicles"] > 0


def test_risk_curve_is_decreasing(client):
    vid = client.get("/fleet/summary?top=1").json()["top_risk"][0]["vehicle_id"]
    r = client.get(f"/risk/{vid}").json()
    s = [r["survival"][str(h)] for h in range(1, 53)]
    assert all(a >= b for a, b in zip(s, s[1:], strict=False))
    assert abs(sum(r["class_probabilities"].values()) - 1) < 1e-6


def test_unknown_vehicle(client):
    assert client.get("/risk/999999999").status_code == 404


def test_workshop_plan_and_capacity(client):
    small = client.get("/workshop-plan?bays=1&parts_in_stock=3").json()
    assert small["n_scheduled"] <= 3


def test_parts_plan(client):
    r = client.get("/parts-plan?lead_time_weeks=4").json()
    assert r["horizon_weeks"] == 8 and r["stock_for_99pct"] >= r["optimal_stock"] - 1
