import os

os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from fastapi.testclient import TestClient

from src.api import app
from src.db import Alert, engine, get_session, init_db


def setup_module(module):
    init_db()
    session = get_session()
    session.query(Alert).delete()
    session.add(Alert(detector="test", window_index=0, anomaly_score=0.1, is_anomaly=0, dest_ports_touched=1))
    session.add(Alert(detector="test", window_index=1, anomaly_score=9.9, is_anomaly=1, dest_ports_touched=15))
    session.commit()
    session.close()


def teardown_module(module):
    engine.dispose()
    if os.path.exists("./test_api.db"):
        os.remove("./test_api.db")


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_alerts():
    resp = client.get("/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2


def test_list_alerts_anomalies_only():
    resp = client.get("/alerts", params={"anomalies_only": True})
    body = resp.json()
    assert len(body) == 1
    assert body[0]["is_anomaly"] is True


def test_stats():
    resp = client.get("/stats")
    body = resp.json()
    assert body["total_windows_scored"] == 2
    assert body["anomalies_flagged"] == 1
    assert body["anomaly_rate"] == 0.5
