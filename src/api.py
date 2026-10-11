"""FastAPI service exposing scored windows and alerts persisted by the
Kafka consumer, plus a health check and a summary stats endpoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.db import Alert, get_session, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Security Event Anomaly Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/alerts")
def list_alerts(limit: int = 50, anomalies_only: bool = False):
    session = get_session()
    try:
        query = session.query(Alert).order_by(Alert.id.desc())
        if anomalies_only:
            query = query.filter(Alert.is_anomaly == 1)
        rows = query.limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "detector": r.detector,
                "window_index": r.window_index,
                "anomaly_score": r.anomaly_score,
                "is_anomaly": bool(r.is_anomaly),
                "dest_ports_touched": r.dest_ports_touched,
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/stats")
def stats():
    session = get_session()
    try:
        total = session.query(Alert).count()
        anomalies = session.query(Alert).filter(Alert.is_anomaly == 1).count()
        return {
            "total_windows_scored": total,
            "anomalies_flagged": anomalies,
            "anomaly_rate": (anomalies / total) if total else 0.0,
        }
    finally:
        session.close()
