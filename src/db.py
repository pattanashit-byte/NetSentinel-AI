"""SQLAlchemy models + session setup.

DATABASE_URL defaults to a local SQLite file for tests/local dev, and is
overridden to a real Postgres DSN in docker-compose / CI — same env-var
pattern used across the other portfolio projects, so tests can run without
a live Postgres instance while CI still verifies against the real thing.
"""
import os

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./local_dev.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    detector = Column(String, nullable=False)
    window_index = Column(Integer, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Integer, nullable=False)  # 0/1
    dest_ports_touched = Column(Integer, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
