"""Integration test against a real Kafka broker (not mocked). Skips cleanly
if no broker is reachable on localhost:9092 — mirrors the pattern used for
the Gemini-API-dependent tests in the repo-qa-agent project: a real
infrastructure test that degrades gracefully rather than faking a pass."""
import json
import socket

import pytest
from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP = "localhost:9092"
TOPIC = "security-events-test"


def _kafka_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 9092), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _kafka_reachable(), reason="No Kafka broker reachable on localhost:9092")


def test_produce_and_consume_round_trip():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    sent_payload = {"Destination Port": 80, "Flow Duration": 123, "Label": "BENIGN"}
    producer.send(TOPIC, sent_payload)
    producer.flush()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=10000,
        group_id="test-round-trip",
    )
    received = [msg.value for msg in consumer]
    consumer.close()

    assert sent_payload in received
