import numpy as np

from src.detectors import (
    IsolationForestDetector,
    LSTMAutoencoder,
    SequenceAutoencoderDetector,
    StatisticalDetector,
    SupervisedBaseline,
    timed_predict,
)


def _benign_and_attack(n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    benign = rng.normal(loc=0.0, scale=1.0, size=(500, n_features))
    attack = rng.normal(loc=8.0, scale=1.0, size=(50, n_features))  # clearly shifted
    return benign, attack


def test_statistical_detector_flags_shifted_points():
    benign, attack = _benign_and_attack()
    detector = StatisticalDetector(z_thresh=3.0).fit(benign)
    assert detector.predict(attack).mean() > 0.8
    assert detector.predict(benign).mean() < 0.1


def test_isolation_forest_flags_shifted_points():
    benign, attack = _benign_and_attack()
    detector = IsolationForestDetector(contamination=0.02).fit(benign)
    assert detector.predict(attack).mean() > 0.8


def test_supervised_baseline_learns_the_boundary():
    benign, attack = _benign_and_attack()
    X = np.vstack([benign, attack])
    y = np.array([0] * len(benign) + [1] * len(attack))
    detector = SupervisedBaseline().fit(X, y)
    preds = detector.predict(X)
    assert (preds == y).mean() > 0.95


def test_lstm_autoencoder_reconstructs_benign_sequences_better_than_shifted_ones():
    rng = np.random.default_rng(0)
    n_features = 3
    benign_seqs = rng.normal(0, 1, size=(200, 10, n_features)).astype(np.float32)
    attack_seqs = rng.normal(6, 1, size=(20, 10, n_features)).astype(np.float32)

    detector = SequenceAutoencoderDetector(n_features=n_features, hidden_size=8, epochs=5)
    detector.fit(benign_seqs)

    benign_scores = detector.score(benign_seqs)
    attack_scores = detector.score(attack_seqs)
    assert attack_scores.mean() > benign_scores.mean()
    assert detector.predict(attack_seqs).mean() > detector.predict(benign_seqs).mean()


def test_timed_predict_returns_latency():
    benign, attack = _benign_and_attack()
    detector = StatisticalDetector().fit(benign)
    preds, latency_ms = timed_predict(detector, attack)
    assert len(preds) == len(attack)
    assert latency_ms >= 0
