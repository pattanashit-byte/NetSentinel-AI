"""Three detection approaches, compared head-to-head in eval/run_eval.py:

1. StatisticalDetector   — per-feature z-score threshold (the naive baseline).
2. IsolationForestDetector — unsupervised, trained on windowed aggregate
   features from benign traffic only.
3. SequenceAutoencoder   — supervised-free, PyTorch LSTM autoencoder trained
   on benign per-flow sequences; flags high reconstruction error.
4. SupervisedBaseline    — Random Forest trained WITH attack labels, included
   only as an upper-bound reference (a real deployment can't assume labeled
   attacks in advance, so this one is not "the model" — it's the ceiling
   the unsupervised approaches are measured against).
"""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class StatisticalDetector:
    """Flags a window if any feature is more than `z_thresh` std devs from
    the training mean — the classic, weak baseline every ML approach here
    is compared against."""

    def __init__(self, z_thresh: float = 3.0):
        self.z_thresh = z_thresh
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return (z.max(axis=1) > self.z_thresh).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn: -1 = anomaly, 1 = normal -> remap to 1 = anomaly, 0 = normal
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)  # higher = more anomalous


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, latent_size: int = 8):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x):
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        z = self.to_latent(h_n[-1])
        h_dec = self.from_latent(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder(h_dec)
        return out


class SequenceAutoencoderDetector:
    """PyTorch LSTM autoencoder over per-flow feature sequences.

    Trained only on benign sequences; a window's anomaly score is its mean
    per-timestep reconstruction MSE. Threshold is set from the training
    (benign) reconstruction-error distribution, not from the labeled data.
    """

    def __init__(self, n_features: int, hidden_size: int = 32, epochs: int = 8, lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.model = LSTMAutoencoder(n_features, hidden_size).to(device)
        self.epochs = epochs
        self.lr = lr
        self.threshold_ = None
        self.feature_mean_ = None
        self.feature_std_ = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean_) / self.feature_std_

    def fit(self, X: np.ndarray, batch_size: int = 64):
        self.feature_mean_ = X.mean(axis=(0, 1), keepdims=True)
        self.feature_std_ = X.std(axis=(0, 1), keepdims=True) + 1e-9
        Xn = self._normalize(X)
        tensor = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        n = tensor.size(0)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                batch = tensor[perm[i:i + batch_size]]
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)

        train_errors = self._reconstruction_error(tensor)
        self.threshold_ = float(np.percentile(train_errors, 99))
        return self

    def _reconstruction_error(self, tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor)
            err = torch.mean((recon - tensor) ** 2, dim=(1, 2))
        return err.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        tensor = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        return self._reconstruction_error(tensor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


class SupervisedBaseline:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def timed_predict(detector, X: np.ndarray):
    """Returns (predictions, mean_latency_ms_per_window)."""
    start = time.perf_counter()
    preds = detector.predict(X)
    elapsed = time.perf_counter() - start
    per_window_ms = (elapsed / max(len(X), 1)) * 1000
    return preds, per_window_ms
