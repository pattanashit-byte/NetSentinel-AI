"""Distribution-shift monitoring: flag when live feature distributions
diverge from the training (reference) distribution, using the two-sample
Kolmogorov-Smirnov test per feature.

This does NOT detect individual anomalous events — it answers a different,
complementary question: "has normal behavior itself shifted enough that the
detector should probably be retrained?"
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    feature_pvalues: dict = field(default_factory=dict)
    drifted_features: list = field(default_factory=list)
    alpha: float = 0.01

    @property
    def drifted(self) -> bool:
        return len(self.drifted_features) > 0


class DriftMonitor:
    def __init__(self, reference: np.ndarray, feature_names: list, alpha: float = 0.01):
        self.reference = reference
        self.feature_names = feature_names
        self.alpha = alpha

    def check(self, current: np.ndarray) -> DriftReport:
        report = DriftReport(alpha=self.alpha)
        for i, name in enumerate(self.feature_names):
            _, p_value = ks_2samp(self.reference[:, i], current[:, i])
            report.feature_pvalues[name] = float(p_value)
            if p_value < self.alpha:
                report.drifted_features.append(name)
        return report
