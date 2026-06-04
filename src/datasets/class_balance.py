from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    class_weights = make_class_weights(labels, num_classes).numpy()
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_class_aware_sampler(
    labels: list[int],
    num_classes: int,
    hard_label_ids: set[int],
    hard_multiplier: float,
) -> WeightedRandomSampler:
    class_weights = make_class_weights(labels, num_classes).numpy()
    multiplier = max(1.0, float(hard_multiplier))
    sample_weights = [
        class_weights[label] * (multiplier if label in hard_label_ids else 1.0)
        for label in labels
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
