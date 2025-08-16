"""Loss functions for model training."""

from __future__ import annotations

import numpy as np


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean squared error and gradient w.r.t ``pred``.

    Parameters
    ----------
    pred, target:
        Vectors of identical shape representing model predictions and
        ground truth values.
    """

    diff = pred - target
    loss = float(np.mean(diff ** 2))
    grad = (2.0 / diff.size) * diff
    return loss, grad
