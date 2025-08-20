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
    loss = float(np.mean(diff**2))
    grad = (2.0 / diff.size) * diff
    return loss, grad


def mass_conservation_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Squared difference between total predicted and target mass.

    Parameters
    ----------
    pred, target:
        Arrays of identical shape representing density or mass fields.
    """

    diff = float(np.sum(pred) - np.sum(target))
    loss = diff**2
    grad = np.full_like(pred, 2.0 * diff)
    return loss, grad


def momentum_conservation_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Squared error between total momentum vectors.

    Parameters
    ----------
    pred, target:
        Arrays with last dimension 2 representing velocity vectors ``(u, v)``.
    """

    pred_tot = np.sum(pred, axis=tuple(range(pred.ndim - 1)))
    target_tot = np.sum(target, axis=tuple(range(target.ndim - 1)))
    diff = pred_tot - target_tot
    loss = float(np.sum(diff**2))
    grad = np.zeros_like(pred)
    grad[..., 0] = 2.0 * diff[0]
    grad[..., 1] = 2.0 * diff[1]
    return loss, grad


def kinetic_energy_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error between kinetic energy of ``pred`` and ``target``.

    Parameters
    ----------
    pred, target:
        Arrays with last dimension 2 representing velocity vectors ``(u, v)``.
    """

    pred_ke = 0.5 * np.sum(pred**2, axis=-1)
    target_ke = 0.5 * np.sum(target**2, axis=-1)
    diff = pred_ke - target_ke
    loss = float(np.mean(diff**2))
    grad_ke = (2.0 / diff.size) * diff
    grad = grad_ke[..., None] * pred
    return loss, grad
