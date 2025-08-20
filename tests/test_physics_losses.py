import sys
from pathlib import Path

import numpy as np
import pytest

# Import losses without triggering heavy package imports
sys.path.append(str(Path(__file__).parent.parent / "src/galenet/training"))

from losses import (  # type: ignore  # noqa: E402
    kinetic_energy_loss,
    mass_conservation_loss,
    momentum_conservation_loss,
)


def test_mass_conservation_loss():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    loss, grad = mass_conservation_loss(pred, target)
    assert loss == pytest.approx(64.0)
    assert np.all(grad == 16.0)


def test_momentum_conservation_loss():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros_like(pred)
    loss, grad = momentum_conservation_loss(pred, target)
    assert loss == pytest.approx(52.0)
    assert np.all(grad[..., 0] == 8.0)
    assert np.all(grad[..., 1] == 12.0)


def test_kinetic_energy_loss():
    pred = np.array([[2.0, 0.0], [0.0, 2.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    loss, grad = kinetic_energy_loss(pred, target)
    assert loss == pytest.approx(2.25)
    expected_grad = np.array([[3.0, 0.0], [0.0, 3.0]])
    assert np.allclose(grad, expected_grad)
