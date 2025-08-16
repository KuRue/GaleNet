"""Minimal training loop for simple linear models."""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Tuple

import numpy as np

from .losses import mse_loss

ArrayPair = Tuple[np.ndarray, np.ndarray]


class Trainer:
    """Tiny trainer operating on numpy arrays.

    The trainer assumes the ``model`` object exposes ``w`` and ``b`` attributes
    (``numpy.ndarray``) and provides an ``infer`` method returning predictions for
    a single input vector. This matches the :class:`~galenet.models.graphcast.GraphCastModel`
    API and is also compatible with similar Pangu-style models.
    """

    def __init__(
        self,
        model: any,
        dataset: Iterable[ArrayPair],
        loss_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]] = mse_loss,
        learning_rate: float = 1e-3,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.learning_rate = float(learning_rate)

    # ------------------------------------------------------------------
    def train(self, epochs: int = 1) -> Iterator[float]:
        """Yield average loss for each epoch."""

        for _ in range(epochs):
            total = 0.0
            count = 0
            for x, y in self.dataset:
                pred = self.model.infer(x)
                loss, grad = self.loss_fn(pred, y)
                # Gradient descent update for linear layer
                self.model.w -= self.learning_rate * np.outer(x, grad)
                self.model.b -= self.learning_rate * grad
                total += loss
                count += 1
            yield total / max(count, 1)
