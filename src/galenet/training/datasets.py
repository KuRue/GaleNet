"""Dataset helpers for training models."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple, List

import numpy as np

from ..data import HurricaneDataPipeline


class HurricaneDataset(Iterable[Tuple[np.ndarray, np.ndarray]]):
    """Iterable of input/target pairs derived from hurricane tracks.

    The dataset iterates over consecutive observations from the storms
    provided at construction time. Each sample contains the state at time
    ``t`` and the state at ``t+1`` as ``numpy.ndarray`` vectors with the
    ordering ``(latitude, longitude, max_wind, min_pressure)``.
    """

    def __init__(self, pipeline: HurricaneDataPipeline, storms: Sequence[str]):
        self.pipeline = pipeline
        self.storms = list(storms)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(
                storm_id, include_era5=False
            )
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            cols = ["latitude", "longitude", "max_wind", "min_pressure"]
            arr = track[cols].to_numpy(dtype=np.float32)
            for i in range(len(arr) - 1):
                self.samples.append((arr[i], arr[i + 1]))

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return iter(self.samples)
