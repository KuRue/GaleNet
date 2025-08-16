"""PyTorch datasets for GaleNet training."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..data import HurricaneDataPipeline


TrackPair = Tuple[torch.Tensor, torch.Tensor]
TrackTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HurricaneDataset(IterableDataset[Tuple[torch.Tensor, ...]]):
    """Stream track windows across multiple storms.

    The dataset yields tuples of ``(sequence, target)`` or
    ``(sequence, target, era5_patch)`` depending on the ``include_era5``
    flag. ``sequence`` spans ``sequence_window`` observations while ``target``
    contains ``forecast_window`` future observations.
    """

    def __init__(
        self,
        pipeline: HurricaneDataPipeline,
        storms: Sequence[str],
        sequence_window: int = 1,
        forecast_window: int = 1,
        include_era5: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.storms = list(storms)
        self.sequence_window = int(sequence_window)
        self.forecast_window = int(forecast_window)
        self.include_era5 = bool(include_era5)

        # Pre-compute the total length for __len__ without storing samples.
        self._length = 0
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(storm_id, include_era5=False)
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            arr = track[cols].to_numpy(dtype=np.float32)
            self._length += max(
                len(arr) - self.sequence_window - self.forecast_window + 1, 0
            )

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(
                storm_id, include_era5=self.include_era5
            )
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            arr = track[cols].to_numpy(dtype=np.float32)

            era5 = data.get("era5") if self.include_era5 else None
            if era5 is not None:
                try:
                    era5_arr = np.asarray(era5)
                except Exception:  # pragma: no cover - very defensive
                    era5_arr = np.asarray(getattr(era5, "to_array")())
            else:
                era5_arr = None

            limit = len(arr) - self.sequence_window - self.forecast_window + 1
            for i in range(max(limit, 0)):
                seq = torch.from_numpy(
                    arr[i : i + self.sequence_window]
                )
                target = torch.from_numpy(
                    arr[
                        i
                        + self.sequence_window : i
                        + self.sequence_window
                        + self.forecast_window
                    ]
                )
                if era5_arr is not None:
                    patch = torch.from_numpy(
                        era5_arr[
                            i
                            + self.sequence_window : i
                            + self.sequence_window
                            + self.forecast_window
                        ]
                    )
                    yield (seq, target, patch)
                else:
                    yield (seq, target)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length


# ---------------------------------------------------------------------------
def create_dataloader(
    dataset: HurricaneDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs: object,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` for ``dataset``."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


__all__ = ["HurricaneDataset", "create_dataloader"]
