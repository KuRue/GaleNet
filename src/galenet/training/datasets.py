"""PyTorch datasets for GaleNet training."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import HurricaneDataPipeline


TrackPair = Tuple[torch.Tensor, torch.Tensor]
TrackTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HurricaneDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """Dataset yielding track sequence windows and optional ERA5 patches.

    Parameters
    ----------
    pipeline:
        ``HurricaneDataPipeline`` instance used to load storm data.
    storms:
        Iterable of storm identifiers.
    window:
        Number of sequential observations to include in the input window.
    include_era5:
        Whether to include ERA5 patches in the samples.  When ``True`` and
        corresponding data are available, each sample becomes a triplet of
        ``(sequence, target, era5_patch)``.  Otherwise it yields ``(sequence,
        target)`` pairs.
    """

    def __init__(
        self,
        pipeline: HurricaneDataPipeline,
        storms: Sequence[str],
        window: int = 1,
        include_era5: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.storms = list(storms)
        self.window = int(window)
        self.include_era5 = bool(include_era5)

        self.samples: List[Tuple[torch.Tensor, ...]] = []
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
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

            for i in range(len(arr) - self.window):
                seq = torch.from_numpy(arr[i : i + self.window])
                target = torch.from_numpy(arr[i + self.window])
                if era5_arr is not None:
                    patch = torch.from_numpy(era5_arr[i + self.window])
                    self.samples.append((seq, target, patch))
                else:
                    self.samples.append((seq, target))

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx]


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
