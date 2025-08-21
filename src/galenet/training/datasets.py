"""PyTorch datasets for GaleNet training."""

from __future__ import annotations

from typing import Any, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..data import HurricaneDataPipeline


class HurricaneDataset(IterableDataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Stream track histories, image patches and targets across storms.

    Each yielded tuple contains ``(track_batch, target_batch, patch_batch)``.
    ``track_batch`` has shape ``(num_storms, sequence_window, features)`` where
    ``features`` correspond to ``[latitude, longitude, max_wind, min_pressure]``.
    ``patch_batch`` has shape ``(num_storms, sequence_window, channels, H, W)``
    while ``target_batch`` has shape ``(num_storms, forecast_window, features)``.
    """

    def __init__(
        self,
        pipeline: HurricaneDataPipeline,
        storms: Sequence[str],
        sequence_window: int = 1,
        forecast_window: int = 1,
        patch_size: float = 25.0,
    ) -> None:
        self.pipeline = pipeline
        self.storms = list(storms)
        self.sequence_window = int(sequence_window)
        self.forecast_window = int(forecast_window)
        self.patch_size = float(patch_size)

        # Determine minimal length across storms using track data only
        lengths: List[int] = []
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(
                storm_id, include_era5=False, include_satellite=False
            )
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            arr = track[cols].to_numpy(dtype=np.float32)
            lengths.append(max(len(arr) - self.sequence_window - self.forecast_window + 1, 0))
        self._length = min(lengths) if lengths else 0

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        tracks: List[np.ndarray] = []
        patches: List[np.ndarray] = []
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(
                storm_id,
                include_era5=True,
                include_satellite=True,
                patch_size=self.patch_size,
            )
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            tracks.append(track[cols].to_numpy(dtype=np.float32))

            patch_arrays: List[np.ndarray] = []
            sat = data.get("satellite")
            if sat is not None:
                sat_arr = (
                    sat.to_array().transpose("time", "variable", "latitude", "longitude").to_numpy()
                )
                patch_arrays.append(sat_arr)
            era5 = data.get("era5")
            if era5 is not None:
                era5_arr = (
                    era5.to_array()
                    .transpose("time", "variable", "latitude", "longitude")
                    .to_numpy()
                )
                patch_arrays.append(era5_arr)
            patches.append(np.concatenate(patch_arrays, axis=1))

        limit = self._length
        for i in range(max(limit, 0)):
            track_batch: List[torch.Tensor] = []
            tgt_batch: List[torch.Tensor] = []
            patch_batch: List[torch.Tensor] = []
            for patch_arr, track_arr in zip(patches, tracks):
                track_batch.append(torch.from_numpy(track_arr[i : i + self.sequence_window]))
                patch_batch.append(torch.from_numpy(patch_arr[i : i + self.sequence_window]))
                tgt_batch.append(
                    torch.from_numpy(
                        track_arr[
                            i
                            + self.sequence_window : i
                            + self.sequence_window
                            + self.forecast_window
                        ]
                    )
                )
            track_tensor = torch.stack(track_batch)
            tgt_tensor = torch.stack(tgt_batch)
            patch_tensor = torch.stack(patch_batch)
            yield (track_tensor, tgt_tensor, patch_tensor)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length


# ---------------------------------------------------------------------------
def create_dataloader(
    dataset: HurricaneDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` for ``dataset``."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


__all__ = ["HurricaneDataset", "create_dataloader"]
