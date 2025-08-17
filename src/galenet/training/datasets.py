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
    """Stream synchronized track windows across multiple storms.

    The dataset yields tuples of ``(sequence_batch, target_batch)`` or
    ``(sequence_batch, target_batch, era5_batch)`` where each batch contains
    windows for **all** storms provided. ``sequence_batch`` has shape
    ``(num_storms, sequence_window, features)`` and ``target_batch`` has shape
    ``(num_storms, forecast_window, features)``. If ``include_era5`` is true an
    additional tensor with shape ``(num_storms, forecast_window, era5_features)``
    is returned.
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
        lengths: List[int] = []
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(storm_id, include_era5=False)
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            arr = track[cols].to_numpy(dtype=np.float32)
            lengths.append(
                max(len(arr) - self.sequence_window - self.forecast_window + 1, 0)
            )
        self._length = min(lengths) if lengths else 0

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        cols = ["latitude", "longitude", "max_wind", "min_pressure"]
        tracks: List[np.ndarray] = []
        era5_list: List[np.ndarray | None] = []
        for storm_id in self.storms:
            data = self.pipeline.load_hurricane_for_training(
                storm_id, include_era5=self.include_era5
            )
            track = data["track"].sort_values("timestamp").reset_index(drop=True)
            tracks.append(track[cols].to_numpy(dtype=np.float32))

            if self.include_era5:
                era5 = data.get("era5")
                if era5 is not None:
                    try:
                        era5_arr = np.asarray(era5)
                    except Exception:  # pragma: no cover - very defensive
                        era5_arr = np.asarray(getattr(era5, "to_array")())
                else:
                    era5_arr = None
                era5_list.append(era5_arr)
            else:
                era5_list.append(None)

        limit = self._length
        for i in range(max(limit, 0)):
            seq_batch: List[torch.Tensor] = []
            tgt_batch: List[torch.Tensor] = []
            era5_batch: List[torch.Tensor] | None = [] if self.include_era5 else None
            for arr, era5_arr in zip(tracks, era5_list):
                seq_batch.append(torch.from_numpy(arr[i : i + self.sequence_window]))
                tgt_batch.append(
                    torch.from_numpy(
                        arr[
                            i
                            + self.sequence_window : i
                            + self.sequence_window
                            + self.forecast_window
                        ]
                    )
                )
                if era5_batch is not None and era5_arr is not None:
                    era5_batch.append(
                        torch.from_numpy(
                            era5_arr[
                                i
                                + self.sequence_window : i
                                + self.sequence_window
                                + self.forecast_window
                            ]
                        )
                    )
            seq_tensor = torch.stack(seq_batch)
            tgt_tensor = torch.stack(tgt_batch)
            if era5_batch is not None:
                era5_tensor = torch.stack(era5_batch)
                yield (seq_tensor, tgt_tensor, era5_tensor)
            else:
                yield (seq_tensor, tgt_tensor)

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
