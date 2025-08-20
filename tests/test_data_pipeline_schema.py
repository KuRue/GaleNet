import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

pandas_spec = importlib.util.find_spec("pandas")
xarray_spec = importlib.util.find_spec("xarray")

pytestmark = pytest.mark.skipif(
    pandas_spec is None or xarray_spec is None,
    reason="pandas and xarray required",
)

if pandas_spec:
    import pandas as pd  # type: ignore
if xarray_spec:
    import xarray as xr  # type: ignore

from galenet.data.pipeline import HurricaneDataPipeline  # noqa: E402


class DummyHURDAT2Loader:
    def get_storm(self, storm_id: str) -> "pd.DataFrame":  # type: ignore
        times = pd.date_range("2023-01-01", periods=2, freq="6H")
        data = {
            "storm_id": [storm_id] * 2,
            "name": ["TEST"] * 2,
            "timestamp": times,
            "record_identifier": ["A", "B"],
            "storm_type": ["TD", "TS"],
            "latitude": [10.0, 10.5],
            "longitude": [-40.0, -40.5],
            "max_wind": [30.0, 40.0],
            "min_pressure": [1008.0, 1005.0],
            "34kt_ne": [20.0, 30.0],
            "34kt_se": [20.0, 30.0],
            "34kt_sw": [15.0, 25.0],
            "34kt_nw": [15.0, 25.0],
        }
        return pd.DataFrame(data)


class DummyERA5Loader:
    def extract_hurricane_patches(
        self,
        track_df: "pd.DataFrame",  # type: ignore
        patch_size: float = 25.0,
        variables=None,
        lead_time_hours: int = 6,
        lag_time_hours: int = 6,
    ) -> "xr.Dataset":  # type: ignore
        time = track_df["timestamp"]
        lat = np.linspace(track_df["latitude"].min() - 1, track_df["latitude"].min() + 1, 2)
        lon = np.linspace(track_df["longitude"].min() - 1, track_df["longitude"].min() + 1, 2)
        shape = (len(time), len(lat), len(lon))
        variables = [
            "u10",
            "v10",
            "msl",
            "t2m",
            "d2m",
            "u200",
            "v200",
            "u850",
            "v850",
            "sst",
        ]
        data_vars = {var: (("time", "latitude", "longitude"), np.zeros(shape)) for var in variables}
        return xr.Dataset(data_vars, coords={"time": time, "latitude": lat, "longitude": lon})


def test_pipeline_output_schema():
    pipeline = HurricaneDataPipeline.__new__(HurricaneDataPipeline)
    pipeline.hurdat2 = DummyHURDAT2Loader()
    pipeline.era5 = DummyERA5Loader()
    pipeline.ibtracs = SimpleNamespace()
    pipeline._cache = {}

    storm_ids = ["AL012023", "AL022023"]
    expected_track_cols = {
        "storm_id",
        "name",
        "timestamp",
        "latitude",
        "longitude",
        "max_wind",
        "min_pressure",
        "34kt_ne",
        "34kt_se",
        "34kt_sw",
        "34kt_nw",
    }
    expected_vars = {
        "u10",
        "v10",
        "msl",
        "t2m",
        "d2m",
        "u200",
        "v200",
        "u850",
        "v850",
        "sst",
    }

    for sid in storm_ids:
        result = pipeline.load_hurricane_for_training(sid)
        track = result["track"]
        ds = result["era5"]

        assert expected_track_cols <= set(track.columns)
        assert track["storm_id"].dtype == object
        assert track["name"].dtype == object
        assert pd.api.types.is_datetime64_any_dtype(track["timestamp"])
        for col in expected_track_cols - {"storm_id", "name", "timestamp"}:
            assert pd.api.types.is_numeric_dtype(track[col])

        assert set(ds.data_vars) == expected_vars
        assert ds.dims["time"] == len(track)
        assert ds.dims["latitude"] == 2
        assert ds.dims["longitude"] == 2
        for var in expected_vars:
            assert np.issubdtype(ds[var].dtype, np.floating)
