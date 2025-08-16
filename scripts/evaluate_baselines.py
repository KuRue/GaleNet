#!/usr/bin/env python3
"""CLI for evaluating baseline and model forecasts across storms.

The tool expects a JSON file containing a list of storms.  Each storm is a
mapping with an ``id`` field and a ``track`` key holding a list of
``[lat, lon, intensity]`` triplets.  The first ``--history`` steps of each track
are used as input for the baselines and the next ``--forecast`` steps are
compared against the forecasts using the evaluation metrics.  A
``GaleNetPipeline`` instance is created (using ``--model-config`` if supplied)
and its model forecasts are evaluated alongside the baselines.  A summary table
is printed and can optionally be written to a CSV file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple
import sys

import pandas as pd

import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from galenet.evaluation.baselines import evaluate_baselines

from galenet import GaleNetPipeline


def _load_storms(path: Path) -> List[Tuple[str, np.ndarray]]:
    """Load storm tracks from ``path``.

    Each entry in the JSON file should contain an ``id`` and ``track`` field.
    When ``id`` is missing a numeric identifier is generated.  The return value
    is a list of ``(storm_id, track_array)`` tuples where ``track_array`` has
    shape ``(N, 3)`` with latitude, longitude and intensity columns.
    """

    with open(path) as f:
        data = json.load(f)

    storms: List[Tuple[str, np.ndarray]] = []
    for idx, storm in enumerate(data):
        storm_id = str(storm.get("id", idx))
        track = np.asarray(storm["track"], dtype=float)
        storms.append((storm_id, track))
    return storms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and model forecasts"
    )
    parser.add_argument("data", type=Path, help="Path to JSON file containing storm tracks")
    parser.add_argument("--history", type=int, default=3, help="Number of history steps")
    parser.add_argument("--forecast", type=int, default=2, help="Number of forecast steps")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional path to config for running GaleNetPipeline forecasts",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name to use for reporting model metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write CSV summary",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip running GaleNetPipeline model forecasts",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        storms_with_ids = _load_storms(args.data)
        tracks: Sequence[np.ndarray] = [t for _, t in storms_with_ids]

        model_forecasts = None
        if not args.no_model:
            pipeline = GaleNetPipeline(
                str(args.model_config) if args.model_config else None
            )
            model_forecasts = []
            for storm_id, track in storms_with_ids:
                logging.info("Running model for storm %s", storm_id)
                history = track[: args.history]
                df_hist = pd.DataFrame(
                    history, columns=["latitude", "longitude", "max_wind"]
                )
                preds = pipeline.model.predict(df_hist, args.forecast, 1)
                model_forecasts.append(
                    preds[["latitude", "longitude", "max_wind"]].to_numpy()
                )

        results = evaluate_baselines(
            tracks,
            args.history,
            args.forecast,
            model_forecasts=model_forecasts,
            model_name=args.model_name,
        )

        df = pd.DataFrame(results).T
        print(df.to_string(float_format=lambda x: f"{x:.3f}"))
        if args.output is not None:
            df.to_csv(args.output)

        if df.isna().any().any():
            raise RuntimeError("NaN encountered in evaluation metrics")

    except Exception as exc:  # pragma: no cover - CLI failure path
        logging.error("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
