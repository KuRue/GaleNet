#!/usr/bin/env python3
"""CLI for evaluating baseline and model forecasts across storms.

This script expects a JSON file containing a list of storms. Each storm is a
mapping with a ``track`` key holding a list of ``[lat, lon, intensity]``
triplets. The first ``--history`` steps of each track are used as input for the
baselines and the next ``--forecast`` steps are compared against the forecasts
using the evaluation metrics.  When ``--model-config`` is provided, a
``GaleNetPipeline`` instance is created and its model is run alongside the
baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

import numpy as np

from galenet.evaluation.baselines import evaluate_baselines
from galenet import GaleNetPipeline


def _load_storms(path: Path) -> List[np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return [np.asarray(storm["track"], dtype=float) for storm in data]


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    storms = _load_storms(args.data)
    model_forecasts = None
    if args.model_config is not None:
        pipeline = GaleNetPipeline(str(args.model_config))
        model_forecasts = []
        for storm in storms:
            history = storm[: args.history]
            df = pd.DataFrame(history, columns=["latitude", "longitude", "max_wind"])
            preds = pipeline.model.predict(df, args.forecast, 1)
            model_forecasts.append(
                preds[["latitude", "longitude", "max_wind"]].to_numpy()
            )

    results = evaluate_baselines(
        storms,
        args.history,
        args.forecast,
        model_forecasts=model_forecasts,
        model_name=args.model_name,
    )
    for baseline, metrics in results.items():
        logging.info("%s:", baseline)
        for name, value in metrics.items():
            logging.info("  %s: %.3f", name, value)


if __name__ == "__main__":
    main()
