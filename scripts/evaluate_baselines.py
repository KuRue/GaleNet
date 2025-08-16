#!/usr/bin/env python3
"""CLI for evaluating baseline forecasts across storms.

This script expects a JSON file containing a list of storms. Each storm is a
mapping with a ``track`` key holding a list of ``[lat, lon, intensity]``
triplets. The first ``--history`` steps of each track are used as input for the
baselines and the next ``--forecast`` steps are compared against the baseline
forecasts using the evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from galenet.evaluation.baselines import evaluate_baselines


def _load_storms(path: Path) -> List[np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return [np.asarray(storm["track"], dtype=float) for storm in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline forecasts")
    parser.add_argument("data", type=Path, help="Path to JSON file containing storm tracks")
    parser.add_argument("--history", type=int, default=3, help="Number of history steps")
    parser.add_argument("--forecast", type=int, default=2, help="Number of forecast steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    storms = _load_storms(args.data)
    results = evaluate_baselines(storms, args.history, args.forecast)
    for baseline, metrics in results.items():
        logging.info("%s:", baseline)
        for name, value in metrics.items():
            logging.info("  %s: %.3f", name, value)


if __name__ == "__main__":
    main()
