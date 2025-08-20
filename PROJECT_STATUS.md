# Project Status

This document provides a quick snapshot of GaleNet's progress and what's next.

## Current Progress
- Core package structure with testing and packaging helpers.
- Loaders for HURDAT2 and IBTrACS hurricane tracks.
- ERA5 patch extraction and preprocessing utilities.
- Baseline training and evaluation scripts with metrics.
- Installation, data pipeline, architecture, training, and API documentation.
- Tutorial notebooks demonstrating basic workflows.
- Full GraphCast integration with accompanying training and pipeline docs.
- Full Pangu-Weather inference integration with accompanying pipeline docs.
- Comprehensive evaluation guide with GraphCast and Pangu-Weather examples.

## Remaining Milestones
- **Completed**
   - Comprehensive evaluation guide with GraphCast and Pangu-Weather examples
- **Upcoming**
   - Expand end-to-end data pipeline documentation and schema coverage
   - Train CNN‑Transformer baseline and report 24–72 h track RMSE (<20 km at 24 h)
   - Prototype physics‑informed modules with ≥10 % improvement in conservation metrics
   - Evaluate ensemble strategy with ≥5 % gain in CRPS and reliability over single models

## Phase Goals
1. **Phase 1 – Data Foundation** ✅ *Completed*: finalize dataset loaders, integrate GraphCast and Pangu baselines, and document evaluation workflows.
2. **Phase 2 – Model Development**: deliver CNN‑Transformer baseline, physics‑informed modules, and ensemble strategy.
3. **Phase 3 – Evaluation & Deployment**: benchmark models, refine APIs, and prepare for public release.

_Last updated: 2025-08-20_
