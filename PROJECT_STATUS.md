# Project Status

This document provides a quick snapshot of GaleNet's progress and what's next.

## Current Progress
- Core package structure and data pipeline scaffolding in place.
- Initial data loaders for HURDAT2 and IBTrACS hurricane track datasets.
- Early preprocessing utilities and validation helpers implemented.
- Repository tooling for testing, linting, and packaging configured.
- Expanded preprocessing and feature engineering for ERA5 dataset.
- Evaluation framework and benchmarking suite established.
- Published data pipeline documentation and tutorial notebooks.
- Installation, architecture, training, and API reference documentation published.

## Remaining Milestones
- Develop full training loop for GraphCast/Pangu-based models.
- Achieve full GraphCast integration into the pipeline.
- Build comprehensive documentation and usage examples.
- **Data Foundation**
   - Finalize dataset loaders for ERA5, HURDAT2, and IBTrACS *(High priority, target: 2025-09)*
   - Validate preprocessing pipeline and feature engineering steps *(High priority, target: 2025-09)*
   - Expand end-to-end data pipeline documentation and schema coverage *(Medium priority, target: 2025-10)*

## Phase Goals
1. **Data Foundation**
   - Finalize dataset loaders for ERA5, HURDAT2, and IBTrACS *(High priority, target: 2025-09)*
   - Validate preprocessing pipeline and feature engineering steps *(High priority, target: 2025-09)*
   - Expand end-to-end data pipeline documentation and schema coverage *(Medium priority, target: 2025-10)*
2. **Model Development** – integrate forecast models and train baseline hurricane predictors.
3. **Evaluation & Deployment** – benchmark models, refine APIs, and prepare for public release.

_Last updated: 2025-08-16_
