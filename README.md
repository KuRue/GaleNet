# 🌀 GaleNet

**Experimental AI Hurricane Forecasting Toolkit**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Pangu-Weather](https://img.shields.io/badge/Pangu--Weather-supported-brightgreen.svg)](https://github.com/198808xc/Pangu-Weather)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GaleNet explores AI‑based techniques for tropical cyclone forecasting. The project currently focuses on data pipelines, baseline training/evaluation scripts, GraphCast training, and Pangu‑Weather inference integration.

## 🌟 Key Features

- **Hurricane Data Pipeline** – loaders for HURDAT2, IBTrACS, and optional ERA5 patches.
- **Baseline Training & Evaluation Scripts** – minimal examples for model experimentation.
- **GraphCast Training & Pangu-Weather Inference** – integrated training utilities for GraphCast and an inference pipeline using Pangu‑Weather; see the [training guide](docs/training.md) and [pipeline docs](docs/data_pipeline.md).
- **Hydra Configuration** – reproducible experiments managed through YAML configs.
- **Modular Design** – architecture prepared for additional forecasting models.

## 🗺️ Roadmap

Phase 1 focuses on establishing the foundation for future work:

- Data loaders for HURDAT2/IBTrACS and optional ERA5 patches
- Baseline training and evaluation scripts

### ✅ Completed

- Full GraphCast training and Pangu‑Weather inference integration with pipeline documentation

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the full list of milestones and progress updates.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KuRue/GaleNet.git
cd GaleNet
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate galenet
```

The provided environment installs the CPU build of PyTorch by default. If you
have a compatible GPU, consult the [PyTorch installation
guide](https://pytorch.org/get-started/locally/) to install the appropriate
CUDA-enabled packages.

3. Install the package:
```bash
pip install -e .
```

4. Download sample data and model weights:
```bash
python scripts/setup_data.py --download-era5 --download-models
```

### Usage Examples

#### Data Pipeline

Prepare datasets for a single storm with the setup script:

```bash
python scripts/setup_data.py --storm AL012022
```

By default, files are saved under `~/data/galenet/processed/AL012022/` (or your chosen `--data-dir`).
See the [Data Pipeline guide](docs/data_pipeline.md) for additional details.

#### Forecasting

```python
from galenet import GaleNetPipeline

# Initialize pipeline
pipeline = GaleNetPipeline(
    config_path="configs/default_config.yaml"
)

# Generate forecast for active storm
forecast = pipeline.forecast_storm(
    storm_id="AL052024",  # Hurricane ID
    forecast_hours=120,    # 5-day forecast
    ensemble_size=5        # Ensemble members
)

# Access results
print(f"24h position: {forecast.get_position(24)}")
print(f"Peak intensity: {forecast.max_intensity} mph")
print(f"Track uncertainty: {forecast.track_cone}")
```

#### Training

```bash
python scripts/train_model.py training.epochs=5 training.batch_size=8
```

#### Evaluation

See the [Evaluation Guide](docs/evaluation.md) for full details.

Evaluate baselines alongside a trained GraphCast or Pangu model (ensure the
config file points to your model checkpoint):

```bash
python scripts/evaluate_baselines.py AL012022 AL022022 --history 3 --forecast 2 \
    --model graphcast=configs/default_config.yaml \
    --model pangu=configs/default_config.yaml \
    --output results/summary.csv --details results/per_storm.csv
```

The command writes a summary table to `results/summary.csv` and per-storm
metrics to `results/per_storm.csv`. Each row of the summary corresponds to a
forecast method (baseline or model) with columns such as `track_error`,
`along_track_error`, `cross_track_error` (km), `intensity_mae` (kt), and
`rapid_intensification_skill` (F1). Lower error values indicate better
performance.

## 📁 Project Structure

```
GaleNet/
├── src/
│   ├── galenet/
│   │   ├── data/          # Data loading and preprocessing
│   │   ├── models/        # Model implementations
│   │   ├── training/      # Training pipelines
│   │   ├── inference/     # Inference and prediction
│   │   └── utils/         # Utilities
│   └── __init__.py
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── docker/                # Docker configurations
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

GaleNet uses Hydra for configuration management. Main configuration file: `configs/default_config.yaml`

```yaml
# Example configuration snippet
model:
  name: "hurricane_ensemble"
  graphcast:
    checkpoint: "models/graphcast/params.npz"
    resolution: 0.25
  ensemble:
    size: 50
    method: "perturbation"

data:
  era5:
    path: "/data/era5"
    variables: ["u10", "v10", "msl", "t2m", "sst"]
  hurdat:
    path: "/data/hurdat2/hurdat2.txt"
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t galenet:latest -f docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 galenet:latest
```

## 📈 Current Development Status

### Phase 1: Foundation ✅ **Completed**
- [x] Project structure and environment setup
- [x] Data pipeline with HURDAT2/IBTrACS loaders
- [x] Baseline training and evaluation scripts
- [x] Full GraphCast integration with extended documentation
- [x] Full Pangu-Weather inference integration with extended documentation
- [x] Comprehensive evaluation guide with GraphCast and Pangu-Weather examples

### Phase 2: Model Development 📃 **Planned**

- [ ] **CNN‑Transformer baseline**
  - *Goal*: 24–72 h track and intensity RMSE compared against GraphCast with <20 km track error at 24 h.

- [ ] **Physics‑informed module prototypes**
  - *Goal*: Conservation metrics (e.g., mass, energy) improve ≥10 % over baseline.

- [ ] **Ensemble strategy**
  - *Goal*: Continuous ranked probability score and reliability diagrams show ≥5 % improvement over best single model.

### Phase 3: Optimization 🔧 **Planned**
- [ ] Memory and performance tuning
- [ ] Comprehensive validation

### Phase 4: Deployment 🚀 **Planned**
- [ ] API and packaging
- [ ] Monitoring and infrastructure

## 📚 Documentation

- [Data Pipeline](docs/data_pipeline.md)
- [Data Workflow](docs/data_workflow.md)
- [Training Guide](docs/training.md)
- [Evaluation Guide](docs/evaluation.md)
- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Model Architecture](docs/architecture.md)
- [Tutorial Notebooks](notebooks/)

## 🔬 Research

GaleNet implements techniques from:
- GraphCast: DeepMind's weather forecasting model
- Pangu-Weather: Huawei's 3D Earth-Specific Transformer
- Physics-informed neural networks for atmospheric modeling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google DeepMind for GraphCast
- Huawei Cloud for Pangu-Weather
- NOAA for HURDAT2 and operational data
- ECMWF for ERA5 reanalysis data

## 📧 Contact

For questions or suggestions, please open an issue on [GitHub](https://github.com/KuRue/GaleNet/issues).

---
⚡ Built with PyTorch | 🌊 Powered by AI | 🌀 Protecting Communities
