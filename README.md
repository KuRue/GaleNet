# 🌀 GaleNet

**Experimental AI Hurricane Forecasting Toolkit**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GaleNet explores AI‑based techniques for tropical cyclone forecasting. The project currently focuses on data pipelines, baseline training/evaluation scripts, and placeholder GraphCast hooks while full integration with weather models such as GraphCast and Pangu‑Weather remains under active development.

## 🌟 Key Features

- **Hurricane Data Pipeline** – loaders for HURDAT2, IBTrACS, and optional ERA5 patches.
- **Baseline Training & Evaluation Scripts** – minimal examples for model experimentation.
- **GraphCast Placeholder** – stub weights and hooks to prototype future GraphCast integration.
- **Hydra Configuration** – reproducible experiments managed through YAML configs.
- **Modular Design** – architecture prepared for future GraphCast and Pangu‑Weather integration.

## 🗺️ Roadmap

Phase 1 focuses on establishing the foundation for future work:

- Data loaders for HURDAT2/IBTrACS and optional ERA5 patches
- Baseline training and evaluation scripts
- Placeholder GraphCast hooks

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

3. Install the package:
```bash
pip install -e .
```

4. Download sample data and placeholder model weights:
```bash
python scripts/setup_data.py --download-era5 --download-models
```

### Usage Examples

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

```bash
python scripts/evaluate_baselines.py data/sample_storms.json --history 3 --forecast 2 --model-config configs/default_config.yaml
```

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

### Phase 1: Foundation ✅ **In Progress**
- [x] Project structure and environment setup
- [x] Data pipeline with HURDAT2/IBTrACS loaders
- [x] Baseline training and evaluation scripts
- [ ] Replace placeholder GraphCast hooks with full integration and extended docs

### Phase 2: Model Development 📃 **Planned**
- [ ] CNN‑Transformer models
- [ ] Physics‑informed refinements
- [ ] Ensemble experimentation

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
