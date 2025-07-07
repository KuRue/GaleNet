# GaleNet Quick Start Guide 🚀

Get up and running with GaleNet in 5 minutes!

## Prerequisites

- Linux/macOS system with NVIDIA GPU (optional but recommended)
- ~50GB free disk space for data
- Conda/Mamba package manager
- Git

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KuRue/GaleNet.git
cd GaleNet
```

### 2. Quick Setup (All-in-One)
```bash
make quickstart
```

This command will:
- Create the conda environment
- Install GaleNet in development mode
- Download hurricane data
- Run validation tests

### 3. Manual Setup (Step-by-Step)

If you prefer manual setup or the quickstart fails:

```bash
# Create environment
conda env create -f environment.yml
conda activate galenet

# Install package
pip install -e .

# Download data
python scripts/setup_data.py --all

# Test installation
python scripts/test_data_loading.py
```

## First Run

### 1. Start Jupyter Lab
```bash
conda activate galenet
make notebook
```

Then open http://localhost:8888 in your browser.

### 2. Run the Quick Start Notebook

Navigate to `notebooks/01_galenet_quickstart.ipynb` and run all cells.

### 3. Test Data Loading (CLI)
```bash
python scripts/test_data_loading.py
```

Expected output:
```
GaleNet Data Loading Test Suite
============================================================
Running: HURDAT2 Loading
✅ HURDAT2 data found
✅ Loaded 50,000+ records from 1,800+ storms
...
🎉 All tests passed!
```

## Basic Usage

### Python API
```python
from galenet import HurricaneDataPipeline

# Initialize pipeline
pipeline = HurricaneDataPipeline()

# Load hurricane data
hurricane_data = pipeline.load_hurricane_for_training(
    storm_id="AL092023",  # Hurricane Lee
    include_era5=False    # Skip ERA5 for quick test
)

# Access track data
track = hurricane_data['track']
print(f"Storm: {track['name'].iloc[0]}")
print(f"Peak intensity: {track['max_wind'].max()} kt")
```

### Command Line
```bash
# Check GPU availability
make check-gpu

# Run tests
make test

# Format code
make format

# Start MLflow tracking
make mlflow
```

## Common Issues

### 1. HURDAT2 Data Not Found
```bash
# Download manually
python scripts/setup_data.py --download-hurdat2
```

### 2. Import Errors
```bash
# Ensure environment is activated
conda activate galenet

# Reinstall in development mode
pip install -e .
```

### 3. No GPU Detected
```bash
# Check CUDA installation
nvidia-smi

# CPU-only mode is supported but slower
```

### 4. ERA5 Download Issues
ERA5 requires Copernicus Climate Data Store credentials:
1. Register at https://cds.climate.copernicus.eu/user/register
2. Get API key from https://cds.climate.copernicus.eu/api-how-to
3. Create `~/.cdsapirc` with your credentials

## Project Structure
```
GaleNet/
├── src/galenet/       # Main package
│   ├── data/         # Data loading & preprocessing
│   ├── models/       # Model implementations (coming soon)
│   ├── training/     # Training pipelines (coming soon)
│   └── utils/        # Utilities
├── notebooks/         # Example notebooks
├── scripts/          # Utility scripts
├── configs/          # Configuration files
└── tests/            # Unit tests
```

## Next Steps

1. **Explore the Data**: Run the quickstart notebook to visualize hurricane tracks
2. **Preprocess Data**: Use `HurricanePreprocessor` to prepare training data
3. **Download ERA5**: Get reanalysis data for atmospheric context
4. **Stay Tuned**: Model training capabilities coming soon!

## Getting Help

- 📚 Check the [full documentation](docs/)
- 🐛 Report issues on [GitHub](https://github.com/KuRue/GaleNet/issues)
- 💬 Join discussions in GitHub Issues
- 📧 Contact: your.email@example.com

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

Happy forecasting! 🌀
