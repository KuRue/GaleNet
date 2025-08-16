# Installation

This guide walks through setting up GaleNet on a new machine.

## Prerequisites

- Linux or macOS
- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/)
- Git

## Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/KuRue/GaleNet.git
   cd GaleNet
   ```
2. **Create the environment**
   ```bash
   conda env create -f environment.yml
   conda activate galenet
   ```
3. **Install in development mode**
   ```bash
   pip install -e .
   ```
4. **Fetch required datasets**
   ```bash
   python scripts/setup_data.py --all
   ```
5. **Verify the installation**
   ```bash
   pytest tests -q
   ```

You are ready to explore the project and train models.
