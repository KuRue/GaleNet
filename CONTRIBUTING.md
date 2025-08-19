# Contributing to GaleNet

Thank you for your interest in contributing to GaleNet! This guide will help you get started.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GaleNet.git
   cd GaleNet
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate galenet
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 📝 Code Style

We use the following tools to maintain code quality:
- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
make lint
```

Format code:
```bash
make format
```

## 🧪 Testing

Write tests for new features:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_loaders.py

# Run with coverage
pytest --cov=galenet
```

## 📚 Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update relevant documentation in `docs/`

Example:
```python
def forecast_storm(storm_id: str, hours: int) -> StormForecast:
    """Generate forecast for a hurricane.
    
    Args:
        storm_id: Hurricane identifier (e.g., "AL052024")
        hours: Forecast horizon in hours
        
    Returns:
        Storm forecast object with track and intensity predictions
    """
```

## 🔄 Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add entry to CHANGELOG.md
4. Submit PR with clear description
5. Link relevant issues

### PR Title Format
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

## 🏗️ Project Structure

```
GaleNet/
├── src/galenet/       # Main package code
│   ├── data/         # Data loading and processing
│   ├── models/       # Model implementations
│   ├── training/     # Training pipelines
│   └── utils/        # Utilities
├── tests/            # Unit tests
├── scripts/          # Utility scripts
├── configs/          # Configuration files
└── notebooks/        # Jupyter notebooks
```

## 🎯 Areas for Contribution

### High Priority
- [x] GraphCast model integration
- [x] Pangu-Weather integration
- [ ] Physics-informed loss functions
- [ ] Real-time data ingestion

### Medium Priority
- [ ] Additional evaluation metrics
- [ ] Visualization improvements
- [ ] Documentation expansion
- [ ] Performance optimization

### Good First Issues
- [ ] Add more unit tests
- [ ] Improve error messages
- [ ] Add type hints
- [ ] Fix documentation typos

## 💡 Feature Requests

Have an idea? Open an issue with:
1. Clear description of the feature
2. Use cases and benefits
3. Potential implementation approach

## 🐛 Bug Reports

Found a bug? Please include:
1. System information (OS, Python version, GPU)
2. Steps to reproduce
3. Expected vs actual behavior
4. Error messages/logs

## 📊 Performance Contributions

When optimizing performance:
1. Profile before and after
2. Include benchmark results
3. Ensure no accuracy degradation
4. Document any trade-offs

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the project

## 📧 Questions?

- Open a GitHub issue
- Join discussions in issues/PRs
- Check existing documentation

Thank you for contributing to GaleNet! 🌀
