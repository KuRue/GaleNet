"""Setup configuration for GaleNet package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("git+")
    ]

setup(
    name="galenet",
    version="0.1.0",
    author="kurue",
    author_email="galenet@kurue.com",
    description=(
        "GaleNet - AI-powered hurricane forecasting system using GraphCast "
        "and Pangu-Weather"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KuRue/GaleNet",
    project_urls={
        "Bug Tracker": "https://github.com/KuRue/GaleNet/issues",
        "Documentation": "https://galenet.readthedocs.io",
        "Source Code": "https://github.com/KuRue/GaleNet",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
        ],
        "viz": [
            "plotly>=5.18.0",
            "bokeh>=3.3.0",
            "folium>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "galenet=galenet.cli:main",
            "galenet-train=galenet.training.train:main",
            "galenet-evaluate=galenet.evaluation.evaluate:main",
            "galenet-serve=galenet.inference.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "galenet": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    zip_safe=False,
)
