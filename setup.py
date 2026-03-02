"""Setup script for gnf-core-foundry."""
from setuptools import setup, find_packages

setup(
    name="gnf-core-foundry",
    version="0.1.0",
    description="Digital biomarker pipeline engine for translational neuroscience",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Global NeuroFoundry",
    author_email="engineering@globalneuro.org",
    url="https://github.com/GlobalNeuroFoundry/gnf-core-foundry",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "pingouin>=0.5.4",
        "statsmodels>=0.14.0",
        "rpy2>=3.5.14",
        "matplotlib>=3.8.0",
        "plotly>=5.20.0",
        "jsonschema>=4.21.0",
        "pyyaml>=6.0.1",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.6.0",
        "pyarrow>=15.0.0",
        "dataclasses-json>=0.6.4",
        "loguru>=0.7.2",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "black>=24.3.0",
            "ruff>=0.3.0",
            "mypy>=1.9.0",
        ],
        "api": [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gnf-run-milestone=gnf_core_foundry.milestone_runner.runner:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
