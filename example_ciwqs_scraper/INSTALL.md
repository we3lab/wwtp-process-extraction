# Setting up environment to run CIWQS scraper

## Prerequisites
- Python 3.8 or higher
- conda (Anaconda or Miniconda)

## Installation Steps

### 1. Create a Conda Environment (Recommended)

It's recommended to create a conda environment to isolate the project dependencies:
```bash
conda create -n ciwqs-scraper python=3.11
```
Activate conda environment
```bash
conda activate ciwqs-scraper
```

### 2. Install the Project in Editable Mode

Install the project requirements from pyproject.toml and all its dependencies using pip:

```bash
pip install -e .
```