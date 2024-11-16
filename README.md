# trajectories

This repo uses Jacobian descent to optimize simple multidimensional functions and plots the obtained
optimization trajectories.

## Installation
```bash
pdm venv create 3.12.3  # Requires python 3.12.3 to be installed
pdm use -i .venv/bin/python
pdm install --frozen-lockfile
pdm run pre-commit install
```

## Usage
```bash
pdm run optimize
pdm run plot
```
