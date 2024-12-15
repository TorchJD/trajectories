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

You might also need some tex packages to be able to generate the plots (see
https://stackoverflow.com/a/53080504)

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

## Usage
```bash
pdm run optimize
pdm run plot
```
