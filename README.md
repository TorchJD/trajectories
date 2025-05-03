# trajectories

This repo uses Jacobian descent to optimize simple multidimensional functions and plots the obtained
optimization trajectories.

## Installation
```bash
uv python install 3.13.3
uv python pin 3.13.3
```

```bash
uv venv
CC=gcc uv pip install "git+ssh://git@github.com/TorchJD/torchjd.git@main[full]"
uv pip install -e . --group check
uv run pre-commit install
```
Note that here, "main" can be replaced with whatever ref (branch, tag or commit hash) of torchjd you
want.

You might also need some tex packages to be able to generate the plots (see
https://stackoverflow.com/a/53080504)

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

## Usage
Please refer to the docstring of the scripts.
