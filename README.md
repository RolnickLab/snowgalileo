# SnowGalileo: A Pre-trained Transformer for Snow Cover Mapping

This repository contains the code for pre-training, fine-tuning, and evaluating the ESA AI4Snow model "SnowGalileo". SnowGalileo is a transformer model for daily fractional snow cover (FSC) mapping at 100 m resolution, based on multi-sensor Earth observation data.

To reproduce the figures in the accompanying paper, please visit `paper_visualizations/`.

## Python Version

This project uses **Python 3.11** and relies on a `Makefile` for standardized, reproducible commands.

You can read more about the makefile [here](.make/README.md).

## Package & Environment Management

- **Environment & Dependency Management:** **[uv](https://docs.astral.sh/uv/)** is the **recommended default** tool for fast, reliable dependency installation and virtual environment creation. It can be configured to use **[Poetry](https://python-poetry.org/docs/)** or `conda` via `Makefile.variables`.
  - When we mention `conda` in this project, we generally mean `mamba` or `micromamba` [See Mamba documentation](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)
- **Configuration:** Review the project-level configurations in [Makefile.variables](Makefile.variables) or set individual preferences in `Makefile.private`.

## Quick Start

First, make sure that either the project's [Makefile.variables](Makefile.variables) or [Makefile.private](Makefile.private.example) your choice of configuration.

You can review your current active configurations using this command:

```bash
make info
```

You can list the available targets using this command:

```bash
make targets
```

### Tool-Specific Setup

Select your preferred development stack below. Ensure your `Makefile.variables` are configured to match your choice.

#### 1. Configure Your Stack

Adjust the variables in `Makefile.private` to match your desired setup if they differ from the project's default configuration found in `Makefile.variables` (do this with care and only if necessary):

| Desired Stack                | `DEFAULT_BUILD_TOOL` | `DEFAULT_INSTALL_ENV` |
| :--------------------------- | :------------------- | :-------------------- |
| **uv** (Default/Recommended) | `uv`                 | `uv`                  |
| **Poetry** (Standard)        | `poetry`             | `poetry`              |
| **Poetry + Conda**           | `poetry`             | `conda`               |
| **Poetry + Venv**            | `poetry`             | `venv`                |

#### 2. Install System Tools

If needed, run the command corresponding to your chosen stack to install the necessary system tools (e.g., `uv`, `poetry`, or `mamba`).

<details open>
<summary><strong>Stack: uv </strong></summary>

```bash
make uv-install
```

</details>

<details> <summary><strong>Stack: Poetry</strong></summary>

```bash
make poetry-install
```

</details>

<details> <summary><strong>Stack: Poetry + Conda</strong></summary>

```bash
# Install both the package manager and environment manager
make mamba-install
make poetry-install
```

</details>

### Installing the Project

Once your tools are configured and installed, run the universal install command. This will create the environment and install all dependencies defined in pyproject.toml.

```bash
make install
```

### Activating the Environment

```bash
# Works for uv, poetry, and conda configurations
eval $(make <tool>-activate)
```

Examples:

- uv: `eval $(make uv-activate)`

- poetry: `eval $(make poetry-activate)`

- conda: `eval $(make conda-activate)`

Note: You can also view environment details (path, python version, etc.) by running `make <tool>-env-info` (`poetry` and `conda` only - `uv` does not provide this functionality).

## Project Usage

Information about input data export using Google Earth Engine can be found in `data/README.md`.

### Description of the configs

Also refer to the data retrieval section in data/.

### How to Run Pre-training

For pre-training SnowGalileo, [data] is required. Then run [config] config file.

### How to Run Fine-Tuning

### How to Run Evaluation Experiments

### How to Run Inference on your own Points (preliminary)

Parts of the code require a WandB account to function entirely. If you would like to make use of this, please set the variable [WANDB_ENTITY] in "src/snow_galileo/data/config.py" to your Belieben.

## Environment & Portability Note

This template is designed for reproducibility using the `uv.lock` file.

### Funding
We are greatful to the ESA AI4Science 4000143295/23/I-DT grant that made this project possible.

### Acknowledgements

This repository builds upon the codebase of [Galileo](https://github.com/nasaharvest/galileo), which is licensed under the [MIT](https://opensource.org/license/mit) license. If you use this repository, please also cite the Galileo paper:

```
@misc{tseng2025galileolearninggloballocal,
      title={Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models},
      author={Gabriel Tseng and Anthony Fuller and Marlena Reil and Henry Herzog and Patrick Beukema and Favyen Bastani and James R. Green and Evan Shelhamer and Hannah Kerner and David Rolnick},
      year={2025},
      eprint={2502.09356},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.09356},
}
```

This repository also uses the [SatelliteCloudGenerator](https://github.com/strath-ai/SatelliteCloudGenerator). If you use functionality based on this package, please cite:

```
@Article{rs15174138,
  author = {Czerkawski, Mikolaj and Atkinson, Robert and Michie, Craig and Tachtatzis, Christos},
  title = {SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images},
  journal = {Remote Sensing},
  volume = {15},
  year = {2023},
  number = {17},
  article-number = {4138},
  url = {https://www.mdpi.com/2072-4292/15/17/4138},
  issn = {2072-4292},
  doi = {10.3390/rs15174138}
}
```

The structure of this README was inspired by a template created by Francis Pelletier.

We gratefully acknowledge all original authors and contributors for making their code openly available.
