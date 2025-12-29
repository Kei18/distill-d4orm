# distill-d4orm

[<img src="https://img.shields.io/badge/arXiv-2503.12204-990000" alt="Arxiv">](https://arxiv.org/abs/2503.12204)
[<img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube">](https://www.youtube.com/watch?v=WuFuecpZQSY)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CI](https://github.com/Kei18/distill-d4orm/actions/workflows/ci.yml/badge.svg)](https://github.com/Kei18/distill-d4orm/actions/workflows/ci.yml)

A refactored version of **D4orm**, an optimization framework for generating kinodynamically feasible and collision-free multi-robot trajectories using an incremental denoising scheme based on diffusion models.

This codebase extends [proroklab/d4orm](https://github.com/proroklab/d4orm), which itself originates from [LeCAR-Lab/model-based-diffusion](https://github.com/LeCAR-Lab/model-based-diffusion).

## Overview

distill-d4orm is a lightweight reimplementation of D4orm designed to keep the core algorithm and evaluation loop easy to read and modify. It focuses on multi-robot trajectory generation with simple environment models and produces trajectory visualizations and metrics out of the box.

## Features

- JAX-based denoising optimization loop (fast, vectorized rollouts)
- Built-in environments: 2D holonomic, 2D holonomic (random start/goal), 2D holonomic (custom YAML), 3D holonomic, and unicycle
- Metrics and artifacts saved per run (YAML + PNG/GIF)
- Clean CLI via `tyro` for quick experimentation

## Installation

To install the required packages, run the following command:

```bash
uv sync
```

### For CUDA environments

```bash
uv sync --extra cuda12
```

## Quickstart

```bash
uv run app.py
```

The results will be stored in `outputs/<timestamp>/`.

To see available options:

```bash
uv run app.py --help
```

Example:

```sh
CUDA_VISIBLE_DEVICES=3 uv run app.py --n_agents 10 --anytime --Niteration 30 --save-gif
```

![](./assets/trajectories.gif)

## Environments

Use `--env-name` to select a scenario:

- `2dholo` (default): 2D holonomic with circular start/goal arrangement
- `2dholo_random`: random start/goal positions within a workspace
- `2dholo_custom`: load starts, goals, and obstacles from a YAML file
- `3dholo`: 3D holonomic setup
- `unicycle`: 2D unicycle dynamics

## Common options

The CLI is generated from `Args` in `app.py`, which merges environment and D4orm configs. A few useful knobs:

- `--n_agents`: number of robots
- `--Nsample`: number of samples per denoising step
- `--Hsample`: planning horizon
- `--Ndiffuse`: number of denoising steps
- `--Niteration`: number of outer iterations
- `--anytime`: keep improving after success until `Niteration`
- `--save-img` / `--save-gif`: save trajectory visualization

For the full list, run `uv run app.py --help`.


### Obstacles, start-goal specification

```sh
uv run app.py --n-agents 2 --env-name 2dholo_custom --external_file assets/2dholo_custom.yaml
```

![](./assets/2dholo_custom.png)

The custom scenario YAML should define starts/goals and (optional) obstacles:

```yaml
problem:
  n_obstacles: 2
  obstacles:
    - center: [0.0, 0.1]
      size: [0.3]
      type: circle
  n_agents: 2
  terminals:
    - start: [0.0, -1.0, 0, 0]
      goal: [0.0, 1.5, 0, 0]
```

## Outputs

Each run writes a timestamped directory under `outputs/`:

- `config.yaml`: CLI arguments used for the run
- `result.yaml`: metrics, environment instance, and per-agent trajectories
- `trajectories.png`: static trajectory plot (if `--save-img`)
- `trajectories.gif`: animation (if `--save-gif`)

## Project structure

- `app.py`: entrypoint CLI
- `src/d4orm`: optimizer, environments, and visualization utilities
- `assets`: example GIFs and custom environment YAML
- `tests`: smoke tests for the optimizer loop

## Development

Run tests:

```bash
uv run pytest
```

## Citation

```bibtex
@article{zhang2025d4orm,
  title={D4orm: Multi-Robot Trajectories with Dynamics-aware Diffusion Denoised Deformations},
  author={Zhang, Yuhao and Okumura, Keisuke and Woo, Heedo and Shankar, Ajay and Prorok, Amanda},
  journal={arXiv preprint arXiv:2503.12204},
  year={2025}
}
```

## License

MIT. See `LICENSE`.
