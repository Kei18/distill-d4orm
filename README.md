<div align="center">
  <h1>D4orm: Multi-Robot Trajectories with <br>Dynamics-aware Diffusion Denoised Deformations</h2>
  <strong>IROS 2025</strong>
  <br>
    <a href="https://github.com/yuhaozhang7" target="_blank">Yuhao Zhang</a><sup>1</sup>,
    <a href="https://kei18.github.io/" target="_blank">Keisuke Okumura</a><sup>1,2</sup>,
    <a href="https://proroklab.org/team/" target="_blank">Heedo Woo</a><sup>1</sup>,
    <a href="https://www.cl.cam.ac.uk/~as3233/" target="_blank">Ajay Shankar</a><sup>1</sup>,
    <a href="https://www.cst.cam.ac.uk/people/asp45" target="_blank">Amanda Prorok</a><sup>1</sup>
  <p>
    <h45>
      <sup>1</sup>Prorok Lab, University of Cambridge &nbsp;&nbsp;
      <sup>2</sup>AIST Japan &nbsp;&nbsp;
    </h5>
  </p>

  [<img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube">](https://www.youtube.com/watch?v=WuFuecpZQSY)
  [<img src="https://img.shields.io/badge/arXiv-2503.12204-990000" alt="Arxiv">](https://arxiv.org/abs/2503.12204)

</div>

<p align="center">
  <img src="assets/d4orm_deployment.gif" alt="Zero-Shot Deployment" width="80%">
</p>


This repository includes code for **D4orm**, based on the [model-based diffusion](https://github.com/LeCAR-Lab/model-based-diffusion).

**D4orm** is an optimization framework for generating kinodynamically feasible and collision-free multi-robot trajectories that exploits an incremental denoising scheme in diffusion models.


## Installation

To install the required packages, run the following command:

```bash
uv sync
```

### For CUDA environment

```bash
uv sync --extra cuda12
```

## Run D4orm

To run multi-robot trajectory planning with D4orm, use the following command:

```bash
uv run src/d4orm/d4orm.py
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
