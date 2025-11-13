from datetime import datetime

import time
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import partial

from loguru import logger
import yaml
from dacite import from_dict, Config

import tyro
import jax
from jax import numpy as jnp, config

from .envs import get_env_cls
from .envs.multibase import MultiBase, EnvConfig
from .viz import save_anim, save_img
from .utils import configure_logger

config.update("jax_default_matmul_precision", "float32")


@dataclass
class D4ormCfg:
    seed: int = 0
    Nsample: int = 2048  # number of samples
    Hsample: int = 100  # horizon
    Ndiffuse: int = 100  # number of denoising steps
    Niteration: int = 10  # number of iterations
    temp_sample: float = 0.3  # temperature for sampling
    beta1: float = 1e-4  # initial beta
    betaT: float = 2e-2  # final beta
    anytime: bool = False

    def __post_init__(self):
        # noise scheduling preparation
        Ndiffuse_ref = 100
        betas_ref = jnp.linspace(self.beta1, self.betaT, Ndiffuse_ref)
        betas_ref = jnp.concatenate([jnp.array([0.0]), betas_ref])
        alphas = 1.0 - betas_ref
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1 / alphas_bar - 1)

        # used during denoising procedure
        self.rng = jax.random.PRNGKey(seed=self.seed)
        self.alphas_bar = alphas_bar
        self.sigmas = sigmas


def d4orm(
    cfg: D4ormCfg,
    env: MultiBase,
    U_base: jax.Array | None = None,
):
    rollout_env_jit = partial(env.rollout)
    Nagent = env.n_agents
    Nu = env.action_size

    @jax.jit
    def reverse_once(carry, _):
        i, rng, U_i, U_base = carry

        # --- Step 1: compute Ubar_i
        Ubar_i = U_i / jnp.sqrt(cfg.alphas_bar[i])

        # --- Step 2: sample from q_i
        rng, rng_sample = jax.random.split(rng)
        eps_u = jax.random.normal(rng_sample, (cfg.Nsample, cfg.Hsample, Nu))
        U_is = eps_u * cfg.sigmas[i] + Ubar_i  # shape (Nsample, Hsample, Nu)

        # --- Step 3: rollout trajectories
        rollout_fn = jax.vmap(rollout_env_jit, in_axes=(0))
        rewss, _, _, _ = rollout_fn(us=U_is + U_base)

        # normalization
        rews = rewss.mean(axis=-1)
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / cfg.temp_sample
        weights = jax.nn.softmax(logp0)
        Ubar_0 = jnp.einsum("n,nij->ij", weights, U_is)

        U_im1 = jnp.sqrt(cfg.alphas_bar[i - 1]) * Ubar_0
        return (i - 1, rng, U_im1, U_base), rew_mean

    @jax.jit
    def denoise(rng, U_base):
        U_N = jnp.zeros([cfg.Hsample, Nu])
        (_, rng, U_deform, _), _ = jax.lax.scan(
            f=reverse_once,
            init=(cfg.Ndiffuse, rng, U_N, U_base),
            length=cfg.Ndiffuse,
        )
        return U_deform

    # main
    _, rng = jax.random.split(cfg.rng)
    if U_base is None:
        U_base = jnp.zeros([cfg.Hsample, Nu])

    success = False
    num_denoising = 0
    while (not success or cfg.anytime) and num_denoising < cfg.Niteration:
        num_denoising += 1
        U_base += denoise(rng, U_base)
        rews, states, goal_masks, collisions = rollout_env_jit(us=U_base)

        # mask out actions after reach and stop at the goal
        if env.use_mask:
            idx_first_1 = jnp.argmax(goal_masks, axis=0)
            cols = jnp.arange(goal_masks.shape[-1])
            goal_masks_action = goal_masks.at[idx_first_1, cols].set(0)
            goal_masks_action = jnp.repeat(
                goal_masks_action, repeats=Nu // Nagent, axis=-1
            )
            U_base = U_base * (1 - goal_masks_action)

        # compute metrics
        num_collisions = int(jnp.sum(collisions).item() / 2)
        goal_reach_rate = jnp.count_nonzero(goal_masks[-1]).item() / Nagent
        reward = rews.mean().item()
        success = int(goal_reach_rate == 1.0 and num_collisions == 0)
        logger.debug(
            f"{num_denoising=:3d}, {success=}, {reward=:.3f}, {num_collisions=}, {goal_reach_rate=:.3f}"
        )

    U_base = env.clip_actions(U_base)
    return U_base, dict(
        num_denoising=num_denoising,
        rollout=rollout_env_jit(us=U_base),
    )


@dataclass
class Args(EnvConfig, D4ormCfg):
    # env
    env_name: str = "multi2dholo"
    # result
    save_img: bool = True
    save_gif: bool = False
    logger: str = "DEBUG"


def main(args: Args):
    configure_logger(args.logger)

    # setup env
    env_cls = get_env_cls(args.env_name)
    env = from_dict(env_cls, asdict(args), config=Config(strict=False))

    # set d4orm parameters
    cfg = from_dict(D4ormCfg, asdict(args), config=Config(strict=False))

    ## run d4orm
    start_time = time.time()
    U_base, aux = d4orm(cfg=cfg, env=env)
    elapsed_time = time.time() - start_time
    rews, states, goal_masks, collisions = aux["rollout"]
    actions = U_base.reshape(U_base.shape[0], env.n_agents, -1)

    # compute metrics
    num_collisions = jnp.sum(collisions).item() / 2
    goal_reach_rate = jnp.count_nonzero(goal_masks[-1]).item() / env.n_agents
    avg_steps_to_goal = (jnp.sum(goal_masks == 0) // env.n_agents).item()
    max_steps_to_goal = (jnp.max(jnp.argmax(goal_masks, axis=0))).item()
    rew_final = rews[: cfg.Hsample].mean().item()

    # trimming
    states = states[:max_steps_to_goal]
    actions = actions[:max_steps_to_goal]

    # construct solution data
    sol = dict(
        success=(goal_reach_rate == 1.0 and num_collisions == 0),
        goal_reach_rate=goal_reach_rate,
        num_collisions=num_collisions,
        elapsed_time_sec=elapsed_time,
        num_denoising=aux["num_denoising"],
        reward_final=rew_final,
        average_steps_to_goal=avg_steps_to_goal,
        max_steps_to_goal=max_steps_to_goal,
    )
    logger.info("results:")
    for key, val in sol.items():
        logger.info(f"- {key:30s}: {val}")

    sol["dt"] = env.dt
    sol["instance"] = env.asdict()
    sol["result"] = []
    for i in range(env.n_agents):
        x_i = states[:, i].tolist()
        u_i = actions[:, i].tolist()
        sol["result"].append(dict(actions=u_i, states=x_i))

    # save results
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(__file__).parents[2] / f"outputs/{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"saved in {output_dir}")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(args), f)
    with open(output_dir / "result.yaml", "w") as f:
        yaml.safe_dump(sol, f)

    if args.save_img:
        save_img(env, states, output_dir / "trajectories.png")
    if args.save_gif:
        save_anim(env, states, output_dir / "trajectories.gif")


if __name__ == "__main__":
    main(args=tyro.cli(Args))
