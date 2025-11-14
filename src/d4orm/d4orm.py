from dataclasses import dataclass
from functools import partial

from loguru import logger

import jax
from jax import numpy as jnp, config

from .envs.multibase import MultiBase

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


# @jax.disable_jit()  # useful for debugging
def d4orm_opt(
    cfg: D4ormCfg,
    env: MultiBase,
    U_base: jax.Array | None = None,
):
    rollout_env_jit = partial(env.rollout)
    rollout_fn = jax.vmap(rollout_env_jit, in_axes=(0))
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


def get_metrics(aux):
    rews, states, goal_masks, collisions = aux["rollout"]
    n_agents = states.shape[1]
    return dict(
        num_collisions=jnp.sum(collisions).item() / 2,
        goal_reach_rate=jnp.count_nonzero(goal_masks[-1]).item() / n_agents,
        avg_steps_to_goal=(jnp.sum(goal_masks == 0) // n_agents).item(),
        max_steps_to_goal=(jnp.max(jnp.argmax(goal_masks, axis=0))).item(),
        reward_final=rews.mean().item(),
    )
