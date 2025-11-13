import jax.numpy as jnp

from d4orm.d4orm import D4ormCfg, d4orm
from d4orm.envs.multi2dholo import Multi2dHolo


def test_D4orm():
    env = Multi2dHolo()
    cfg = D4ormCfg()
    U_base, aux = d4orm(cfg=cfg, env=env)
    rews, states, goal_masks, collisions = aux["rollout"]
    num_collisions = jnp.sum(collisions).item() / 2
    goal_reach_rate = jnp.count_nonzero(goal_masks[-1]).item() / env.n_agents
    avg_steps_to_goal = (jnp.sum(goal_masks == 0) // env.n_agents).item()
    max_steps_to_goal = (jnp.max(jnp.argmax(goal_masks, axis=0))).item()
    rew_final = rews[: cfg.Hsample].mean().item()

    print(
        f"{num_collisions=}\t",
        f"{goal_reach_rate=}\t",
        f"{avg_steps_to_goal=}\t",
        f"{max_steps_to_goal=}\t",
        f"{rew_final=:.3f}",
    )
