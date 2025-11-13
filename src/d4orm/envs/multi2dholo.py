import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass

from .multibase import MultiBase

from .utils import generate_sphere_configuraiton
import yaml
from pathlib import Path


@dataclass(eq=False)
class Multi2dHolo(MultiBase):
    obsv_dim_agent: int = 4
    pos_dim_agent: int = 2
    action_dim_agent: int = 2

    diameter: float = 5.0
    agent_radius: float = 0.25

    mv: float = 5.0  # max velocity
    ma: float = 2.0  # max acceleration

    def __post_init__(self):
        super().__post_init__()

    def get_start_goal_configuration(self):
        return generate_sphere_configuraiton(
            self.diameter,
            self.n_agents,
            self.obsv_dim_agent,
            self.pos_dim_agent,
            rng=self.rng,
        )

    @partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x, u):
        """
        x[0]: position x
        x[1]: position y
        x[2]: velocity in x (vx)
        x[3]: velocity in y (vy)
        u[0]: acceleration in x (ax)
        u[1]: acceleration in y (ay)

        Returns the time derivative of the state.
        """
        x_pos, y_pos, vx, vy = x

        acc_norm = jnp.linalg.norm(u)
        scale_acc = jnp.minimum(1.0, self.ma / acc_norm)
        u = u * scale_acc

        x_dot = vx
        y_dot = vy
        vx_dot = u[0]
        vy_dot = u[1]

        return jnp.array([x_dot, y_dot, vx_dot, vy_dot])

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor=1):
        traj = traj.reshape(-1, self.n_agents, self.action_dim_agent)
        norm = jnp.linalg.norm(traj, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.ma * factor / norm)
        traj = traj * scale
        return traj.reshape(-1, self.action_dim_agent * self.n_agents)

    def clip_velocity(self, x):
        vx, vy = x[2], x[3]
        vel_norm = jnp.linalg.norm(jnp.array([vx, vy]))
        scale_vel = jnp.minimum(1.0, self.mv / vel_norm)
        x = x.at[2].set(vx * scale_vel)
        x = x.at[3].set(vy * scale_vel)
        return x

    def get_current_velocity(self, q):
        return jnp.linalg.norm(q[:, [2, 3]], axis=-1)

    @property
    def action_size(self):
        return self.action_dim_agent * self.n_agents

    @property
    def env_constraints_dict(self):
        return dict(
            velocity_bounds=dict(
                min=[-self.mv] * self.pos_dim_agent,
                max=[self.mv] * self.pos_dim_agent,
            ),
            action_bounds=dict(
                min=[-self.ma] * self.action_dim_agent,
                max=[self.ma] * self.action_dim_agent,
            ),
        )


@dataclass(eq=False)
class Multi2dHoloCustom(Multi2dHolo):
    def __post_init__(self):
        super().__post_init__()
        self.max_distances = jnp.linalg.norm(self.x0 - self.xg, axis=1)

    def get_start_goal_configuration(self):
        with open(Path(self.external_file)) as f:
            external_cfg = yaml.safe_load(f)

        starts, goals = [], []
        for s_g in external_cfg["problem"]["terminals"][: self.n_agents]:
            starts.append(s_g["start"])
            goals.append(s_g["goal"])

        return jnp.array(starts), jnp.array(goals)
