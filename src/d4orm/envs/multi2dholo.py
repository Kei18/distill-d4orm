import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass

from .multibase import MultiBase


@dataclass(eq=False)
class Multi2dHolo(MultiBase):
    obsv_dim_agent: int = 4
    pos_dim_agent: int = 2
    action_dim_agent: int = 2

    diameter: float = 5
    safe_margin: float = 0.05
    agent_radius: float = 0.25
    stop_velocity: float = 0.1  # max velocity for termination when reach the goal
    mv: float = 5.0  # max velocity
    ma: float = 2.0  # max acceleration

    def __post_init__(self):
        super().__post_init__()

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

        max_acceleration = self.ma
        acc_norm = jnp.linalg.norm(u)
        scale_acc = jnp.minimum(1.0, max_acceleration / acc_norm)
        u = u * scale_acc

        x_dot = vx
        y_dot = vy
        vx_dot = u[0]
        vy_dot = u[1]

        return jnp.array([x_dot, y_dot, vx_dot, vy_dot])

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor=1):
        traj = traj.reshape(-1, self.num_agents, self.action_dim_agent)
        norm = jnp.linalg.norm(traj, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.ma * factor / norm)
        traj = traj * scale
        return traj.reshape(-1, self.action_dim_agent * self.num_agents)

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
        return self.action_dim_agent * self.num_agents
