import jax
from jax import numpy as jnp
from functools import partial
from dataclasses import dataclass

from .multibase import MultiBase
from .utils import generate_sphere_configuration


@dataclass(eq=False)
class Unicycle(MultiBase):
    obsv_dim_agent: int = 4
    pos_dim_agent: int = 2
    action_dim_agent: int = 2
    agent_radius: float = 0.25

    mav: float = jnp.pi / 2  # max angular velocity
    mlv: float = 1.0  # max linear velocity
    mla: float = 1.0  # max linear acceleration

    def __post_init__(self):
        super().__post_init__()

    def get_start_goal_configuration(self):
        x0, xg = generate_sphere_configuration(
            5.0,
            self.n_agents,
            self.obsv_dim_agent,
            self.pos_dim_agent,
            rng=self.rng,
        )
        directions = xg[:, : self.pos_dim_agent] - x0[:, : self.pos_dim_agent]
        angles = jnp.arctan2(directions[:, 1], directions[:, 0])
        x0 = x0.at[:, 2].set(angles)
        return x0, xg

    @partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x, u):
        """
        x[0]: position x
        x[1]: position y
        x[2]: heading angle theta
        x[3]: linear velocity v
        u[0]: angular velocity
        u[1]: linear acceleration

        Returns the time derivative of the state.
        """
        # Extract state variables
        x_pos, y_pos, theta, v = x
        angular_velocity = jnp.clip(u[0], -self.mav, self.mav)
        linear_acceleration = jnp.clip(u[1], -self.mla, self.mla)

        # Dynamics equations
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = angular_velocity
        v_dot = linear_acceleration

        return jnp.array([x_dot, y_dot, theta_dot, v_dot])

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor=1):
        traj = traj.reshape(-1, self.n_agents, self.action_dim_agent)
        traj = jnp.stack(
            [
                jnp.clip(traj[..., 0], -self.mav * factor, self.mav * factor),
                jnp.clip(traj[..., 1], -self.mla * factor, self.mla * factor),
            ],
            axis=-1,
        )
        return traj.reshape(-1, self.action_dim_agent * self.n_agents)

    def clip_velocity(self, x):
        v = jnp.clip(x[3], -self.mlv, self.mlv)
        return x.at[3].set(v)

    def get_current_velocity(self, q):
        return q[:, 3]

    @property
    def action_size(self):
        return (
            self.action_dim_agent * self.n_agents
        )  # Two actions (steering and acceleration) per agent

    def get_heading_line(self, state, position, agent_idx):
        theta = state[2]
        dx = self.agent_radius * jnp.cos(theta)
        dy = self.agent_radius * jnp.sin(theta)
        return [position[0], position[0] + dx], [position[1], position[1] + dy]

    @property
    def env_constraints_dict(self):
        return dict(
            max_angular_velocity=self.mav,
            max_linear_velocity=self.mlv,
            max_linear_acceleration=self.mla,
        )
