import jax
from jax import numpy as jnp
from functools import partial

from .multibase import MultiBase


class Multi2dHolo(MultiBase):
    def __init__(self, num_agents):
        super().__init__(num_agents)
        self.num_agents = num_agents
        self.action_dim_agent = 2
        self.obsv_dim_agent = 4
        self.pos_dim_agent = 2
        self.diameter = 5
        self.safe_margin = 0.05
        self.agent_radius = 0.25
        self.stop_distance = (
            self.agent_radius / 2
        )  # max distance to goal for termination
        self.stop_velocity = 0.1  # max velocity for termination when reach the goal
        self.mv = 5.0  # max velocity
        self.ma = 2.0  # max acceleration

        initial_states, goal_states = self.generate_positions(self.diameter, num_agents)
        self.lim = self.diameter / 2 + 1

        self.x0 = initial_states.flatten()
        self.xg = goal_states.flatten()
        self.max_distance = self.diameter

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
