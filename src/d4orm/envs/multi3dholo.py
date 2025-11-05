import jax
from jax import numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .multibase import MultiBase


class Multi3dHolo(MultiBase):
    def __init__(self, num_agents):
        super().__init__(num_agents)
        self.num_agents = num_agents
        self.action_dim_agent = 3
        self.obsv_dim_agent = 6
        self.pos_dim_agent = 3
        self.diameter = 5
        self.agent_radius = 0.15
        self.safe_margin = self.agent_radius * 2 + 0.05
        self.stop_distance = (
            self.agent_radius / 2
        )  # max distance to goal for termination
        self.stop_velocity = float(
            "inf"
        )  # max velocity for termination when reach the goal
        self.mv = 1.0  # max velocity
        self.ma = 1.0  # max acceleration

        initial_states, goal_states = self.generate_positions(self.diameter, num_agents)
        self.lim = self.diameter / 2 + 1

        self.x0 = initial_states.flatten()
        self.xg = goal_states.flatten()
        self.max_distance = self.diameter

    def generate_positions(self, diameter, num_agents):
        radius = diameter / 2.0

        indices = jnp.arange(0, num_agents)
        phi = jnp.arccos(1 - 2 * (indices + 0.5) / num_agents)
        theta = jnp.pi * (1 + 5**0.5) * indices

        x = radius * jnp.sin(phi) * jnp.cos(theta)
        y = radius * jnp.sin(phi) * jnp.sin(theta)
        z = radius * jnp.cos(phi)

        initial_states = jnp.stack(
            [
                x,
                y,
                z,
                jnp.zeros(num_agents),
                jnp.zeros(num_agents),
                jnp.zeros(num_agents),
            ],
            axis=-1,
        )
        goal_states = -initial_states

        return initial_states, goal_states

    def generate_positions_circle(self, diameter, num_agents):
        radius = diameter / 2.0
        angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)

        initial_states = jnp.stack(
            [
                radius * jnp.cos(angles),
                radius * jnp.sin(angles),
                jnp.zeros(num_agents),
                jnp.zeros(num_agents),
                jnp.zeros(num_agents),
                jnp.zeros(num_agents),
            ],
            axis=-1,
        )

        goal_states = -initial_states

        return initial_states, goal_states

    @partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x, u):
        """
        x[0]: position x
        x[1]: position y
        x[2]: position z
        x[3]: velocity in x (vx)
        x[4]: velocity in y (vy)
        x[5]: velocity in z (vz)
        u[0]: acceleration in x (ax)
        u[1]: acceleration in y (ay)
        u[2]: acceleration in z (az)

        Returns the time derivative of the state.
        """
        x_pos, y_pos, z_pos, vx, vy, vz = x

        max_acceleration = self.ma
        acc_norm = jnp.linalg.norm(u)
        scale_acc = jnp.minimum(1.0, max_acceleration / acc_norm)
        u = u * scale_acc

        x_dot = vx
        y_dot = vy
        z_dot = vz
        vx_dot = u[0]
        vy_dot = u[1]
        vz_dot = u[2]

        return jnp.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor=1):
        traj = traj.reshape(-1, self.num_agents, self.action_dim_agent)
        norm = jnp.linalg.norm(traj, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.ma * factor / norm)
        traj = traj * scale
        return traj.reshape(-1, self.action_dim_agent * self.num_agents)

    def clip_velocity(self, x):
        vx, vy, vz = x[3], x[4], x[5]
        vel_norm = jnp.linalg.norm(jnp.array([vx, vy, vz]))
        scale_vel = jnp.minimum(1.0, self.mv / vel_norm)
        x = x.at[3].set(vx * scale_vel)
        x = x.at[4].set(vy * scale_vel)
        x = x.at[5].set(vz * scale_vel)
        return x

    def get_current_velocity(self, q):
        return jnp.linalg.norm(q[:, [3, 4, 5]], axis=-1)

    @property
    def action_size(self):
        return self.action_dim_agent * self.num_agents

    def render_gif(self, xs: jnp.ndarray, output_path, trajectory_image_path):
        xs = xs.reshape(-1, self.num_agents, self.obsv_dim_agent)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.clear()
            ax.set_xlim(-self.lim, self.lim)
            ax.set_ylim(-self.lim, self.lim)
            ax.set_zlim(-self.lim, self.lim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            for i in range(self.num_agents):
                pos = xs[: frame + 1, i, : self.pos_dim_agent]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"Agent {i + 1}")
                ax.scatter(
                    pos[-1, 0],
                    pos[-1, 1],
                    pos[-1, 2],
                    s=(self.agent_radius * 100) ** 2,
                    label=f"Current {i + 1}",
                )

        anim = FuncAnimation(fig, update, frames=xs.shape[0], interval=100)
        anim.save(output_path, writer=PillowWriter(fps=10))
        plt.close(fig)

    def render_gif_interactive(self, xs: jnp.ndarray):
        """Interactive visualization of agent motion in 3D space."""
        xs = xs.reshape(-1, self.num_agents, self.obsv_dim_agent)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-self.lim, self.lim)
        ax.set_ylim(-self.lim, self.lim)
        ax.set_zlim(-self.lim, self.lim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        def update(frame):
            ax.clear()
            ax.set_xlim(-self.lim, self.lim)
            ax.set_ylim(-self.lim, self.lim)
            ax.set_zlim(-self.lim, self.lim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            for i in range(self.num_agents):
                pos = xs[: frame + 1, i, : self.pos_dim_agent]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"Agent {i + 1}")
                ax.scatter(
                    pos[-1, 0],
                    pos[-1, 1],
                    pos[-1, 2],
                    s=(self.agent_radius * 100) ** 2,
                    label=f"Current {i + 1}",
                )

        FuncAnimation(fig, update, frames=xs.shape[0], interval=100, repeat=True)
        plt.show()
