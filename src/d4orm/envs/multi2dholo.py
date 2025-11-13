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
    agent_radius: float = 0.25

    mv: float = 5.0  # max velocity
    ma: float = 2.0  # max acceleration

    def __post_init__(self):
        super().__post_init__()

    def get_start_goal_configuration(self):
        return generate_sphere_configuraiton(
            5.0,
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
class Multi2dHoloRandom(Multi2dHolo):
    workspace_size: float = 3.0

    def get_start_goal_configuration(self):
        _, rng = jax.random.split(self.rng)
        th = self.agent_radius * 2 + self.safe_margin
        starts, goals = [], []
        params = dict(
            shape=(2, self.pos_dim_agent),
            minval=-self.workspace_size,
            maxval=self.workspace_size,
        )
        dim_zero_states = self.obsv_dim_agent - self.pos_dim_agent

        while len(starts) < self.n_agents:
            _, rng = jax.random.split(rng)
            s, g = jax.random.uniform(rng, **params)
            if len(starts) > 0:
                d_s = jnp.linalg.norm(
                    jnp.array(starts)[:, : self.pos_dim_agent] - s, axis=1
                )
                d_g = jnp.linalg.norm(
                    jnp.array(goals)[:, : self.pos_dim_agent] - g, axis=1
                )
                if any(d_s < th) or any(d_g < th):
                    continue
            starts.append(s.tolist() + [0] * dim_zero_states)
            goals.append(g.tolist() + [0] * dim_zero_states)

        return jnp.array(starts), jnp.array(goals)


@dataclass(eq=False)
class Multi2dHoloCustom(Multi2dHolo):
    def __post_init__(self):
        with open(Path(self.external_file)) as f:
            self.external_cfg = yaml.safe_load(f)
        super().__post_init__()
        self.max_distances = jnp.linalg.norm(self.x0 - self.xg, axis=1)

        # set obstacles
        obstacles = []
        obstacles_rad = []
        for o in self.external_cfg["problem"].get("obstacles", []):
            obstacles.append(o["center"])
            obstacles_rad.append(max(o["size"]))
        self.obs_center = jnp.array(obstacles)
        self.obs_rad = jnp.array(obstacles_rad)

        # reward
        self.penalty_weight_obs = self.penalty_weight_collision * 2

    def get_start_goal_configuration(self):
        starts, goals = [], []
        for s_g in self.external_cfg["problem"]["terminals"][: self.n_agents]:
            starts.append(s_g["start"])
            goals.append(s_g["goal"])

        return jnp.array(starts), jnp.array(goals)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(
        self,
        q: jax.Array,
        distances_to_goals: jax.Array,
    ) -> float:
        rewards, collision = super().get_reward(q, distances_to_goals)

        hard_th = self.agent_radius + self.obs_rad
        soft_th = hard_th + self.safe_margin

        agent_positions = q[:, : self.pos_dim_agent]
        agent_obs_differences = (
            agent_positions[:, None, :] - self.obs_center[None, :, :]
        )
        agent_obs_distances = jnp.linalg.norm(
            agent_obs_differences, axis=-1
        )  # Shape (Nagent, Nobs)

        hard_col_obs = jnp.where(agent_obs_distances <= hard_th, 1.0, 0.0)
        soft_col_obs = jnp.where(agent_obs_distances <= soft_th, 1.0, 0.0)

        collision = jnp.logical_or(collision, jnp.any(hard_col_obs != 0.0, axis=1))
        rewards -= soft_col_obs.sum(axis=1) * self.penalty_weight_obs
        return rewards, collision
