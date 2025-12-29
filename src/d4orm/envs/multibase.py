from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    reward: jnp.ndarray
    mask: jnp.ndarray
    collision: jnp.ndarray


@dataclass(eq=False)
class EnvConfig:
    seed: int = 0
    n_agents: int = 2  # number of agents
    dt: float = 0.1
    stop_distance: float = 0.1
    stop_velocity: float = 0.5  # max velocity for termination when reach the goal
    use_mask: bool = True  # masking
    penalty_weight_collision: float = 2.0  # collision penalty
    safe_margin: float = 0.02
    external_file: str = ""


@dataclass(eq=False)
class MultiBase(EnvConfig, ABC):
    obsv_dim_agent: int = 1000
    pos_dim_agent: int = 1000
    agent_radius: float = 1000

    def __post_init__(self):
        self.rng = jax.random.PRNGKey(seed=self.seed)
        self.x0, self.xg = self.get_start_goal_configuration()
        self.max_distances = jnp.linalg.norm(self.x0 - self.xg, axis=1)

    @abstractmethod
    def get_start_goal_configuration(self):
        raise NotImplementedError

    @property
    def state_init(self):
        return State(
            pipeline_state=self.x0,
            reward=jnp.zeros(self.n_agents, dtype=jnp.float32),
            mask=jnp.zeros(self.n_agents, dtype=jnp.float32),
            collision=jnp.zeros(self.n_agents, dtype=jnp.float32),
        )

    @partial(jax.jit, static_argnums=(0,))
    @abstractmethod
    def clip_actions(self, traj, factor=1):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    @abstractmethod
    def agent_dynamics(self, x, u):
        raise NotImplementedError

    @abstractmethod
    def clip_velocity(self, x):
        """x is state for single robot"""
        raise NotImplementedError

    @abstractmethod
    def get_current_velocity(self, q):
        """q is joint state for all robots"""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def euler(self, x, u):
        x_next = x + self.dt * self.agent_dynamics(x, u)
        return self.clip_velocity(x_next)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, us: jax.Array):
        def step_wrapper(state: State, u: jax.Array):
            state = self.step(state, u)
            return state, (
                state.reward,
                state.pipeline_state,
                state.mask,
                state.collision,
            )

        _, (rews, pipeline_states, masks, collisions) = jax.lax.scan(
            step_wrapper, self.state_init, us
        )

        rews = rews.mean(axis=0)

        return rews, pipeline_states, masks, collisions

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: State,
        action: jax.Array,
    ) -> State:
        """Step Once"""
        q = state.pipeline_state
        actions = action.reshape(self.n_agents, -1)

        # Get new q
        q_new = jax.vmap(
            lambda agent_state, agent_action: self.euler(agent_state, agent_action)
        )(q, actions)

        # Don't update for stopped state
        previously_stopped_mask = jnp.broadcast_to(state.mask, (self.n_agents,)).astype(
            bool
        )
        q_new = jnp.where(
            self.use_mask, jnp.where(previously_stopped_mask[:, None], q, q_new), q_new
        )

        dist_to_goals = jax.vmap(
            lambda agent_position, goal_position: jnp.linalg.norm(
                agent_position[: self.pos_dim_agent]
                - goal_position[: self.pos_dim_agent]
            )
        )(q_new, self.xg)

        curr_vel = self.get_current_velocity(q)
        stop_update_mask = (dist_to_goals < self.stop_distance) & (
            curr_vel <= self.stop_velocity
        )
        previously_stopped_mask = jnp.broadcast_to(state.mask, (self.n_agents,)).astype(
            bool
        )
        combined_stop_mask = stop_update_mask | previously_stopped_mask

        agent_wise_reward, collision = self.get_reward(
            q=q_new,
            distances_to_goals=dist_to_goals,
        )

        mask = combined_stop_mask.astype(float)
        collision = collision.astype(float)

        return state.replace(
            pipeline_state=q_new,
            reward=agent_wise_reward,
            mask=mask,
            collision=collision,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(
        self,
        q: jax.Array,
        distances_to_goals: jax.Array,
    ) -> float:
        agent_positions = q[:, : self.pos_dim_agent]

        # Calculate rewards using distance
        rewards = 1.0 - distances_to_goals / (self.max_distances + 1e-5)

        # Compute pairwise penalties
        pairwise_differences = agent_positions[:, None, :] - agent_positions[None, :, :]
        pairwise_distances = jnp.linalg.norm(pairwise_differences, axis=-1)
        mask = ~jnp.eye(self.n_agents, dtype=bool)  # Mask for non-diagonal elements
        valid_distances = jnp.where(mask, pairwise_distances, jnp.inf)

        hard_th = 2 * self.agent_radius
        soft_th = hard_th + self.safe_margin

        hard_col_agent = jnp.where(valid_distances <= hard_th, 1.0, 0.0)
        soft_col_agent = jnp.where(valid_distances <= soft_th, 1.0, 0.0)
        collision = jnp.any(hard_col_agent != 0.0, axis=1)

        # Compute agent-wise reward
        total_agent_penalty = soft_col_agent.sum(axis=1) * self.penalty_weight_collision

        rewards = rewards - total_agent_penalty

        # Calculate total reward
        return rewards, collision

    @property
    @property
    @abstractmethod
    def action_size(self):
        raise NotImplementedError

    @property
    def observation_size(self):
        return self.obsv_dim_agent * self.n_agents

    def get_heading_line(self, state, position, agent_idx):
        return [], []

    @property
    @property
    @abstractmethod
    def env_constraints_dict(self):
        raise NotImplementedError

    def asdict(self) -> dict:
        ret = dict(
            environment=dict(
                workspace_min=[-10] * self.pos_dim_agent,
                workspace_max=[10] * self.pos_dim_agent,
                dynamics_type=self.__class__.__name__,
                **self.env_constraints_dict,
                robot_radius=self.agent_radius,
            ),
            problem=dict(
                n_obstacles=0,
                n_agents=self.n_agents,
                terminals=[
                    dict(
                        start=self.x0[i].tolist(),
                        goal=self.xg[i].tolist(),
                    )
                    for i in range(self.n_agents)
                ],
            ),
        )
        return ret
