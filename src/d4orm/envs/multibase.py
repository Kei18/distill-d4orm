import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    reward: jnp.ndarray
    mask: jnp.ndarray
    collision: jnp.ndarray


class MultiBase:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.obsv_dim_agent = 1000
        self.pos_dim_agent = 1000
        self.diameter = 1000
        self.safe_margin = 1000
        self.agent_radius = 1000
        self.stop_distance = self.agent_radius / 2
        self.stop_velocity = 1000
        self.offset = 1

        initial_states, goal_states = self.generate_positions(self.diameter, num_agents)
        self.lim = self.diameter / 2 + 1

        self.x0 = initial_states.flatten()
        self.xg = goal_states.flatten()
        self.max_distance = self.diameter

    def generate_positions(self, diameter, num_agents):
        radius = diameter / 2.0
        angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
        position_components = [
            radius * jnp.cos(angles),  # x
            radius * jnp.sin(angles),  # y
        ]
        zero_components = [
            jnp.zeros_like(angles)
            for _ in range(self.obsv_dim_agent - self.pos_dim_agent)
        ]
        initial_states = jnp.stack(position_components + zero_components, axis=-1)
        goal_states = -initial_states
        return initial_states, goal_states

    def reset(self, rng: jax.Array):
        return State(
            pipeline_state=self.x0,
            reward=jnp.zeros(self.num_agents, dtype=jnp.float32),
            mask=jnp.zeros(self.num_agents, dtype=jnp.float32),
            collision=jnp.zeros(self.num_agents, dtype=jnp.float32),
        )

    def reset_conditioned(self, x0: jax.Array, rng: jax.Array):
        return State(
            pipeline_state=x0,
            reward=jnp.zeros(self.num_agents, dtype=jnp.float32),
            mask=jnp.zeros(self.num_agents, dtype=jnp.float32),
            collision=jnp.zeros(self.num_agents, dtype=jnp.float32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj, factor=1):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x, u):
        raise NotImplementedError

    def clip_velocity(self, x):
        """x is state for single robot"""
        raise NotImplementedError

    def get_current_velocity(self, q):
        """q is joint state for all robots"""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def euler(self, x, u, dt):
        x_next = x + dt * self.agent_dynamics(x, u)
        return self.clip_velocity(x_next)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(
        self,
        state: State,
        xg: jax.Array,
        us: jax.Array,
        penalty_weight: float = 1.0,
        use_mask: bool = True,
        margin_factor: int = 1,
        dt: float = 0.1,
    ):
        init_pos = state.pipeline_state.reshape(self.num_agents, -1)
        goal_pos = xg.reshape(self.num_agents, -1)
        max_distances = jnp.linalg.norm(init_pos - goal_pos, axis=1)

        def step_wrapper(state: State, u: jax.Array):
            state = self.step(
                state, xg, u, max_distances, penalty_weight, use_mask, margin_factor, dt
            )
            return state, (
                state.reward,
                state.pipeline_state,
                state.mask,
                state.collision,
            )

        _, (rews, pipline_states, masks, collisions) = jax.lax.scan(
            step_wrapper, state, us
        )

        rews = rews.mean(axis=0)

        return rews, pipline_states, masks, collisions

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: State,
        xg: jax.Array,
        action: jax.Array,
        max_distances: jax.Array,
        penalty_weight: float = 1.0,
        use_mask: bool = True,
        margin_factor: int = 1,
        dt: float = 0.1,
    ) -> State:
        """Step Once"""
        q = state.pipeline_state.reshape(self.num_agents, -1)
        actions = action.reshape(self.num_agents, -1)
        goals = xg.reshape(self.num_agents, -1)

        # Get new q
        q_new = jax.vmap(
            lambda agent_state, agent_action: self.euler(agent_state, agent_action, dt)
        )(q, actions)

        # Don't update for stopped state
        previously_stopped_mask = jnp.broadcast_to(
            state.mask, (self.num_agents,)
        ).astype(bool)
        q_new = jnp.where(
            use_mask, jnp.where(previously_stopped_mask[:, None], q, q_new), q_new
        )

        dist_to_goals = jax.vmap(
            lambda agent_position, goal_position: jnp.linalg.norm(
                agent_position[: self.pos_dim_agent]
                - goal_position[: self.pos_dim_agent]
            )
        )(q_new, goals)

        curr_vel = self.get_current_velocity(q)
        stop_update_mask = (dist_to_goals < self.stop_distance) & (
            curr_vel <= self.stop_velocity
        )
        previously_stopped_mask = jnp.broadcast_to(
            state.mask, (self.num_agents,)
        ).astype(bool)
        combined_stop_mask = stop_update_mask | previously_stopped_mask

        agent_wise_reward, collision = self.get_reward(
            q=q_new,
            distances_to_goals=dist_to_goals,
            max_distances=max_distances,
            penalty_weight=penalty_weight,
            margin_factor=margin_factor,
        )

        mask = combined_stop_mask.astype(float)
        collision = collision.astype(float)

        return state.replace(
            pipeline_state=q_new.flatten(),
            reward=agent_wise_reward,
            mask=mask,
            collision=collision,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(
        self,
        q: jax.Array,
        distances_to_goals: jax.Array,
        max_distances: jax.Array,
        penalty_weight: float = 1.0,
        margin_factor: int = 1,
    ) -> float:
        agent_positions = q[:, : self.pos_dim_agent]

        # Calculate rewards using distance
        rewards = 1.0 - distances_to_goals / max_distances

        # Compute pairwise penalties
        pairwise_differences = agent_positions[:, None, :] - agent_positions[None, :, :]
        pairwise_distances = jnp.linalg.norm(pairwise_differences, axis=-1)
        mask = ~jnp.eye(self.num_agents, dtype=bool)  # Mask for non-diagonal elements
        valid_distances = jnp.where(mask, pairwise_distances, jnp.inf)
        agent_collision_threshold = (
            2 * self.agent_radius + self.safe_margin * margin_factor
        )

        penalties_agent = jnp.where(
            valid_distances <= agent_collision_threshold, 1.0, 0.0
        )

        collision = jnp.any(penalties_agent != 0.0, axis=1)

        # Compute agent-wise reward
        total_agent_penalty = penalties_agent.sum(axis=1) * penalty_weight
        rewards = rewards - total_agent_penalty

        # Calculate total reward
        return rewards, collision

    @property
    def action_size(self):
        raise NotImplementedError

    @property
    def observation_size(self):
        return self.obsv_dim_agent * self.num_agents

    def get_heading_line(self, state, position, agent_idx):
        return [], []

    def get_color(self, i, colormaps):
        return cm.get_cmap(colormaps[i % len(colormaps)])

    def render_gif(
        self, xs: jnp.ndarray, gif_output_path, trajectory_image_path, ids=None
    ):
        # Reshape trajectory for rendering
        xs = xs.reshape(-1, self.num_agents, self.obsv_dim_agent)

        # --- Initialize GIF Rendering ---
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set(xlim=(-self.lim, self.lim), ylim=(-self.lim, self.lim), aspect="equal")

        colormaps = ["Reds", "Greens", "Purples", "Oranges", "Blues"]
        circles, headings = [], []

        for i in range(self.num_agents):
            cmap = self.get_color(i, colormaps)
            color = cmap(0.6)

            circle = Circle((0, 0), radius=self.agent_radius, facecolor=color)
            ax.add_patch(circle)
            circles.append(circle)

            (heading,) = ax.plot([], [], color="black", lw=1.5)
            headings.append(heading)

        def update(frame):
            for i, (circle, heading) in enumerate(zip(circles, headings)):
                state = xs[frame * self.offset, i]
                position = state[: self.pos_dim_agent]
                circle.set_center(position)
                x_line, y_line = self.get_heading_line(state, position, i)
                heading.set_data(x_line, y_line)
            return circles + headings

        anim = FuncAnimation(
            fig,
            update,
            frames=xs.shape[0] // self.offset + 1,
            blit=True,
            interval=100,
        )

        if gif_output_path != "None":
            anim.save(gif_output_path, writer=PillowWriter(fps=10 // self.offset))
        plt.close(fig)

        # --- Generate Static Trajectory Image ---
        xs = xs[:: self.offset]
        fig_traj, ax_traj = plt.subplots()
        ax_traj.set(
            xlim=(-self.lim, self.lim), ylim=(-self.lim, self.lim), aspect="equal"
        )
        ax_traj.scatter([], [], color="k", alpha=0.5, label="Obstacle", s=200)

        num_colormaps = len(colormaps)
        for i in range(self.num_agents):
            cmap_index = i % num_colormaps if ids is None else int(ids[i]) + 1
            cmap = cm.get_cmap(colormaps[cmap_index])
            color = cmap(0.6)

            traj_x, traj_y = xs[:, i, 0], xs[:, i, 1]
            ax_traj.plot(
                traj_x, traj_y, color=color, linestyle="--", linewidth=1, alpha=0.5
            )

            start_circle = Circle(
                (traj_x[0], traj_y[0]), self.agent_radius, color=color, zorder=5
            )
            ax_traj.add_artist(start_circle)

        # --- Collision Detection ---
        collision_positions = []
        for t in range(xs.shape[0]):
            positions = xs[t, :, : self.pos_dim_agent]
            diffs = positions[:, None, :] - positions[None, :, :]
            dists = jnp.linalg.norm(diffs, axis=-1)
            collision_matrix = (dists < self.agent_radius * 2 + self.safe_margin) & (
                dists > 0
            )
            for i in range(self.num_agents):
                if jnp.any(collision_matrix[i]):
                    collision_positions.append(positions[i])

        for pos in collision_positions:
            ax_traj.plot(pos[0], pos[1], "rx", markersize=10, markeredgewidth=1)

        # --- Plot Goal Positions ---
        xg_reshaped = self.xg.reshape(self.num_agents, -1)
        goal_x, goal_y = xg_reshaped[:, 0], xg_reshaped[:, 1]
        ax_traj.plot(
            goal_x,
            goal_y,
            "+",
            color="k",
            alpha=0.5,
            markersize=10,
            markeredgewidth=1,
            zorder=10,
        )

        fig_traj.savefig(trajectory_image_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_traj)

    def save_traj(self, Y, filename):
        raise NotImplementedError

    def asdict(self) -> dict:
        ret = dict(
            environment=dict(
                workspace_min=[-10] * self.pos_dim_agent,
                workspace_max=[10] * self.pos_dim_agent,
                dynamics_type=self.__class__.__name__,
                velocity_bounds=dict(
                    min=[-self.mv] * self.pos_dim_agent,
                    max=[self.mv] * self.pos_dim_agent,
                ),
                action_bounds=dict(
                    min=[-self.ma] * self.action_dim_agent,
                    max=[self.ma] * self.action_dim_agent,
                ),
                robot_radius=self.agent_radius,
            ),
            problem=dict(
                n_obstacles=0,
                n_agents=self.num_agents,
                terminals=[
                    dict(
                        start=self.x0[
                            i * self.obsv_dim_agent : i + 1 * self.obsv_dim_agent
                        ].tolist(),
                        goal=self.xg[
                            i * self.obsv_dim_agent : i + 1 * self.obsv_dim_agent
                        ].tolist(),
                    )
                    for i in range(self.num_agents)
                ],
            ),
        )
        return ret
