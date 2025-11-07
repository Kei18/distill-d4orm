import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from pathlib import Path

from .envs.multibase import MultiBase

COLORMAPS = ["Reds", "Greens", "Purples", "Oranges", "Blues"]


def get_color(i):
    return cm.get_cmap(COLORMAPS[i % len(COLORMAPS)])(0.6)


def save_anim(
    env: MultiBase, xs: jnp.ndarray, output_path: Path, ids=None, offset: int = 1
):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set(xlim=(-env.lim, env.lim), ylim=(-env.lim, env.lim), aspect="equal")

    circles, headings = [], []

    for i in range(env.num_agents):
        color = get_color(i)
        circle = Circle((0, 0), radius=env.agent_radius, facecolor=color)
        ax.add_patch(circle)
        circles.append(circle)

        (heading,) = ax.plot([], [], color="black", lw=1.5)
        headings.append(heading)

    def update(frame):
        for i, (circle, heading) in enumerate(zip(circles, headings)):
            state = xs[frame * offset, i]
            position = state[: env.pos_dim_agent]
            circle.set_center(position)
            x_line, y_line = env.get_heading_line(state, position, i)
            heading.set_data(x_line, y_line)
        return circles + headings

    anim = FuncAnimation(
        fig,
        update,
        frames=xs.shape[0] // offset + 1,
        blit=True,
        interval=100,
    )
    anim.save(output_path, writer=PillowWriter(fps=10 // offset))
    plt.close(fig)


def save_img(
    env: MultiBase,
    xs: jnp.ndarray,
    output_path: Path,
    ids=None,
    offset: int = 1,
):
    # --- Generate Static Trajectory Image ---
    xs = xs[::offset]
    fig, ax = plt.subplots()
    ax.set(xlim=(-env.lim, env.lim), ylim=(-env.lim, env.lim), aspect="equal")
    ax.scatter([], [], color="k", alpha=0.5, label="Obstacle", s=200)

    for i in range(env.num_agents):
        color = get_color(i)

        traj_x, traj_y = xs[:, i, 0], xs[:, i, 1]
        ax.plot(traj_x, traj_y, color=color, linestyle="--", linewidth=1, alpha=0.5)

        start_circle = Circle(
            (traj_x[0], traj_y[0]), env.agent_radius, color=color, zorder=5
        )
        ax.add_artist(start_circle)

    # --- Collision Detection ---
    collision_positions = []
    for t in range(xs.shape[0]):
        positions = xs[t, :, : env.pos_dim_agent]
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = jnp.linalg.norm(diffs, axis=-1)
        collision_matrix = (dists < env.agent_radius * 2 + env.safe_margin) & (
            dists > 0
        )
        for i in range(env.num_agents):
            if jnp.any(collision_matrix[i]):
                collision_positions.append(positions[i])

    for pos in collision_positions:
        ax.plot(pos[0], pos[1], "rx", markersize=10, markeredgewidth=1)

    # --- Plot Goal Positions ---
    xg_reshaped = env.xg.reshape(env.num_agents, -1)
    goal_x, goal_y = xg_reshaped[:, 0], xg_reshaped[:, 1]
    ax.plot(
        goal_x,
        goal_y,
        "+",
        color="k",
        alpha=0.5,
        markersize=10,
        markeredgewidth=1,
        zorder=10,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
