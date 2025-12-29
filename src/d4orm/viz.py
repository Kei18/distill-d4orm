import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from pathlib import Path

from .envs.multibase import MultiBase
from .envs.multi2dholo import Multi2dHoloCustom

COLORMAPS = ["Reds", "Greens", "Purples", "Oranges", "Blues"]


def get_color(i):
    return cm.get_cmap(COLORMAPS[i % len(COLORMAPS)])(0.6)


def set_ax_lim(ax, env: MultiBase, xs: jnp.ndarray) -> None:
    xmin = min(xs[:, :, 0].min(), env.xg[:, 0].min()) - 1
    xmax = max(xs[:, :, 0].max(), env.xg[:, 0].max()) + 1
    ymin = min(xs[:, :, 1].min(), env.xg[:, 1].min()) - 1
    ymax = max(xs[:, :, 1].max(), env.xg[:, 1].max()) + 1
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")


def set_obs(ax, env: MultiBase):
    if isinstance(env, Multi2dHoloCustom):
        for p, r in zip(env.obs_center, env.obs_rad):
            c = Circle(p, r, color="black")
            ax.add_artist(c)


def save_anim(env: MultiBase, xs: jnp.ndarray, output_path: Path, offset: int = 1):
    if env.pos_dim_agent == 3:
        save_anim_3d(env, xs, output_path)
        return

    fig, ax = plt.subplots(constrained_layout=True)
    set_ax_lim(ax, env, xs)
    set_obs(ax, env)

    circles, headings = [], []

    for i in range(env.n_agents):
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
        frames=xs.shape[0] // offset,
        blit=True,
        interval=100,
    )
    anim.save(output_path, writer=PillowWriter(fps=10 // offset))
    plt.close(fig)


def save_img(
    env: MultiBase,
    xs: jnp.ndarray,
    output_path: Path,
    offset: int = 1,
):
    if env.pos_dim_agent == 3:
        save_img_3d(env, xs, output_path)
        return

    # --- Generate Static Trajectory Image ---
    xs = xs[::offset]
    fig, ax = plt.subplots()
    set_ax_lim(ax, env, xs)
    set_obs(ax, env)

    for i in range(env.n_agents):
        color = get_color(i)

        traj_x, traj_y = xs[:, i, 0], xs[:, i, 1]
        ax.plot(traj_x, traj_y, color=color, linestyle="--", linewidth=1, alpha=0.5)

        start_circle = Circle(
            (traj_x[0], traj_y[0]), env.agent_radius, color=color, zorder=5
        )
        ax.add_artist(start_circle)

    # --- Collision Detection ---
    positions = xs[:, :, : env.pos_dim_agent]
    diffs = positions[:, :, None, :] - positions[:, None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    collision_matrix = (dists < env.agent_radius * 2 + env.safe_margin) & (dists > 0)
    collision_mask = jnp.any(collision_matrix, axis=-1)
    collision_positions = positions[collision_mask]
    if collision_positions.size > 0:
        ax.plot(
            collision_positions[:, 0],
            collision_positions[:, 1],
            "rx",
            markersize=10,
            markeredgewidth=1,
        )

    # --- Plot Goal Positions ---
    xg_reshaped = env.xg.reshape(env.n_agents, -1)
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


def save_anim_3d(env: MultiBase, xs: jnp.ndarray, output_path: Path, offset: int = 1):
    xs = xs.reshape(-1, env.n_agents, env.obsv_dim_agent)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xmin = min(xs[:, :, 0].min().item(), env.xg[:, 0].min().item()) - 1
    xmax = max(xs[:, :, 0].max().item(), env.xg[:, 0].max().item()) + 1
    ymin = min(xs[:, :, 1].min().item(), env.xg[:, 1].min().item()) - 1
    ymax = max(xs[:, :, 1].max().item(), env.xg[:, 1].max().item()) + 1
    zmin = min(xs[:, :, 2].min().item(), env.xg[:, 2].min().item()) - 1
    zmax = max(xs[:, :, 2].max().item(), env.xg[:, 2].max().item()) + 1

    def update(frame):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for i in range(env.n_agents):
            pos = xs[: frame + 1, i, : env.pos_dim_agent]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
            ax.scatter(*pos[-1], s=(env.agent_radius * 100) ** 2)

    anim = FuncAnimation(fig, update, frames=xs.shape[0], interval=100)
    anim.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def save_img_3d(
    env: MultiBase,
    xs: jnp.ndarray,
    output_path: Path,
    offset: int = 1,
):
    xs = xs[::offset]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(env.n_agents):
        color = get_color(i)
        ms = (env.agent_radius * 100) ** 2
        ax.plot(*xs[:, i, :3].T, color=color, linestyle="--", linewidth=1, alpha=0.5)
        ax.scatter(*xs[0, i, :3].T, color=color, s=ms)
        ax.scatter(*env.xg[i, :3].T, color=color, marker="+", s=ms)

    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
    )
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
