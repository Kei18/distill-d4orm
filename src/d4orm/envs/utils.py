import jax
import jax.numpy as jnp


def generate_sphere_configuraiton(
    diameter: float,
    n_agents: int,
    obsv_dim_agent: int,
    pos_dim_agent: int,
    rng: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    s: float = 0.0
    if rng is not None:
        _, rng = jax.random.split(rng)
        s = jax.random.uniform(rng, maxval=2 * jnp.pi).item()

    radius = diameter / 2.0
    if pos_dim_agent == 2:
        angles = jnp.linspace(s, s + 2 * jnp.pi, n_agents, endpoint=False)
        position_components = [
            radius * jnp.cos(angles),  # x
            radius * jnp.sin(angles),  # y
        ]
    elif pos_dim_agent == 3:
        indices = jnp.arange(n_agents)
        phi = jnp.arccos(1 - 2 * (indices + 0.5) / n_agents)
        theta = jnp.pi * (1 + 5**0.5) * indices
        position_components = [
            radius * jnp.sin(phi) * jnp.cos(theta),
            radius * jnp.sin(phi) * jnp.sin(theta),
            radius * jnp.cos(phi),
        ]
    zero_components = [
        jnp.zeros(n_agents) for _ in range(obsv_dim_agent - pos_dim_agent)
    ]
    initial_states = jnp.stack(position_components + zero_components, axis=-1)
    goal_states = -initial_states
    return initial_states, goal_states
