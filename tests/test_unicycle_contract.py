import jax.numpy as jnp
import numpy as np

from d4orm.envs.unicycle import Unicycle


class FixedUnicycle(Unicycle):
    def __init__(self, starts, goals, **kwargs):
        self._starts = starts
        self._goals = goals
        super().__init__(**kwargs)

    def get_start_goal_configuration(self):
        return np.array(self._starts), np.array(self._goals)


def test_unicycle_reward_normalizes_by_position_distance_only():
    env = FixedUnicycle(
        starts=[[0.0, 0.0, 3.14, 0.0]],
        goals=[[1.0, 0.0, -3.14, 0.0]],
        n_agents=1,
    )

    assert float(env.max_distances[0]) == 1.0


def test_unicycle_stop_mask_uses_absolute_velocity():
    env = FixedUnicycle(
        starts=[[0.01, 0.0, 0.0, -1.0]],
        goals=[[0.0, 0.0, 0.0, 0.0]],
        n_agents=1,
        dt=0.01,
        stop_distance=0.02,
        stop_velocity=0.03,
    )

    state = env.state_init
    next_state = env.step(state, jnp.array([0.0, 0.0]))

    assert float(next_state.mask[0]) == 0.0
