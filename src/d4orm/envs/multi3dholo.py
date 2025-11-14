from dataclasses import dataclass

from .multi2dholo import Multi2dHolo


@dataclass(eq=False)
class Multi3dHolo(Multi2dHolo):
    obsv_dim_agent: int = 6
    pos_dim_agent: int = 3
    action_dim_agent: int = 3
    agent_radius: float = 0.15

    mv: float = 10.0  # max velocity
    ma: float = 5.0  # max acceleration
