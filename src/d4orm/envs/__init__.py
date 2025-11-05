from .multi2dholo import Multi2dHolo
from .multi3dholo import Multi3dHolo


def get_env(env_name: str, num_agents: int):
    if env_name == "multi2dholo":
        return Multi2dHolo(num_agents)
    elif env_name == "multi3dholo":
        return Multi3dHolo(num_agents)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
