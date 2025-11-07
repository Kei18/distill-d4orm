from .multi2dholo import Multi2dHolo


def get_env(env_name: str, num_agents: int):
    if env_name == "multi2dholo":
        return Multi2dHolo(num_agents=num_agents)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
