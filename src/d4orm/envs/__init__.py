from .multi2dholo import Multi2dHolo


def get_env_cls(env_name: str):
    if env_name == "multi2dholo":
        return Multi2dHolo
    else:
        raise ValueError(f"Unknown environment: {env_name}")
