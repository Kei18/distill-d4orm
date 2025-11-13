from .multi2dholo import Multi2dHolo, Multi2dHoloCustom


def get_env_cls(env_name: str):
    if env_name in ["2dholo", "multi2dholo"]:
        return Multi2dHolo
    if env_name in ["2dholo_custom", "multi2dholo_custom"]:
        return Multi2dHoloCustom
    else:
        raise ValueError(f"Unknown environment: {env_name}")
