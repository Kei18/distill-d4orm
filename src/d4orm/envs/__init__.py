from .multi2dholo import Multi2dHolo, Multi2dHoloRandom, Multi2dHoloCustom
from .unicycle import Unicycle


def get_env_cls(env_name: str):
    if env_name in ["2dholo", "multi2dholo"]:
        return Multi2dHolo
    if env_name in ["2dholo_random", "multi2dholo_random"]:
        return Multi2dHoloRandom
    if env_name in ["2dholo_custom", "multi2dholo_custom"]:
        return Multi2dHoloCustom
    if env_name in ["unicycle"]:
        return Unicycle
    else:
        raise ValueError(f"Unknown environment: {env_name}")
