from datetime import datetime

import time
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml
from dacite import from_dict, Config
import tyro
from loguru import logger

from d4orm import (
    D4ormCfg,
    EnvConfig,
    get_env_cls,
    save_anim,
    save_img,
    d4orm_opt,
    get_metrics,
)


@dataclass
class Args(EnvConfig, D4ormCfg):
    # env
    env_name: str = "multi2dholo"
    # result
    save_img: bool = True
    save_gif: bool = False


def main(args: Args):
    # setup env
    env_cls = get_env_cls(args.env_name)
    env = from_dict(env_cls, asdict(args), config=Config(strict=False))
    logger.info(f"finish {env.__class__.__name__} setup, start D4orm")

    # set d4orm parameters
    cfg = from_dict(D4ormCfg, asdict(args), config=Config(strict=False))

    ## run d4orm
    start_time = time.time()
    U_base, aux = d4orm_opt(cfg=cfg, env=env)
    elapsed_time = time.time() - start_time
    rews, states, goal_masks, collisions = aux["rollout"]
    actions = U_base.reshape(U_base.shape[0], env.n_agents, -1)
    metrics = get_metrics(aux)

    # trimming
    states = states[: metrics["max_steps_to_goal"]]
    actions = actions[: metrics["max_steps_to_goal"]]

    # construct solution data
    sol = dict(
        success=(metrics["goal_reach_rate"] == 1.0 and metrics["num_collisions"] == 0),
        elapsed_time_sec=elapsed_time,
        num_denoising=aux["num_denoising"],
        **metrics,
    )
    logger.info("results:")
    for key, val in sol.items():
        logger.info(f"- {key:30s}: {val}")

    sol["dt"] = env.dt
    sol["instance"] = env.asdict()
    sol["result"] = []
    for i in range(env.n_agents):
        x_i = states[:, i].tolist()
        u_i = actions[:, i].tolist()
        sol["result"].append(dict(actions=u_i, states=x_i))

    # save results
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(__file__).parent / f"outputs/{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"saved in {output_dir}")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(args), f)
    with open(output_dir / "result.yaml", "w") as f:
        yaml.safe_dump(sol, f)

    if args.save_img:
        save_img(env, states, output_dir / "trajectories.png")
    if args.save_gif:
        save_anim(env, states, output_dir / "trajectories.gif")


if __name__ == "__main__":
    main(args=tyro.cli(Args))
