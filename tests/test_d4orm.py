from d4orm.d4orm import D4ormCfg, d4orm_opt, get_metrics
from d4orm.envs.multi2dholo import Multi2dHolo


def test_D4orm():
    env = Multi2dHolo()
    cfg = D4ormCfg()
    U_base, aux = d4orm_opt(cfg=cfg, env=env)
    print(get_metrics(aux))
