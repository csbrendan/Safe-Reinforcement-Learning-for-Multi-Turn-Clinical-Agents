"""SkyRL entrypoint for SD-Bench GRPO (Azure GPU VM only; see V2/skyrl_plan.md).

Same pattern as SkyRL's own custom-env examples: register the env inside the Ray entrypoint task, then run the
standard PPO/GRPO experiment. The env config file travels in V2_ENV_CONFIG (set by run_pilot_27b.sh).
"""
import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    from tasks.v2_sdbench.skyrl_env import register_env
    register_env()
    BasePPOExp(cfg).run()


def main() -> None:
    if not os.environ.get("V2_ENV_CONFIG"):
        sys.exit("set V2_ENV_CONFIG to a V2/sdbench_configs/*.json file")
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
