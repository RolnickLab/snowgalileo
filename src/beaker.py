import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("__main__")


@dataclass
class BeakerRuntimeConfig:
    beaker_workload_id: str
    beaker_node_hostname: str
    beaker_experiment_url: str


def is_beaker_job() -> bool:
    return "BEAKER_JOB_ID" in os.environ


def maybe_get_beaker_config():
    return BeakerRuntimeConfig(
        beaker_workload_id=os.environ["BEAKER_EXPERIMENT_ID"],
        beaker_node_hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        beaker_experiment_url=f"https://beaker.org/ex/{os.environ['BEAKER_EXPERIMENT_ID']}/",
    )
