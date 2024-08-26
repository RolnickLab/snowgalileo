import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("__main__")


@dataclass
class BeakerRuntimeConfig:
    beaker_workload_id: str
    beaker_node_hostname: str
    beaker_experiment_url: str
    beaker_dataset_ids: Optional[List[str]] = None
    beaker_dataset_id_urls: Optional[List[str]] = None


def is_beaker_job() -> bool:
    return "BEAKER_JOB_ID" in os.environ


def get_beaker_dataset_ids(experiment_id: str) -> Optional[List[str]]:
    get_experiment_command = f"beaker experiment get {experiment_id} --format json"
    process = subprocess.Popen(
        ["bash", "-c", get_experiment_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Failed to get Beaker experiment: {stderr}")
        return None
    experiment = json.loads(stdout)[0]
    result_ids = [job["result"]["beaker"] for job in experiment["jobs"]]
    dataset_ids = []
    for result_id in result_ids:
        get_dataset_command = f"beaker dataset get {result_id} --format json"
        process = subprocess.Popen(
            ["bash", "-c", get_dataset_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Failed to get Beaker dataset: {stderr}")
            return None
        datasets = json.loads(stdout)
        dataset_ids.extend([dataset["id"] for dataset in datasets])
    return dataset_ids


def get_beaker_whoami() -> Optional[str]:
    get_beaker_whoami_command = "beaker account whoami --format json"
    process = subprocess.Popen(
        ["bash", "-c", get_beaker_whoami_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Failed to get Beaker account: {stderr}")
        return None
    accounts = json.loads(stdout)
    return accounts[0]["name"]


def maybe_get_beaker_config():
    beaker_dataset_ids = [get_beaker_dataset_ids(os.environ["BEAKER_EXPERIMENT_ID"])]
    beaker_dataset_id_urls = [
        f"https://beaker.org/ds/{dataset_id}" for dataset_id in beaker_dataset_ids
    ]
    return BeakerRuntimeConfig(
        beaker_workload_id=os.environ["BEAKER_EXPERIMENT_ID"],
        beaker_node_hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        beaker_experiment_url=f"https://beaker.org/ex/{os.environ['BEAKER_EXPERIMENT_ID']}/",
        beaker_dataset_ids=get_beaker_dataset_ids(os.environ["BEAKER_EXPERIMENT_ID"]),
        beaker_dataset_id_urls=beaker_dataset_id_urls,
    )
