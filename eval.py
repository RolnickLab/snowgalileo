import argparse
import json
from typing import List

import psutil
import torch

from src.config import DEFAULT_SEED
from src.data.config import OUTPUT_FOLDER
from src.eval import (
    So2SatEval,
)
from src.eval.eval import EvalTask
from src.flexipresto import Encoder
from src.utils import device, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str)
args = argparser.parse_args().__dict__

encoder = Encoder.load_from_folder(OUTPUT_FOLDER).to(device)

eval_tasks: List[EvalTask] = [
    *[So2SatEval(geobench=geobench) for geobench in [True, False]],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder, model_modes="KNNat5")
    print(json.dumps(results, indent=2), flush=True)
