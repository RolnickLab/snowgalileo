import argparse
import json
from pathlib import Path
from typing import List

import psutil
import torch

from src.config import DEFAULT_SEED
from src.eval import (
    BinaryCropHarvestEval,
    EuroSatEval,
    PastisEval,
    So2SatEval,
    TreeSatEval,
)
from src.eval.eval import EvalTask
from src.flexipresto import Encoder
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str)
args = argparser.parse_args().__dict__

encoder = Encoder.load_from_folder(Path(args["output_folder"]))

eval_tasks: List[EvalTask] = [
    *[TreeSatEval(mode, patch_size) for mode in ["s1", "s2", "combined"] for patch_size in [6, 3]],
    *[EuroSatEval(rgb) for rgb in [True, False]],
    So2SatEval(),
    PastisEval(),
    *[BinaryCropHarvestEval(country=country) for country in ["Kenya", "Togo", "Brazil", "China"]],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder)
    print(json.dumps(results, indent=2), flush=True)
