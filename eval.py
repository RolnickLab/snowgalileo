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
    PastisPatchEval,
    PastisPixelEval,
    So2SatEval,
    TreeSatEval,
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

encoder = Encoder.load_from_folder(Path(args["output_folder"])).to(device)

eval_tasks: List[EvalTask] = [
    *[So2SatEval(geobench=geobench) for geobench in [True, False]],
    *[
        PastisPatchEval(
            output_mode=output_mode,
            num_subtiles_per_image=num_subtiles_per_image,
            band_mode=band_mode,
        )
        for output_mode in ["norm_counts", "mode"]
        # 4 has input hw 64, 16 has input hw 32
        for num_subtiles_per_image in [4, 16]
        for band_mode in ["combined", "s2"]
    ],
    *[
        TreeSatEval(mode=mode, patch_size=patch_size)
        for mode in ["s1", "s2", "combined"]
        for patch_size in [6, 3]
    ],
    *[
        EuroSatEval(rgb=rgb, include_latlons=include_latlons)
        for rgb in [True, False]
        for include_latlons in [True, False]
    ],
    So2SatEval(),
    PastisPixelEval(),
    *[BinaryCropHarvestEval(country=country) for country in ["Kenya", "Togo", "Brazil", "China"]],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder, model_modes=["KNNat5"])
    print(json.dumps(results, indent=2), flush=True)
