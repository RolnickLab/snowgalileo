import argparse
import json
import os
from pathlib import Path

import wandb

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.eval import LandsatEval
from src.eval.eval import EvalTask
from src.snowgalileo import Encoder
from src.utils import device, load_check_config, seed_everything

seed_everything(DEFAULT_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain", default="none", type=str, choices=["none", "snow"])
parser.add_argument("--resample", action="store_true")
parser.add_argument("--num_finetune_epochs", type=int, default=25)
# TODO: make the choices of naming more descriptive
parser.add_argument(
    "--decoding_strategy",
    type=str,
    default="attention_probe",
    choices=["finetune", "linear_probe", "attention_probe", "sklearn"],
    help="Decoding strategy to use. 'Finetune' uses a linear decoder and finetunes the entire model. 'Linear_probe' uses a linear decoder and only trains the decoder. 'Attention_probe' uses an attention-based decoder and fine-tunes the entire model. 'sklearn' uses the frozen encoder features for a sklearn model.",
)
parser.add_argument(
    "--eval_config",
    type=str,
    default="fsc_train_100m.json",
    help="Which eval config to use. Options are stored in src/eval/eval_configs/",
)
parser.add_argument(
    "--h5pys_only",
    action="store_true",
    help="Where to only use h5pys (faster, but need to be already stored in this format)",
)
args = parser.parse_args()
pretrain = args.pretrain

# TODO: discuss which metric to optimize
sweep_configuration = {
    "name": f"sweep_pretrain_{args.pretrain}_resample_{args.resample}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]},
        "lr_schedule": {"values": [True, False]},
        "warmup_fraction": {"values": [0.0, 0.05, 0.1, 0.2]},
        "batch_size": {"values": [16]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
        "num_workers": {"values": [4]},
        "sigmoid_slope": {"values": [0.01, 0.1, 0.5, 1.0]},
        "loss_fn": {"values": ["MSE"]},
    },
}


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train_and_validate():
    args = parser.parse_args()

    with (
        Path(__file__).parents[0] / Path("src/eval/eval_configs") / Path(args["eval_config"])
    ).open("r") as f:
        eval_config = json.load(f)

    with wandb.init(project="ai4snow_sweeps") as sweep_run:
        if args.pretrain == "snow":
            # load pretrained snowgalileo encoder
            encoder = Encoder.load_from_folder(
                Path(DATA_FOLDER / "outputs/checkpoints_tiny/epoch_80")
            ).to(device)
            initialization_id = "snowgalileo_pretrained"
        else:
            # randomly initialized snowgalileo encoder
            config = load_check_config("ai4snow_ps10.json")
            encoder = Encoder(**config["model"]["encoder"]).to(device)
            initialization_id = "snowgalileo_random"

        sweep_run.config.update(args)
        sweep_run.config.update({"initialization_id": initialization_id})

        eval_task: EvalTask = LandsatEval(
            exclude_prediction_high_res=False,
            resample=args.resample,
            num_finetune_epochs=args.num_finetune_epochs,
            decoder_mode=args.decoding_strategy,
            h5pys_only=args.h5pys_only,
            eval_config=eval_config,
        )

        results = eval_task.train_and_evaluate_model_on_task(
            pretrained_model=encoder,
            model_modes=["Regression"],
            baseline_galileo=(args.pretrain == "galileo"),
            hyperparams_config=sweep_run.config,
            log_wandb=False,
            initialization_id=initialization_id,
            sweep_run=sweep_run,
            save_final_checkpoint=False,
        )
        # log metric to sweep run
        # TODO: change the metric names based on eval config
        sweep_run.log(
            {
                "r2": results.get("landsat_s42_ps10_8_r2", -1),
                "rmse": results.get("landsat_s42_ps10_8_rmse", -1),
                "overall_accuracy": results.get("landsat_s42_ps10_8_overall_accuracy", -1),
                "balanced_accuracy": results.get("landsat_s42_ps10_8_balanced_accuracy", -1),
                "recall": results.get("landsat_s42_ps10_8_recall", -1),
                "precision": results.get("landsat_s42_ps10_8_precision", -1),
                "f1": results.get("landsat_s42_ps10_8_f1", -1),
                "miou": results.get("landsat_s42_ps10_8_miou", -1),
            }
        )
        sweep_run.finish()


def main():
    wandb.login()

    sweep_config = sweep_configuration
    # number of runs in the sweep
    count = 100

    sweep_id = wandb.sweep(sweep=sweep_config, project="ai4snow_sweeps", entity="sea-ice")
    wandb.agent(sweep_id, function=train_and_validate, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
