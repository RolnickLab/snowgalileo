import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange

from src.fsc.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_segmentation_metrics,
)
from src.fsc.utils import SigmoidSlopeScheduler, landsat_binary_mapping
from src.snowgalileo import AttentionProbe, adjust_learning_rate
from src.utils import checkpoints_dir, save_checkpoint


class EncoderWithHead(nn.Module):
    def __init__(
        self,
        encoder,
        patch_size_high_res=10,
        inputs_per_target=10,
        num_patches_per_dim=10,
        sigmoid_slope=1.0,
        eval_config=None,
        med_and_low_res_repeat=True,
    ):
        super(EncoderWithHead, self).__init__()
        self.encoder = deepcopy(encoder)  # just in case
        # for segmentation
        # since our patch size is 10x10 and targets 100m resolution, each patch predicts 1 x 1 of 100m
        # since we do regression, we predict one value per patch
        self.logits_per_patch = int(
            (patch_size_high_res / inputs_per_target) * (patch_size_high_res / inputs_per_target)
        )
        self.number_of_patches = int(num_patches_per_dim * num_patches_per_dim)
        self.token_mapping = eval_config["token_mapping"]
        self.med_and_low_res_repeat = med_and_low_res_repeat
        self.eval_config = eval_config

        # first check if config has attn over spatial variable
        if "attend_over_spatial" in self.eval_config:
            if self.eval_config["attend_over_spatial"]:
                self.attn_output_dim = self.logits_per_patch
            else:
                self.attn_output_dim = self.number_of_patches * self.logits_per_patch

        if self.token_mapping == "spatial_mean":
            self.head = nn.Linear(encoder.embedding_size, self.logits_per_patch)
        elif self.token_mapping == "attention_probe":
            self.head = AttentionProbe(
                d_in=encoder.embedding_size,
                output_dim=self.attn_output_dim,
                n_heads=self.eval_config["n_heads"],
                attn_dropout_p=self.eval_config["attn_dropout_p"],
                use_tanh=self.eval_config["use_tanh"],
                hidden_dim=self.eval_config["hidden_dim"],
            )

        # attach a sigmoid to squeeze outputs to [0, 1]
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("sigmoid_slope", torch.tensor(sigmoid_slope))

    def forward(
        self,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        s_t_m_m,
        s_t_l_m,
        sp_m,
        t_m,
        st_m,
        months,
        patch_size_high_res=10,
        patch_size_med_res=1,
        patch_size_low_res=1,
    ):
        encodings = self.encoder(
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            months,
            patch_size_high_res=patch_size_high_res,
            patch_size_med_res=patch_size_med_res,
            patch_size_low_res=patch_size_low_res,
        )
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            _,
        ) = encodings

        # map patch mean of tokens to output using a linear layer + sigmoid.
        # maps from [batch, spatial_patches, embedding_dim] to [batch, spatial_patches, logits_per_patch].
        if self.token_mapping == "spatial_mean":
            encodings = self.encoder.apply_mask_and_average_tokens_per_highres_spatial_patch(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                med_and_low_res_repeat=self.med_and_low_res_repeat,
            )
            output = self.sigmoid(self.head(encodings) * self.sigmoid_slope)
        # map token sequence to patch output using attention probes.
        # maps from [batch, spatial_patches, tokens_per_patch, embedding_dim] to [batch, spatial_patches, logits_per_patch]
        # when attend_over_spatial=True
        elif self.token_mapping == "attention_probe":
            x, m, pos = self.encoder.preprocess_tokens_for_attention_probe(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                attend_over_spatial=self.eval_config.get("attend_over_spatial", True),
                med_and_low_res_repeat=self.med_and_low_res_repeat,
            )
            output = self.sigmoid(self.head(x, m, pos) * self.sigmoid_slope)
        else:
            raise ValueError(f"Unknown token mapping: {self.token_mapping}")
        return output


def finetune_and_eval_seg(
    loaders,
    encoder,
    device,
    identifier,
    eval_config,
    hyperparameter_config,
    num_finetune_epochs=50,
    log_wandb=False,
    wandb_id_parsed=None,
    sweep_run=None,
    checkpointing=False,
    job_id="",
):
    if log_wandb:
        wandb.init(
            entity="sea-ice",
            project="ai4snow_finetune_final",
            name=f"{identifier}-lr{hyperparameter_config.get('learning_rate')}",
            id=wandb_id_parsed,
            resume="allow",
        )
        wandb.config.update(hyperparameter_config)
        wandb.config.update(
            {
                "identifier": identifier,
                "num_finetune_epochs": num_finetune_epochs,
            }
        )
        wandb.config.update(eval_config)
        sweep_name = wandb.run.id
    elif sweep_run is not None:
        sweep_name = sweep_run.id
    else:
        sweep_name = "no_wandb"

    finetuned_model = finetune_seg(
        data_loaders=loaders,
        epochs=num_finetune_epochs,
        encoder=encoder,
        device=device,
        hyperparameter_config=hyperparameter_config,
        eval_config=eval_config,
        log_wandb=log_wandb,
        sweep_run=sweep_run,
        wandb_id_parsed=wandb_id_parsed,
        checkpointing=checkpointing,
        identifier=identifier,
        job_id=job_id,
    )
    results = evaluate_seg(
        data_loader=loaders["test"],
        finetuned_model=finetuned_model,
        device=device,
    )
    if checkpointing:
        filename = (
            f"{identifier}_{hyperparameter_config['initialization_id']}_{sweep_name}_{job_id}.pth"
        )
        save_checkpoint(finetuned_model, filename)
    return results


def get_finetune_results_on_val_set(
    loaders,
    encoder,
    num_runs,
    device,
    identifier,
    eval_config,
    hyperparameter_config,
    num_finetune_epochs,
    log_wandb=False,
    sweep_run=None,
    wandb_id_parsed=None,
    checkpointing=False,
    job_id="",
):
    final_vals = []
    for _ in range(num_runs):
        val = finetune_and_eval_seg(
            loaders=loaders,
            encoder=encoder,
            device=device,
            identifier=identifier,
            eval_config=eval_config,
            num_finetune_epochs=num_finetune_epochs,
            log_wandb=log_wandb,
            hyperparameter_config=hyperparameter_config,
            sweep_run=sweep_run,
            wandb_id_parsed=wandb_id_parsed,
            checkpointing=checkpointing,
            job_id=job_id,
        )
        final_vals.append(val)

    return final_vals


def finetune_seg(
    data_loaders,
    epochs,
    encoder,
    device,
    hyperparameter_config,
    eval_config,
    patch_size_high_res=10,
    inputs_per_target=10,
    log_wandb=False,
    sweep_run=None,
    wandb_id_parsed=None,
    checkpointing=False,
    identifier="",
    job_id="",
):
    # Use the wandB id as storage name if available, else the config if (less safe because not necessarily unique)
    run_id = (
        wandb_id_parsed
        if wandb_id_parsed is not None
        else wandb.run.id
        if log_wandb
        else sweep_run.id
        if sweep_run is not None
        else hyperparameter_config.get("initialization_id", "default")
    )
    run_path = Path(checkpoints_dir / run_id)
    config_path = run_path / "config.json"

    lr = hyperparameter_config.get("learning_rate", 0.1)
    weight_decay = hyperparameter_config.get("weight_decay", 0.0)
    lr_schedule = hyperparameter_config.get("lr_schedule", True)
    optimizer = hyperparameter_config.get("optimizer", "Adam")
    adam_beta_2 = hyperparameter_config.get("adam_beta_2", 0.999)
    schedule_sigmoid_slope = hyperparameter_config.get("schedule_sigmoid_slope", False)
    sigmoid_slope = hyperparameter_config.get("sigmoid_slope", 1.0)
    loss_fn = hyperparameter_config.get("loss_fn", "MSE")
    warmup_fraction = hyperparameter_config.get("warmup_fraction", 0.1)
    med_and_low_res_repeat = hyperparameter_config.get("med_and_low_res_repeat", True)
    num_patches_per_dim = hyperparameter_config.get("num_patches_per_dim", 10)

    train_loader = data_loaders["train"]
    val_loader = data_loaders["test"]

    finetuned_encoder = EncoderWithHead(
        encoder=encoder,
        patch_size_high_res=patch_size_high_res,
        inputs_per_target=inputs_per_target,
        num_patches_per_dim=num_patches_per_dim,
        sigmoid_slope=sigmoid_slope,
        eval_config=eval_config,
        med_and_low_res_repeat=med_and_low_res_repeat,
    ).to(device)

    finetuned_encoder = finetuned_encoder.train()

    if optimizer == "SGD":
        opt = torch.optim.SGD(
            finetuned_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        opt = torch.optim.Adam(
            finetuned_encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, adam_beta_2),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if eval_config["freeze_encoder"]:
        for param in finetuned_encoder.encoder.parameters():
            param.requires_grad = False

        # check that the encoder is frozen, while the linear layer is trainable
        for param in finetuned_encoder.encoder.parameters():
            assert not param.requires_grad
        for param in finetuned_encoder.head.parameters():
            assert param.requires_grad

    grad_accum = int(256 / train_loader.batch_size)
    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * warmup_fraction),
        "min_lr": 1.0e-6,
        "epochs": epochs,
    }

    # if checkpointing folder exists, load checkpoint
    if run_path.exists() and any(run_path.iterdir()):
        with config_path.open("r") as f:
            config = json.load(f)
        start_epoch = config.get("cur_epoch", 0)

        config.setdefault("restart_history", [])
        config["restart_history"].append(
            {
                "job_id": job_id,
                "resumed_from_epoch": start_epoch,
            }
        )

        with config_path.open("w") as f:
            json.dump(config, f, indent=4)

        print("Restarting from epoch:", start_epoch)
        finetuned_encoder.load_state_dict(
            torch.load(
                run_path
                / f"encoder.pt",
                map_location=device,
            )
        )
        opt.load_state_dict(
        torch.load(run_path / f"optimizer.pt", map_location=device)
        )
    else:
        run_path.mkdir(parents=True, exist_ok=True)

        config = {
            "cur_epoch": 0,
            "restart_history": [
                {
                    "job_id": job_id,
                    "resumed_from_epoch": 0,
                }
            ],
        }

        with config_path.open("w") as f:
            json.dump(config, f, indent=4)

        start_epoch = 0

    if loss_fn == "MSE":
        loss_function = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    for epoch in range(start_epoch, epochs):
        for i, (masked_output, labels, _) in enumerate(train_loader):
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = [t.to(device) for t in masked_output]

            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = finetuned_encoder(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size_high_res=patch_size_high_res,
                patch_size_med_res=1,
                patch_size_low_res=1,
            )
            spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
            logits = rearrange(
                torch.squeeze(logits),
                "b (h w) -> b h w",
                h=spatial_patches_per_dim,
                w=spatial_patches_per_dim,
            )
            loss = loss_function(logits, labels.to(device))

            (loss / grad_accum).backward()

            if ((i + 1) % grad_accum == 0) or (i + 1 == len(train_loader)):
                epoch_fraction = epoch + (i / len(train_loader))

                if schedule_sigmoid_slope:
                    raise NotImplementedError

                if lr_schedule:
                    set_lr = adjust_learning_rate(
                        optimizer=opt,
                        epoch=epoch_fraction,
                        total_epochs=sched_config["epochs"],
                        warmup_epochs=sched_config["warmup_epochs"],
                        max_lr=sched_config["lr"],
                        min_lr=sched_config["min_lr"],
                    )  # get LR for this epoch
                    for g in opt.param_groups:
                        g["lr"] = set_lr  # update

                torch.nn.utils.clip_grad_norm_(finetuned_encoder.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        torch.save(finetuned_encoder.state_dict(), run_path / f"encoder.pt")
        torch.save(opt.state_dict(), run_path / f"optimizer.pt")
        config["cur_epoch"] = epoch + 1
        with (run_path / "config.json").open("w") as f:
            json.dump(config, f)


        if epoch % 5 == 0 and checkpointing:
            file_path = Path(
                run_path
                / f"{identifier}_{hyperparameter_config['initialization_id']}_{run_id}_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(finetuned_encoder, file_path)
            torch.save(
                opt.state_dict(), run_path / f"optimizer_epoch_{epoch + 1}.pt"
            )

        if log_wandb or sweep_run is not None:
            if epoch % 5 == 0 or epoch == epochs - 1:
                results = evaluate_seg(
                    data_loader=val_loader,
                    finetuned_model=finetuned_encoder,
                    device=device,
                    patch_size_high_res=patch_size_high_res,
                )
                current_slope = finetuned_encoder.sigmoid_slope
                results["sigmoid_slope"] = current_slope
                results["learning_rate"] = opt.param_groups[0]["lr"]
                results["epoch"] = epoch
                if log_wandb:
                    wandb.log(results, step=epoch)
                if sweep_run is not None:
                    sweep_run.log(results, step=epoch)
                print(f"Finished epoch {epoch + 1}/{epochs}")

    return finetuned_encoder


def evaluate_binary(data_loader, finetuned_model, device, patch_size_high_res=10):
    finetuned_model = finetuned_model.eval()

    all_preds_1D = []
    all_labels_1D = []

    with torch.no_grad():
        for masked_output, labels, _ in data_loader:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = [t.to(device) for t in masked_output]

            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = finetuned_model(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size_high_res=patch_size_high_res,
                patch_size_med_res=1,
                patch_size_low_res=1,
            )

            # check that all predictions are between 0 and 1
            assert logits.min() >= 0 and logits.max() <= 1

            all_preds_1D.append(
                rearrange(torch.squeeze(logits, -1), "b s -> (b s)").float().cpu().numpy()
            )
            all_labels_1D.append(rearrange(labels, "b h w -> (b h w)").float().cpu().numpy())

    predictions = landsat_binary_mapping(np.concatenate(all_preds_1D))
    landsat_labels = landsat_binary_mapping(np.concatenate(all_labels_1D))

    results = compute_classification_metrics(predictions, landsat_labels)

    results_path = Path("./snowgalileo_binary_results.json")
    with results_path.open("w") as f:
        json.dump(results, f)


def evaluate_seg(
    data_loader,
    finetuned_model,
    device,
    patch_size_high_res=10,
):
    finetuned_model = finetuned_model.eval()

    all_preds_1D = []
    all_labels_1D = []

    all_preds_2D = []
    all_labels_2D = []

    with torch.no_grad():
        for masked_output, labels, _ in data_loader:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = [t.to(device) for t in masked_output]

            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = finetuned_model(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size_high_res=patch_size_high_res,
                patch_size_med_res=1,
                patch_size_low_res=1,
            )

            # check that all predictions are between 0 and 1
            assert logits.min() >= 0 and logits.max() <= 1

            all_preds_1D.append(
                rearrange(torch.squeeze(logits, -1), "b s -> (b s)").float().cpu().numpy()
            )
            all_labels_1D.append(rearrange(labels, "b h w -> (b h w)").float().cpu().numpy())

            spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
            logits = rearrange(
                torch.squeeze(logits, -1),
                "b (h w) -> b h w",
                h=spatial_patches_per_dim,
                w=spatial_patches_per_dim,
            )

            all_preds_2D.append(logits.float().cpu())
            all_labels_2D.append(labels.float().cpu())

    # sequence prediction
    all_preds_1D = np.concatenate(all_preds_1D)
    majority_baseline_preds_1D = np.zeros_like(all_preds_1D)
    all_labels_1D = np.concatenate(all_labels_1D)

    # mask for computing metrics without boundary values
    mask = (all_labels_1D > 0) & (all_labels_1D < 1)
    all_labels_1D_f = all_labels_1D[mask]
    all_preds_1D_f = all_preds_1D[mask]

    # create 10 bins for multi-class classification
    multi_class_bins = np.linspace(0.1, 1, 9)
    binned_preds_np = np.digitize(all_preds_1D, bins=multi_class_bins)
    binned_targets_np = np.digitize(all_labels_1D, bins=multi_class_bins)

    binned_preds_np_f = np.digitize(all_preds_1D_f, bins=multi_class_bins)
    binned_targets_np_f = np.digitize(all_labels_1D_f, bins=multi_class_bins)

    results = {
        "model": {},
        "baseline": {
            "majority": {},
            "balanced": {},
        },
    }

    results["model"]["regression"] = compute_regression_metrics(all_preds_1D, all_labels_1D)

    results["baseline"]["majority"]["regression"] = compute_regression_metrics(
        majority_baseline_preds_1D, all_labels_1D
    )

    results["baseline"]["balanced"]["regression"] = compute_regression_metrics(
        all_preds_1D_f, all_labels_1D_f
    )

    results["model"]["classification"] = compute_classification_metrics(
        binned_preds_np, binned_targets_np
    )

    results["baseline"]["majority"]["classification"] = compute_classification_metrics(
        majority_baseline_preds_1D, binned_targets_np
    )

    results["baseline"]["balanced"]["classification"] = compute_classification_metrics(
        binned_preds_np_f, binned_targets_np_f
    )

    # spatial prediction
    all_preds_2D = torch.cat(all_preds_2D)
    majority_baseline_preds_2D = torch.zeros_like(all_preds_2D)
    all_labels_2D = torch.cat(all_labels_2D)

    # create 10 bins for multi-class segmentation
    multi_class_bins = np.linspace(0.1, 1, 9)
    binned_preds_np = np.digitize(all_preds_2D, bins=multi_class_bins)
    binned_targets_np = np.digitize(all_labels_2D, bins=multi_class_bins)

    results["model"]["segmentation"] = compute_segmentation_metrics(
        binned_preds_np, binned_targets_np
    )

    results["baseline"]["majority"]["segmentation"] = compute_segmentation_metrics(
        majority_baseline_preds_2D, binned_targets_np
    )

    return results
