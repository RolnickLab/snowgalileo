import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.flexipresto import adjust_learning_rate
from sklearn.metrics import root_mean_squared_error, r2_score, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

from .metrics import mean_iou
import numpy as np
from typing import Dict

#FT_LRs = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]
FT_LRs = [1e-5]

class EncoderWithHead(nn.Module):
    def __init__(self, encoder, patch_size_high_res=10, inputs_per_target=10):
        super(EncoderWithHead, self).__init__()
        self.encoder = deepcopy(encoder)  # just in case
        # for segmentation
        # since our patch size is 10x10 and targets 100m resolution, each patch predicts 1 x 1 of 100m
        # since we do regression, we predict one value per patch
        logits_per_patch = int((patch_size_high_res / inputs_per_target) * (patch_size_high_res / inputs_per_target))
        self.head = nn.Linear(encoder.embedding_size, logits_per_patch)
        # attach a sigmoid to squeeze outputs to [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m, months, patch_size_high_res=10, patch_size_med_res=1, patch_size_low_res=1):
        encodings = self.encoder(s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m, months, patch_size_high_res=patch_size_high_res, patch_size_med_res=patch_size_med_res, patch_size_low_res=patch_size_low_res)
        s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m, _ = encodings
        encodings = self.encoder.apply_mask_and_average_tokens_per_patch(
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
            )
        output = self.sigmoid(self.head(encodings))
        return output


def finetune_and_eval_seg(lr, loaders, encoder, device, identifier, num_finetune_epochs=50):
    finetuned_encoder = finetune_seg(
        data_loader=loaders["train"],
        lr=lr,
        epochs=num_finetune_epochs,
        encoder=encoder,
        device=device,
        freeze_encoder=False,
    )
    #val_miou = evaluate_seg(
    #    data_loader=loaders["valid"],
    #    finetuned_encoder=finetuned_encoder,
    #    num_classes=config["num_classes"],
    #    device=device,
    #)
    test_miou = evaluate_seg(
        data_loader=loaders["test"],
        finetuned_encoder=finetuned_encoder,
        device=device,
        identifier=identifier,
    )
    return test_miou


def linear_probe_and_eval_seg(lr, loaders, encoder, device, identifier):
    # we train the regression head for one epoch, while the encoder remains frozen
    encoder = finetune_seg(
        data_loader=loaders["train"],
        lr=lr,
        epochs=1,
        encoder=encoder,
        device=device,
        freeze_encoder=True,
    )
    #val_miou = evaluate_seg(
    #    data_loader=loaders["valid"],
    #    finetuned_encoder=finetuned_encoder,
    #    num_classes=config["num_classes"],
    #    device=device,
    #)
    test_miou = evaluate_seg(
        data_loader=loaders["test"],
        finetuned_encoder=encoder,
        device=device,
        identifier=identifier,
    )
    return test_miou

# TODO: implement validation too
def get_finetune_results_with_val(loaders, encoder, num_runs, device):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        vals = []
        tests = []
        for lr in FT_LRs:
            val, test = finetune_and_eval_seg(
                lr=lr, loaders=loaders, encoder=encoder, device=device
            )
            vals.append(val)
            tests.append(test)

        final_tests.append(tests[vals.index(max(vals))])

    return final_tests

def get_finetune_results(loaders, encoder, num_runs, device, identifier, num_finetune_epochs):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        tests = []
        for lr in FT_LRs:
            test = finetune_and_eval_seg(
                lr=lr, loaders=loaders, encoder=encoder, device=device, identifier=identifier, num_finetune_epochs=num_finetune_epochs
            )
            tests.append(test)

        final_tests.append(tests)

    return final_tests

def get_linear_probe_results(loaders, encoder, num_runs, device, identifier):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        tests = []
        for lr in FT_LRs:
            test = linear_probe_and_eval_seg(
                lr=lr, loaders=loaders, encoder=encoder, device=device, identifier=identifier
            )
            tests.append(test)

        final_tests.append(tests)

    return final_tests

def finetune_seg(data_loader, lr, epochs, encoder, device, freeze_encoder=False, patch_size_high_res=10, inputs_per_target=10):
    finetuned_encoder = EncoderWithHead(encoder=encoder, patch_size_high_res=patch_size_high_res, inputs_per_target=inputs_per_target).to(device)
    finetuned_encoder = finetuned_encoder.train()
    opt = torch.optim.AdamW(finetuned_encoder.parameters(), lr=lr)

    if freeze_encoder:
        for param in finetuned_encoder.encoder.parameters():
            param.requires_grad = False

        # check that the encoder is frozen, while the linear layer is trainable
        for param in finetuned_encoder.encoder.parameters():
            assert not param.requires_grad
        for param in finetuned_encoder.head.parameters():
            assert param.requires_grad

    grad_accum = int(256 / data_loader.batch_size)
    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-6,
        "epochs": epochs,
    }

    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        for i, (masked_output, labels, _) in enumerate(data_loader):
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


            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

            if ((i + 1) % grad_accum == 0) or (i + 1 == len(data_loader)):
                epoch_fraction = epoch + (i / len(data_loader))
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

    return finetuned_encoder

def compute_regression_metrics(identifier: str, preds: np.ndarray, target: np.ndarray, baseline=False) -> Dict[str, float]:
    if baseline:
        bs = "baseline_"
    else:
        bs = ""

    return {
        f"{bs}{identifier}_rmse": root_mean_squared_error(target, preds),
        f"{bs}{identifier}_r2": r2_score(target, preds),
    }

def compute_classification_metrics(identifier: str, preds: np.ndarray, target: np.ndarray, baseline=False) -> Dict[str, float]:
    if baseline:
        bs = "baseline_"
    else:
        bs = ""

    return {
        f"{bs}{identifier}_overall_accuracy": accuracy_score(target, preds),
        f"{bs}{identifier}_balanced_accuracy": balanced_accuracy_score(target, preds),
        f"{bs}{identifier}_recall": recall_score(target, preds, average='weighted'),
        f"{bs}{identifier}_precision": precision_score(target, preds, average='weighted'),
        f"{bs}{identifier}_f1": f1_score(target, preds, average='weighted'),
    }

def compute_segmentation_metrics(identifier: str, preds: np.ndarray, target: np.ndarray, baseline=False) -> Dict[str, float]:
    if baseline:
        bs = "baseline_"
    else:
        bs = ""

    return {
        f"{bs}{identifier}_rmse": mean_iou(preds, target, num_classes=10),
    }

def evaluate_seg(data_loader, finetuned_encoder, device, identifier, patch_size_high_res=10):
    finetuned_encoder = finetuned_encoder.eval()

    all_preds_1D = []
    all_labels_1D = []

    all_preds_2D = []
    all_labels_2D = []

    results_dict: Dict[str, float] = {}

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

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

            # check that all predictions are between 0 and 1
            assert logits.min() >= 0 and logits.max() <= 1

            all_preds_1D.append(rearrange(torch.squeeze(logits), "b (h w) -> (b h w)").numpy().cpu())
            all_labels_1D.append(rearrange(labels, "b h w -> (b h w)").numpy().cpu())

            spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
            logits = rearrange(
                torch.squeeze(logits),
                "b (h w) -> b h w",
                h=spatial_patches_per_dim,
                w=spatial_patches_per_dim,
            )

            all_preds_2D.append(logits.cpu())
            all_labels_2D.append(labels.cpu())

    # sequence prediction
    all_preds_1D = np.concatenate(all_preds_1D)
    baseline_preds_1D = np.zeros_like(all_preds_1D)
    all_labels_1D = np.concatenate(all_labels_1D)

    # create 10 bins for multi-class classification
    multi_class_bins = np.linspace(0.1, 1, 9)
    binned_preds_np = np.digitize(all_preds_1D, bins=multi_class_bins)
    binned_targets_np = np.digitize(all_labels_1D, bins=multi_class_bins)

    # sequence regression
    results_dict.update(
        compute_regression_metrics(
            identifier,
            all_preds_1D,
            all_labels_1D,
            baseline=False
        )
    )
    # sequence regression (baseline)
    results_dict.update(
        compute_regression_metrics(
            identifier,
            baseline_preds_1D,
            all_labels_1D,
            baseline=True
        )
    )
    # sequence classification
    results_dict.update(
        compute_classification_metrics(
            identifier,
            binned_preds_np,
            binned_targets_np,
            baseline=False,
        )
    )
    # sequence classification (baseline)
    results_dict.update(
        compute_classification_metrics(
            identifier,
            baseline_preds_1D,
            binned_targets_np,
            baseline=True,
        )
    )

    # spatial prediction
    all_preds_2D = torch.cat(all_preds_2D)
    baseline_preds_2D = torch.zeros_like(all_preds_2D)
    all_labels_2D = torch.cat(all_labels_2D)

    # create 10 bins for multi-class segmentation
    multi_class_bins = np.linspace(0.1, 1, 9)
    binned_preds_np = np.digitize(all_preds_2D, bins=multi_class_bins)
    binned_targets_np = np.digitize(all_labels_2D, bins=multi_class_bins)

    results_dict.append(
        compute_segmentation_metrics(
            identifier,
            binned_preds_np,
            binned_targets_np,
            baseline=False
        )
    )
    results_dict.append(
        compute_segmentation_metrics(
            identifier,
            baseline_preds_2D,
            binned_targets_np,
            baseline=True
        )
    )

    return results_dict