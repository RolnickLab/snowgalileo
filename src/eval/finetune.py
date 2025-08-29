import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.flexipresto import adjust_learning_rate

from .metrics import mean_iou

FT_LRs = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]


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
        encodings = rearrange(
            self.encoder.apply_mask_and_average_tokens_per_patch(
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
            ),
            "b n_t n_f -> (b n_t) n_f",
        )
        output = self.sigmoid(self.head(encodings))
        return output


def finetune_and_eval_seg(lr, loaders, encoder, device):
    finetuned_encoder = finetune_seg(
        data_loader=loaders["train"],
        lr=lr,
        epochs=50,
        encoder=encoder,
        device=device,
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
    )
    return test_miou


def get_finetune_results_with_val(loaders, encoder, num_runs, device):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        vals = []
        tests = []
        for lr in FT_LRs:
            val, test = finetune_and_eval_seg(
                lr=lr, config=config, loaders=loaders, encoder=encoder, device=device
            )
            vals.append(val)
            tests.append(test)

        final_tests.append(tests[vals.index(max(vals))])

    return final_tests

def get_finetune_results(loaders, encoder, num_runs, device):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        tests = []
        for lr in FT_LRs:
            test = finetune_and_eval_seg(
                lr=lr, loaders=loaders, encoder=encoder, device=device
            )
            tests.append(test)

        final_tests.append(tests)

    return final_tests

def finetune_seg(data_loader, lr, epochs, encoder, device, num_classes=1, patch_size_high_res=10, inputs_per_target=10):
    finetuned_encoder = EncoderWithHead(encoder=encoder, patch_size_high_res=patch_size_high_res, inputs_per_target=inputs_per_target).to(device)
    finetuned_encoder = finetuned_encoder.train()
    opt = torch.optim.AdamW(finetuned_encoder.parameters(), lr=lr)

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
                    logits,
                    "b (h w) (c i j) -> b c (h i) (w j)",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                    c=num_classes,
                    i=1,
                    j=1,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(labels.shape[-2], labels.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bsz, num_classes, H, W)
                loss = loss_function(logits, labels.to(device))

            (loss / grad_accum).backward()

            if ((i + 1) % grad_accum == 0) or (i + 1 == len(data_loader)):
                epoch_fraction = epoch + (i / len(data_loader))
                set_lr = adjust_learning_rate(
                    epoch_fraction, sched_config
                )  # get LR for this epoch
                for g in opt.param_groups:
                    g["lr"] = set_lr  # update

                torch.nn.utils.clip_grad_norm_(finetuned_encoder.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

    return finetuned_encoder

def evaluate_seg(data_loader, finetuned_encoder, device, num_classes=1, patch_size_high_res=10):
    finetuned_encoder = finetuned_encoder.eval()

    all_preds = []
    all_labels = []
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
                spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                logits = rearrange(
                    logits,
                    "b (h w) (c i j) -> b c (h i) (w j)",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                    c=num_classes,
                    i=1,
                    j=1,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(labels.shape[-2], labels.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bsz, num_classes, H, W)

            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    miou = mean_iou(all_preds, all_labels, num_classes=num_classes, ignore_label=-1)
    return miou
