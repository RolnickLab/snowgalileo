import os
from copy import deepcopy

import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER, EE_PROJECT
from src.masked_datasets import PrestoToPrestoMaskedDataset
from src.presto import Encoder, PrestoDecoder
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


# this should live elsewhere
num_epochs = 2
batch_size = 1
ema = (0.996, 1.0)
mask_ratio = 0.5

print("Loading dataset and dataloader")
dataset = PrestoToPrestoMaskedDataset(DATA_FOLDER / "test_files" / "presto_tif", mask_ratio=mask_ratio, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Loading models")
encoder = Encoder(embedding_size=64).to(device)
predictor = PrestoDecoder(encoder_embedding_size=64, decoder_embedding_size=64).to(device)
target_encoder = deepcopy(encoder)


param_groups = [
    {
        "params": (
            p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
        )
    },
    {
        "params": (
            p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
        )
    },
    {
        "params": (
            p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
        ),
        "WD_exclude": True,
        "weight_decay": 0,
    },
    {
        "params": (
            p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
        ),
        "WD_exclude": True,
        "weight_decay": 0,
    },
]
# todo - implement schedule following IJEPA
optimizer = torch.optim.AdamW(param_groups)  # type: ignore
iterations_per_epoch = len(dataset)
momentum_scheduler = (
    ema[0] + i * (ema[1] - ema[0]) / (iterations_per_epoch * num_epochs)
    for i in range(int(iterations_per_epoch * num_epochs) + 1)
)

for e in tqdm(range(num_epochs)):
    for i, b in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        b = [t.to(device) for t in b]
        d_x, s_x, d_m, s_m, months = b
        reversed_d, reversed_s = (1 - d_m).bool(), (1 - s_m).bool()

        # generate the predictions. TODO: add layer norm
        p_d, p_s, _, _ = predictor(
            *encoder(d_x.float(), s_x.float(), d_m.float(), s_m.float(), months.long())
        )
        # generate the targets
        with torch.no_grad():
            t_d, t_s, _, _, _ = target_encoder(
                d_x.float(),
                s_x.float(),
                torch.zeros_like(d_m),
                torch.zeros_like(s_m),
                months.long(),
            )

        loss = F.smooth_l1_loss(
            torch.concat([p_d[reversed_d], p_s[reversed_s]]),
            torch.concat([t_d[reversed_d], t_s[reversed_s]]),
        )
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {e}, iteration {i}: loss = {loss.item()}, memory used: {process.memory_info().rss}",
            flush=True,
        )
        optimizer.zero_grad()
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
