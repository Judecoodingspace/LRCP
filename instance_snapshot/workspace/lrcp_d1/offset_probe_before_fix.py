import os
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

ROOT = Path("/workspace/LRCP/opencood/logs")
MODEL_DIR = ROOT / (
    "v2xsim_point_pillar_lidar_deformable_attn_"
    "2026_09_03_21_27_18"
)

DELAYS = [0, 1, 2, 3, 4, 5]
MAX_BATCHES = 30

random.seed(303)
np.random.seed(303)
torch.manual_seed(303)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

assert device.type == "cuda", "PPU unavailable"

opt = SimpleNamespace(model_dir=str(MODEL_DIR))
hypes = yaml_utils.load_yaml(None, opt)
hypes["train_params"]["batch_size"] = 1
hypes["validate_dir"] = hypes["test_dir"]

print("D1 START", flush=True)
print("Model:", MODEL_DIR, flush=True)
print("Device:", device, flush=True)

model = train_utils.create_model(hypes)
epoch, model = train_utils.load_saved_model(
    str(MODEL_DIR), model
)
assert epoch == 19

model = model.to(device)
model.eval()

results = []

for delay in DELAYS:
    hypes["time_delay"] = delay

    dataset = build_dataset(
        hypes, visualize=True, train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        drop_last=False
    )

    counts = {
        "batches": 0,
        "agents": 0,
        "pixels": 0,
        "active": 0
    }

    sums = {
        "all_error": 0.0,
        "all_zero": 0.0,
        "active_error": 0.0,
        "active_zero": 0.0,
        "active_predmag": 0.0,
        "active_cos": 0.0
    }

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch = train_utils.to_device(
                batch, device
            )
            ego = batch["ego"]
            output = model(ego)

            pred = output["pred_offset"].float()
            target = ego["label_dict"]["offset"].float()

            assert pred.shape == target.shape
            assert pred.ndim == 4
            assert pred.shape[-1] == 2

            cal = ego["calibrate_data"]
            assert pred.shape[2] == len(cal)

            for j, slot in enumerate(cal):
                valid = cal[slot]["valid_mask"]
                valid = valid.to(
                    pred.device, dtype=torch.bool
                )

                if not bool(valid.any()):
                    continue

                p = pred[valid, :, j, :]
                g = target[valid, :, j, :]

                pmag = torch.linalg.vector_norm(
                    p, dim=-1
                )
                gmag = torch.linalg.vector_norm(
                    g, dim=-1
                )
                err = torch.linalg.vector_norm(
                    p - g, dim=-1
                )

                active = gmag > 1e-5

                counts["agents"] += p.shape[0]
                counts["pixels"] += err.numel()
                counts["active"] += int(
                    active.sum().item()
                )

                sums["all_error"] += err.sum().item()
                sums["all_zero"] += gmag.sum().item()

                if bool(active.any()):
                    sums["active_error"] += (
                        err[active].sum().item()
                    )
                    sums["active_zero"] += (
                        gmag[active].sum().item()
                    )
                    sums["active_predmag"] += (
                        pmag[active].sum().item()
                    )
                    cos = F.cosine_similarity(
                        p[active], g[active],
                        dim=-1, eps=1e-8
                    )
                    sums["active_cos"] += (
                        cos.sum().item()
                    )

            counts["batches"] += 1
            if counts["batches"] >= MAX_BATCHES:
                break

    assert counts["batches"] > 0
    assert counts["active"] > 0

    n = counts["active"]
    total = counts["pixels"]

    result = {
        "delay_index": delay,
        "delay_label_ms": delay * 100,
        "batches": counts["batches"],
        "valid_agent_instances": counts["agents"],
        "active_fraction": n / total,
        "epe_all_pred": sums["all_error"] / total,
        "epe_all_zero": sums["all_zero"] / total,
        "epe_active_pred": sums["active_error"] / n,
        "epe_active_zero": sums["active_zero"] / n,
        "mean_cos_active": sums["active_cos"] / n,
        "magnitude_ratio_active": (
            sums["active_predmag"] /
            max(sums["active_zero"], 1e-8)
        )
    }

    results.append(result)
    print(json.dumps(result, indent=2), flush=True)

out = Path(os.environ["D1_OUTPUT"])
with out.open("x") as f:
    json.dump(results, f, indent=2)

print("D1_PILOT_COMPLETE", flush=True)
print("RESULTS:", out, flush=True)
