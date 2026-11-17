import os
import json
import random
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

ROOT = Path("/workspace/LRCP")
MODEL_DIR = ROOT / (
    "opencood/logs/"
    "v2xsim_point_pillar_lidar_deformable_attn_"
    "2026_09_03_21_27_18"
)

EXPECTED_SHA = (
    "666f444e6127e0594788b9eb9df8de11bd1fa23043a0773835cf438235a62724"
)

DELAYS = [1, 2]
ALPHAS = [1.0, 0.5, 0.0]
MAX_BATCHES = 1000000000

random.seed(303)
np.random.seed(303)
torch.manual_seed(303)

device = torch.device("cuda:0")
assert torch.cuda.is_available()

ckpt = MODEL_DIR / "net_epoch_bestval_at19.pth"
sha = hashlib.sha256()
with ckpt.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha.update(chunk)
assert sha.hexdigest() == EXPECTED_SHA

opt = SimpleNamespace(model_dir=str(MODEL_DIR))
hypes = yaml_utils.load_yaml(None, opt)
hypes["train_params"]["batch_size"] = 1
hypes["validate_dir"] = hypes["test_dir"]

model = train_utils.create_model(hypes)
epoch, model = train_utils.load_saved_model(
    str(MODEL_DIR), model
)
assert epoch == 19

model = model.to(device)
model.eval()

assert hasattr(model, "meta_flow")

state = {"alpha": 1.0, "calls": 0}

def scale_offset(module, inputs, output):
    state["calls"] += 1
    assert output.ndim == 4
    assert output.shape[-1] == 2
    return output * state["alpha"]

hook = model.meta_flow.register_forward_hook(scale_offset)

def new_stats():
    return {
        threshold: {
            "tp": [], "fp": [], "gt": 0, "score": []
        }
        for threshold in [0.5, 0.7]
    }

results = []

print("D2 PAIRED PILOT START", flush=True)
print("Checkpoint verified", flush=True)

for delay in DELAYS:
    hypes["time_delay"] = delay

    dataset = build_dataset(
        hypes, visualize=True, train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
        drop_last=False
    )

    stats = {a: new_stats() for a in ALPHAS}
    sample_ids = []
    count = 0
    start_calls = state["calls"]

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]

            sample_ids.append(str(ego.get("sample_idx")))
            # Reuse the exact same processed batch
            # for all compensation strengths.
            for alpha in ALPHAS:
                state["alpha"] = alpha

                result = (
                    inference_utils.inference_intermediate_fusion(
                        batch, model, dataset
                    )
                )

                for threshold in (0.5, 0.7):
                    eval_utils.caluclate_tp_fp(
                        result["pred_box_tensor"],
                        result["pred_score"],
                        result["gt_box_tensor"],
                        stats[alpha],
                        threshold
                    )

            count += 1

            if count % 20 == 0:
                print(
                    f"delay={delay} batches={count}",
                    flush=True
                )

            if count >= MAX_BATCHES:
                break

    assert count > 0, (
        f"No valid batches at delay {delay}"
    )
    assert state["calls"] > start_calls

    row = {
        "delay_label_ms": delay * 100,
        "valid_batches": count,
        "sample_ids": sample_ids,
        "results": {}
    }

    for alpha in ALPHAS:
        ap50, _, _ = eval_utils.calculate_ap(
            stats[alpha], 0.5
        )
        ap70, _, _ = eval_utils.calculate_ap(
            stats[alpha], 0.7
        )
        row["results"][str(alpha)] = {
            "ap50": float(ap50),
            "ap70": float(ap70)
        }

    results.append(row)

    print(json.dumps({
        "delay_label_ms": row["delay_label_ms"],
        "valid_batches": count,
        "results": row["results"]
    }, indent=2), flush=True)

    output_path = Path(os.environ["D2_OUTPUT"])
    output_path.write_text(
        json.dumps(results, indent=2)
    )

hook.remove()

print("D2_PILOT_COMPLETE", flush=True)
