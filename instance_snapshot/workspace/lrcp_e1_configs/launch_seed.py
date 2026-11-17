import os
import sys
import random
import runpy

import numpy as np
import torch

seed = int(os.environ["E1_SEED"])
config = os.environ["E1_CONFIG"]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("=== E1 TRAINING ===", flush=True)
print("Seed:", seed, flush=True)
print("Config:", config, flush=True)
print("Torch:", torch.__version__, flush=True)
print("Device:", torch.cuda.get_device_name(0), flush=True)

os.chdir("/workspace/LRCP")
sys.path.insert(0, "/workspace/LRCP")

sys.argv = [
    "opencood/tools/train_w_delay.py",
    "--hypes_yaml", config,
    "--fusion_method", "intermediate"
]

runpy.run_path(
    "opencood/tools/train_w_delay.py",
    run_name="__main__"
)
