import sys
import runpy
import torch

sys.path.insert(0, "/workspace/LRCP")

from opencood.models.meta_flow import MetaFlow

if len(sys.argv) != 2:
    raise SystemExit("Usage: python launch_b0_zero_offset.py MODEL_DIR")

model_dir = sys.argv[1]

original_forward = MetaFlow.forward
call_counter = {"count": 0}

def zero_offset_forward(self, features, time):
    # Execute original MetaFlow, but remove its contribution
    # at the output boundary.
    predicted = original_forward(self, features, time)

    assert predicted.ndim == 4
    assert predicted.shape[-1] == 2
    assert tuple(predicted.shape[1:3]) == (
        self.bev_h, self.bev_w
    )

    call_counter["count"] += 1

    return torch.zeros_like(predicted)

MetaFlow.forward = zero_offset_forward

print("=== B0 FROZEN NOCOMP ===", flush=True)
print("Model directory:", model_dir, flush=True)
print("MetaFlow output: ZERO", flush=True)

sys.argv = [
    "opencood/tools/inference_w_delay.py",
    "--model_dir", model_dir,
    "--fusion_method", "intermediate",
    "--save_vis_interval", "1000000",
    "--note", "_B0_NoComp"
]

runpy.run_path(
    "opencood/tools/inference_w_delay.py",
    run_name="__main__"
)

print("MetaFlow intercepted calls:",
      call_counter["count"], flush=True)

assert call_counter["count"] > 0, (
    "MetaFlow was never called"
)

print("B0_INFERENCE_COMPLETE", flush=True)
