# Asset Inventory

All source paths below are currently INSTANCE_LOCAL because they are beneath `/workspace` on the root overlay filesystem. No persistent destination has been confirmed.

| Asset | Source path | Size / count | Preservation form | Persistent destination | Integrity proof | Status |
|---|---|---:|---|---|---|---|
| LRCP source + repair | `/workspace/LRCP` | 327M | Git branch/commit plus patch | GitHub `Judecoodingspace/LRCP` main | remote commit SHA | PASS |
| Experiment docs | `summary_md/experiments/2026-09-01/exp_20260901_001_*` | reports, backups, log | Markdown tracked by intended Git workflow | GitHub `Judecoodingspace/LRCP` main | remote commit SHA | PASS |
| Python environment | `/workspace/venvs/lrcp` | not copied | `pip_freeze.txt`, Python/torch text | GitHub `Judecoodingspace/LRCP` main | remote commit SHA + bundle SHA256 | PASS |
| PPU environment | `/usr/local/PPU_SDK` | not copied | SDK release + PPU-SMI text | GitHub `Judecoodingspace/LRCP` main | remote commit SHA + bundle SHA256 | PASS |
| train PKL | `datasets/v2xsim2_info_generated/v2xsim_infos_train.pkl` | 234,088,928 bytes | original `.pkl` | None confirmed | SHA256 recorded | FAIL |
| val PKL | `datasets/v2xsim2_info_generated/v2xsim_infos_val.pkl` | 23,086,686 bytes | original `.pkl` | None confirmed | SHA256 recorded | FAIL |
| test PKL | `datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl` | 20,017,686 bytes | original `.pkl` | None confirmed | SHA256 recorded | FAIL |
| offset maps | `datasets/V2X-Sim-2.0/offset_maps` | 10,000 `.npy`; 37,505,280,000 bytes (~35G) | original directory tree | None confirmed | count, byte total, 5 deterministic SHA256 samples | FAIL |
| Raw V2X-Sim | `datasets/V2X-Sim-2.0` | 126G; 152,335 files total | original directory tree | None confirmed | size/file count | FAIL |
| Stage-1 checkpoints | `opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_01_19_46_02/` | `net_epoch1.pth` 68,286,099 bytes; bestval 68,291,430 bytes | native `.pth` | None confirmed | SHA256 recorded | FAIL |
| Training logs | same run directory and experiment log | TensorBoard events + 232,474-byte text log | original text/events | GitHub `Judecoodingspace/LRCP` main | remote commit SHA | PASS |

Notes: `FORMAL_STAGE1_CHECKPOINT = CREATED` (epoch 1), not a smoke-test artifact. The formal run log records validation loss 0.770299 and completion. Stage-2 remains unresolved.
