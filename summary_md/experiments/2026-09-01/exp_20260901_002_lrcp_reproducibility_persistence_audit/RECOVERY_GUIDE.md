# Recovery Guide

## Current recovery status

This guide documents recovery inputs but does **not** establish off-instance recovery because no persistent destination has been confirmed. If the instance is destroyed now, all `/workspace` assets listed below may be lost.

## Code

- Upstream remote: `https://github.com/junjie-wang-dev/LRCP.git`
- Frozen upstream commit: `1eb3793b9befd9d36dd8705e461d895a41b39d33`
- Current branch: `main`
- Remote-visible commit: only the frozen upstream `main` commit.
- The compatibility repair is an uncommitted worktree state. Local insurance patches are `/workspace/lrcp_recovery/lrcp_stage1_compatibility.patch` and `full_tracked_worktree.patch`; they are not off-instance backups.

To recover after a successful future backup: clone the remote, checkout the recorded commit, retrieve the saved worktree patch from the confirmed persistent destination, review/apply it, and verify `git diff --check` plus the Stage-1 invariant scripts/reports.

## Data and generated assets

Restore these exact paths beneath the new instance's `/workspace/datasets`:

- `V2X-Sim-2.0/` raw data
- `v2xsim2_info_generated/v2xsim_infos_train.pkl`
- `v2xsim2_info_generated/v2xsim_infos_val.pkl`
- `v2xsim2_info_generated/v2xsim_infos_test.pkl`
- `V2X-Sim-2.0/offset_maps/` as the original `.npy` directory tree

Validate PKLs with `sha256sum -c SHA256SUMS_info_pkls.txt`; validate offset-map count (10,000), byte total (37,505,280,000), and deterministic sample hashes from `SHA256SUMS_offset_maps_sample.txt`.

## Environment

Use the recovery bundle files: `python_version.txt`, `pip_freeze.txt`, `torch_env.txt`, `ppu_sdk_release.yaml`, and `ppu_smi.txt`. Recorded environment: Python 3.10.13, torch 2.6.0, CUDA interface 12.6, PPU SDK 1.6.1-14620b, PPU-ZW810E. `pip_freeze.txt` does not recreate the PPU SDK by itself.

## Training and checkpoint placement

YAML: `opencood/hypes_yaml/v2xsim/pointpillar_deformable_attn_stage1.yaml`; batch size 8; operational loader setting `num_workers=0` due a 64MiB `/dev/shm` constraint. Existing epoch-1 checkpoints belong in `opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_01_19_46_02/`: `net_epoch1.pth` and `net_epoch_bestval_at1.pth`. Verify their hashes with `SHA256SUMS_checkpoints.txt`.

Stage status: `STAGE1_COMPATIBILITY_REPAIR = PASS`; `STAGE2_COMPATIBILITY = UNRESOLVED`.
