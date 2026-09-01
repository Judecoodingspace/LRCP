# LRCP V2X-Sim Stage-1 Recovery Manifest

- Upstream commit: `1eb3793b9befd9d36dd8705e461d895a41b39d33`
- Compatibility branch: `main` (current repair is an uncommitted worktree state)
- Compatibility commit: `NOT_CREATED`
- Remote push verified: `NO`; `git ls-remote` shows only upstream `main` at the frozen commit
- Stage: V2X-Sim Stage-1
- `STAGE1_COMPATIBILITY_REPAIR`: `PASS`
- `STAGE2_COMPATIBILITY`: `UNRESOLVED`
- Dataset root: `/workspace/datasets/V2X-Sim-2.0`
- PKLs: `/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_{train,val,test}.pkl`
- Offset maps: `/workspace/datasets/V2X-Sim-2.0/offset_maps` (10,000 `.npy`, 37,505,280,000 bytes)
- YAML: `opencood/hypes_yaml/v2xsim/pointpillar_deformable_attn_stage1.yaml`
- Batch size: `8`
- Operational loader setting: `num_workers=0` due `/dev/shm=64MiB`
- PPU: `PPU-ZW810E`; torch `2.6.0`; PPU SDK `1.6.1-14620b`
- Formal epoch-1 checkpoints: `opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_01_19_46_02/`

Important compatibility changes: slot-based agent semantics; agent×agent pairwise transform; delayed remote pose; remote-slot collate; history boundary guard.

This bundle is stored in `/workspace/lrcp_recovery`, which is INSTANCE_LOCAL and not an off-instance backup.
