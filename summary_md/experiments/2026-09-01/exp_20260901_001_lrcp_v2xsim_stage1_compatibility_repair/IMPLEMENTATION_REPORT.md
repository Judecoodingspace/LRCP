# Implementation report

## Classification

This is a **compatibility repair** for V2X-Sim Stage-1 only.

## Changes

- `get_pairwise_transformation_with_history` now returns `(max_cav, max_cav, 4, 4)`, preserves input slot order, uses the current ego pose and a remote delayed-observation `params_single['lidar_pose']` when available, and retains the existing i-to-j transform convention.
- Latency-fusion collation now uses remote slots 1–4 from each sample's `cav_id_list`, pads absent slots with zero offsets/delay and `valid_mask=False`, and exposes physical CAV IDs only as diagnostic metadata.
- The multi-scale deformable encoder consumes only `:N-1` remote offsets for each sample.

## Boundary

`STAGE2_COMPATIBILITY = UNRESOLVED`. No Stage-2 execution or claim is included.

## HISTORY_BOUNDARY_GUARD

The authorized minimal guard was added in `V2XSIMBaseDataset.retrieve_base_data` immediately after `frame_select(scene)`. When either return value is `None`, it returns `None` to the existing `__getitem__`/collate rejection path. This is not a new sampling method: it does not alter `frame_select`, its retry count, delay/history settings, offsets, or PKL files.
