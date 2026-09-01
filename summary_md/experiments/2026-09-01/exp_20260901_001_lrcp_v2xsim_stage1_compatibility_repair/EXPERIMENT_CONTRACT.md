# LRCP / V2X-Sim Stage-1 compatibility repair

## Scope

This experiment repairs public V2X-Sim **Stage-1** implementation compatibility only. It is not a new method, algorithm improvement, performance optimization, or a complete training run.

## Existing evidence

The frozen audit established three incompatible public-code behaviors: the history-aware pairwise matrix used a time axis where downstream code requires an agent-slot axis; calibration tensors were bucketed by physical CAV ID although features use randomized sample slots; and those physical-ID buckets create incompatible variable-length batch tensors.

## Semantic contract

For every sample, slot 0 is ego and slots 1 through `N - 1` are remotes in `cav_id_list` order. Physical CAV IDs are lookup and diagnostic metadata only. Feature, offset, pose, pairwise transform, history, and delay must share this slot order. Pairwise transforms have shape `(max_cav, max_cav, 4, 4)` with both leading dimensions denoting slots. A remote pairwise pose preferentially uses its delayed-observation `params_single['lidar_pose']`.

## Allowed changes

Only the Stage-1 compatibility path may be changed: `get_pairwise_transformation_with_history`, latency-fusion calibration collation, and Stage-1 effective remote-slot consumption. Diagnostics/tests and experiment documentation are allowed.

## Forbidden changes

Do not alter time delay, history-frame count, GT flow or offset-map contents, BEV geometry, voxel settings, model/deformable-attention architecture, loss, optimizer, learning rate, augmentation, anchors, epochs, evaluation thresholds, or Stage-1/Stage-2 scientific definitions.

## Stage-2 boundary

`STAGE2_COMPATIBILITY = UNRESOLVED`. Stage-2 execution is blocked; no claim is made that its MetaFlow history/padding semantics are repaired.

## Acceptance gates

The repair must preserve scope, produce finite `(B, 5, 5, 4, 4)` pairwise tensors, maintain slot alignment, construct every remote-slot tensor with batch dimension `B`, validate non-special validation samples, pass real PPU forwards for batch 1 and heterogeneous batch 8, and complete exactly one Stage-1 training smoke step. Only then may `STAGE1_COMPATIBILITY_REPAIR = PASS` be reported.
