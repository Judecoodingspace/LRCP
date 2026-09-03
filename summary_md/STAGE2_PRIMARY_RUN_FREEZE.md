# LRCP Stage-2 Primary Run Freeze

## Identity and initialization

- Frozen upstream commit: `1eb3793b9befd9d36dd8705e461d895a41b39d33`
- Current compatibility fork HEAD: `8f7cef8d1b5c92df3a05516cd5fa61f31e3b34b7`
- Primary target: compatibility-corrected, released-code-faithful reproduction; not bitwise upstream or unmodified-code reproduction.
- Stage-1 checkpoint: `/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_02_15_37_55/net_epoch_bestval_at13.pth`
- Stage-1 checkpoint SHA256: `dbd713f67122d11bc0b7606177658712e3ab729ca095623eb0e8d4a2367c7f20`

## Frozen primary training configuration

- Stage-2 optimizer: Adam, LR `0.0002`, eps `1e-10`; no configured weight decay.
- Scheduler: MultiStep, milestones `[10, 15]`, gamma `0.1`.
- Epochs: 20; batch size: 8.
- `history_frame`: 5; maximum train delay index: 3.
- Internal temporal encoding: `time_index * 100`; it must not be changed to `*200`.
- Physical V2X-Sim reporting (5 Hz): index 0/1/2/3 = 0/200/400/600 ms. Do not cite internal 100/200/300-ms labels as physical V2X-Sim milliseconds.

## Approved compatibility/runtime semantics

- Remote calibration is indexed by feature slot; padded remote slots carry a valid mask.
- Pairwise transforms are agent-to-agent; history boundaries reject samples through the existing `None` path.
- PPU compatibility changes and `DataLoader(num_workers=0, prefetch_factor=None)` are retained as runtime safeguards.

## Final evaluation plan

Use the same final Stage-2 checkpoint for both protocols and never mix their numbers:

- **C — released-code evaluation:** retain manual postprocessor filtering.
- **P — paper-style evaluation:** remove only the manual filtering identified in the released README.

Paper LR `5e-4` is a later, separately authorized sensitivity configuration only; it is excluded from the primary run.

## Preflight result

- Checkpoint exists and SHA256 matches the frozen value.
- YAML parsed as `train_stage=stage2`, Adam, LR `0.0002`, batch size 8, `history_frame=5`, `time_delay=3`.
- Stage-2 model loaded the Stage-1 checkpoint with `strict=False`: 107 missing keys, all and only `meta_flow.*`; 0 unexpected keys.
- Explicit trainability audit: 99 trainable parameters, all `meta_flow.*`; 343 detector parameters frozen (`requires_grad=False`).
- YAML diff audit: **EXPECTED CHANGE: `stage1_model` path only.** No other YAML scientific parameter changed.
- No epoch training, formal training, sensitivity run, or inference was started.

## Local path adaptation

The following five YAML changes relative to frozen released YAML are **RUNTIME/PATH ADAPTATION**, not scientific configuration changes:

- `data_dir` → `/workspace/datasets/V2X-Sim-2.0`
- `root_dir` → `/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_train.pkl`
- `validate_dir` → `/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_val.pkl`
- `test_dir` → `/workspace/datasets/v2xsim2_info_generated/v2xsim_infos_test.pkl`
- `stage1_model` → the frozen formal Stage-1 checkpoint above.

Diff audit result: these are the only five changes from frozen Stage-2 YAML. Confirmed unchanged: LR `0.0002`, epochs `20`, batch `8`, `history_frame=5`, `time_delay=3`, milestones `[10,15]`, gamma `0.1`.

## One-batch Stage-2 smoke evidence

- PPU runtime: `PPU-ZW810E`; checkpoint SHA256 matched the frozen value.
- Real Stage-2 train dataset built from the adapted train PKL: length 8000.
- Exactly one batch was collated and submitted to forward. It contained 8 valid real samples (`record_len=[2,2,2,2,2,2,2,2]`); four boundary samples were rejected through the approved existing rejection path before collation.
- Slot masks, historical lidar inputs, and target offsets were valid and finite. Model loading and the `meta_flow.*`-only trainability audit passed before forward.
- **Forward failed before `pred_offset` was produced; therefore no loss, backward, or optimizer step occurred. No second batch and no epoch loop were run.**

### Blocker B2: MetaFlow BEV-size inconsistency

Observed traceback endpoint:

```text
opencood/models/meta_flow.py:185, bev_queries = bev_queries + temporal_encoding
RuntimeError: The size of tensor a (25200) must match the size of tensor b (6400) at non-singleton dimension 1
```

Cause established by source/config evidence: `MetaFlow.__init__` hardcodes `self.bev_h=100` and `self.bev_w=252` (`opencood/models/meta_flow.py:119-120`), creating 25,200 BEV queries, while the frozen Stage-2 YAML and data pipeline configure `bev_h=80`, `bev_w=80`, producing a 6,400-location temporal encoding (`opencood/hypes_yaml/v2xsim/pointpillar_deformable_attn_stage2.yaml:12-13`; `meta_flow.py:181-185`). This pre-existing upstream Stage-2 execution defect is outside the authorized path-adaptation scope. No patch was made.

`STAGE2_PRIMARY_PREFLIGHT = BLOCKED`

Blocker: resolve the MetaFlow BEV-size configuration/implementation mismatch under a separately authorized compatibility-repair scope, then rerun this one-batch smoke from the beginning.


## B2 MetaFlow BEV config repair attempt (2026-09-03)

### Pre-change audit

- Parsed Stage-2 MetaFlow arguments: `bev_h=80`, `bev_w=80`.
- The real train batch's GT offsets had spatial shape `80×80×2`.
- `MetaFlow.encoder` produces the configured 80×80 feature grid on the previous smoke path.
- `opencood/models/meta_flow.py` contained exactly one actual 100×252 query source: the hardcoded `self.bev_h=100`, `self.bev_w=252` at lines 119–120. The only downstream `self.bev_h/self.bev_w` uses are `LearnedPositionalEncoding`, `bev_embedding`, `bev_mask`, and `spatial_shapes_self`; no independent active 100×252 assumption was found in MetaFlow.

### Authorized change

`META_FLOW_BEV_CONFIG_REPAIR` — **COMPATIBILITY / IMPLEMENTATION REPAIR**:

```python
self.bev_h = args['bev_h']
self.bev_w = args['bev_w']
```

Configured V2X-Sim BEV remains 80×80. No YAML BEV setting, voxel setting, temporal encoding, attention configuration, loss, LR, history setting, or checkpoint was changed. `python -m py_compile opencood/models/meta_flow.py` and `git diff --check` passed.

### One-batch validation result

One real Stage-2 batch (batch size 8) was rebuilt from the adapted train dataset. `record_len=[2,2,2,2,2,2,2,2]`; all valid history inputs and 80×80×2 targets passed pre-forward checks. The model used the frozen Stage-1 checkpoint and `meta_flow.*`-only trainability configuration.

The original 25,200-vs-6,400 error did not recur. However, before `pred_offset` could be produced, forward encountered this new first traceback:

```text
opencood/models/point_pillar_deformable.py:100
  lidar_history_features = self.get_batch_pillar_features(
      data_dict['calibrate_data'][cav_id]['lidar_history'], ...)
opencood/models/point_pillar_deformable.py:65
  batch_lidars = self.pillar_vfe(batch_lidars)
opencood/models/sub_modules/pillar_vfe.py:115
  batch_dict['voxel_features'], batch_dict['voxel_num_points'], ...
TypeError: 'NoneType' object is not subscriptable
```

Interpretation: a padded remote calibration slot has `lidar_history=None`, but Stage-2 model forward iterates the slot without honoring the existing `valid_mask`. This is a new slot/padding compatibility blocker, outside the authorized MetaFlow-only repair scope. No additional patch, second batch, loss, backward, optimizer step, epoch loop, or commit was performed.

`B2_META_FLOW_BEV_REPAIR = BLOCKED`

`STAGE2_PRIMARY_PREFLIGHT = BLOCKED`

Blocker: separately authorize and audit minimal Stage-2 handling of padded remote slots/`valid_mask`, then restart one-batch validation from the beginning.


## B2/B3 resolution — validated primary preflight (2026-09-03)

- **B2_META_FLOW_BEV_CONFIG_REPAIR = PASS.** The configuration-driven MetaFlow BEV dimensions were exercised by the successful Stage-2 forward: encoder, BEV queries, and temporal encoding each used the configured 80×80 / 6,400-location contract. Classification remains `META_FLOW_BEV_CONFIG_REPAIR` — compatibility / implementation repair; configured V2X-Sim BEV remains 80×80.
- **B3_PADDED_SLOT_VALID_MASK_PROPAGATION = PASS.** The repair filters `lidar_history`, `time_delay`, and pairwise lookup by the same original valid row indices, then scatters predictions back to the full B dimension. Fully padded slots bypass PillarVFE and MetaFlow and produce exact zero predictions.

### B3 mixed-mask alignment diagnostic

With `valid_mask=[True, False, True, False]`, valid histories and delays remained rows 0 and 2 (`[10.0, 30.0]`); ResNet projection consumed original pairwise rows 0 and 2; scatter yielded predicted/zero/predicted/zero for rows 0/1/2/3. No batch row was reordered.

### B3 one-real-batch smoke evidence

- Dataset: real Stage-2 train PKL, length 8000; exactly one collated mixed batch of 8 samples, `record_len=[2,2,2,2,4,4,4,4]`.
- Valid rows: slot 1 `[0,1,2,3,4,5,6,7]`; slots 2 and 3 `[4,5,6,7]`; slot 4 `[]`.
- Runtime projection recorded precisely the same original rows; no padded `None` reached PillarVFE.
- `pred_offset` shape: `(8,6400,4,2)`; finite. Each padded row/slot prediction was exactly zero; targets were finite.
- Offset loss: `0.5407617688179016`, finite. Backward passed with 99 finite `meta_flow.*` gradients and zero detector gradients. One optimizer step updated 99 MetaFlow parameters and zero detector parameters.
- No second real batch, epoch loop, formal training, inference, or sensitivity run was started.

`B2_META_FLOW_BEV_CONFIG_REPAIR = PASS`

`B3_PADDED_SLOT_VALID_MASK_PROPAGATION = PASS`

`STAGE2_PRIMARY_PREFLIGHT = PASS`

`READY_FOR_RESEARCHER_REVIEW_OF_FORMAL_RUN_AUTHORIZATION`
