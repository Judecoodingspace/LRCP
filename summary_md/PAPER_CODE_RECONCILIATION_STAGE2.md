# LRCP Stage-2 Paper–Code Reconciliation

**Audit target.** WACV 2025 accepted paper and CVF supplemental; frozen upstream commit `1eb3793b9befd9d36dd8705e461d895a41b39d33`; current reproduction HEAD `8f7cef8d1b5c92df3a05516cd5fa61f31e3b34b7`. This is a read-only review: no source, YAML, model, checkpoint, training, inference, or offset map was changed.

**Evidence conventions.** “Paper §4.2/§4.3” is the accepted CVF paper. “Supplement Table 6” is the CVF supplemental's V2X-Sim table. Code references are immutable paths and line numbers at the audited commits. The frozen/current Stage-2 YAML content is identical; current-fork differences cited below were confirmed with `git diff frozen..HEAD`.

| Item | Paper | Supplement | Frozen Code | Current Fork | Classification | Materiality | Evidence | Decision Needed |
|---|---|---|---|---|---|---|---|---|
| V2X-Sim frequency and reported latency points | 5 Hz, hence 200 ms/frame; evaluates V2X-Sim through 600 ms (§4.3). | Table 6 reports V2X-Sim at 0, 200, 400, 600 ms. | Inference iterates delay indices 0..5 and names them `index*100` ms. | Same. | SCIENTIFIC_CONFIGURATION_DISCREPANCY | HIGH | `opencood/tools/inference_w_delay.py:113-131`; Paper §4.3; Supp. Table 6. | Yes—freeze paper-faithful 0/200/400/600 indexing or code-faithful 0..500 ms before formal Stage-2 evaluation. |
| Training delay range and latency encoding | Paper specifies non-discrete delay concept but not the training draw/range. | No contrary setting found. | YAML max train delay=3; random integer 0..3; input time encoded as `index*100`. | Same. | PAPER_SILENT_CODE_DEFINED | MEDIUM | `...stage2.yaml:10,14`; `v2xsim_basedataset.py:75-88,272-296,231`; `meta_flow.py:16-54,181-185`. | Yes—record whether the formal run follows the code's 0/100/200/300 time units despite the paper's 5-Hz statement. |
| Dataset split paths | Paper names V2X-Sim but does not enumerate PKL artifacts. | No contrary setting found. | Separate train/val/test paths configured. | Same Stage-2 YAML. | PAPER_SILENT_CODE_DEFINED | LOW | `...stage2.yaml:2-5`. | No, but pin PKL hashes in the run manifest. |
| Ego policy | Random ego in training; fixed ego at testing (§4.2). | No contrary setting found. | Train permutes CAV IDs; evaluation uses ordered 1..N, making CAV1 ego. | Same core policy; prior compatibility validation confirmed CAV1 slot 0. | ALIGNED | NONE | `v2xsim_basedataset.py:119-139`; Paper §4.2. | No. |
| History length | Paper ablates cached frames and selects k=5 default. | Consistent tables/appendix evidence. | `history_frame=5`, MetaFlow `n_frame=5`. | Same. | ALIGNED | NONE | Paper §4.3/Fig.7; `...stage2.yaml:14,99-112`. | No. |
| Stage-1 calibration target | Stage 1 uses GT flow for reference-point calibration. | No contrary setting found. | Stage 1 places GT offsets in batch and decoder uses GT offsets. | Same. | ALIGNED | NONE | `point_pillar_deformable.py:90-115`; `defor_encoder_multi_scale.py` frozen equivalent calibration branch. | No. |
| Stage-2 initialization | Stage 2 learns the flow model after detection training. | No contrary setting found. | Loads configured Stage-1 checkpoint with `strict=False`. | Same; configured historical path must be replaced by a researcher-selected reproduced Stage-1 artifact for an actual run. | PAPER_SILENT_CODE_DEFINED | MEDIUM | `...stage2.yaml:8-10`; `train_w_delay.py:83-95`. | Yes—freeze exact Stage-1 checkpoint SHA/path. |
| Stage-2 trainable/frozen modules and BN | Detection components frozen; flow module trained. | No contrary setting found. | Only `meta_flow.*` has gradients; `model.eval()` then `meta_flow.train()` keeps detector BN/eval while MetaFlow BN updates. | Same. | ALIGNED | NONE | `train_w_delay.py:83-95,140-160`. | No. |
| Stage-2 objective | Weighted SmoothL1 flow loss. | No contrary setting found. | Returns offset loss only in Stage 2; detection and single-view detection losses are bypassed. | Same. | ALIGNED | NONE | Paper §4.2; `point_pillar_loss.py:70-83`; `train_w_delay.py:151-160`. | No. |
| Flow foreground/background weights | 1 / 0.005 (§4.2). | No contrary setting found. | Exact per-coordinate zero test assigns .005, else 1. | Same. | ALIGNED | LOW | Paper §4.2; `point_pillar_loss.py:72-79`. | No; document the code's per-coordinate (not explicit per-object/pixel) interpretation. |
| PointPillar/ResNet, voxel and range | PointPillar + ResNet; 0.4 m; V2X-Sim ±32 m × ±32 m (§4.2). | No contrary setting found. | PointPillar, ResNet, voxel [0.4,0.4,5], range [-32,-32,...,32,32,...]. | Same. | ALIGNED | NONE | `...stage2.yaml:46-53,91,128-136`; Paper §4.2. | No. |
| Fusion decoder | 4 deformable cross-attention + 2 self-attention; 8 heads/8 points (§4.2). | Table 4 supports 4 cross + 2 self default. | Six configured blocks, first four cross and last two self; each 8 heads/8 points. | Same. | ALIGNED | NONE | `...stage2.yaml:137-154`; Paper §4.2; Supp. Table 4. | No. |
| MetaFlow architecture | 3 cross + 2 self, 8 heads/8 points; cached history. | No contrary setting found. | 3 cross then 2 self blocks, 8 heads/8 points, n_frame=5, multi-scale [64,128,256]. | Same. | ALIGNED | NONE | `...stage2.yaml:95-112`; `meta_flow.py:113-207`; Paper §4.2. | No. |
| Separate single-agent head | Separate head supervises each agent (§4.2). | Supp. Table 5 ablates single supervision. | Separate head enabled; Stage 1 adds its loss, Stage 2 does not. | Same. | ALIGNED | NONE | `...stage2.yaml:113-114`; `point_pillar_deformable.py:136-153`; `train_w_delay.py:154-156`. | No. |
| Optimizer/LR | Stage 1 LR 0.002; Stage 2 flow LR 5e-4 (§4.2). | No explanation of a lower code LR found. | Adam, stage 1 .002, stage 2 .0002. | Same Stage-2 YAML. | SCIENTIFIC_CONFIGURATION_DISCREPANCY | HIGH | Paper §4.2; `stage1.yaml:191-210`; `stage2.yaml:203-214`; `git log --all -- stage2.yaml` contains no explanatory revision. | Yes—choose and record 5e-4 (paper-faithful) or 2e-4 (released-code-faithful). |
| Epochs/scheduler/batch/weight decay | Paper does not give these. | No contrary setting found. | batch 8, 20 epochs, multistep [10,15], gamma .1; stage 1 decay 1e-4; stage 2 decay omitted. | Same. | PAPER_SILENT_CODE_DEFINED | LOW | `stage1.yaml:24-29,191-210`; `stage2.yaml:25-30,203-214`. | No, but freeze in experiment contract. |
| AP and IoU | AP at IoU .5 and .7 (§4.2). | Tables report AP@.5/.7. | Computes .3/.5/.7; published-relevant values are .5/.7. | Same. | ALIGNED | NONE | Paper §4.2; Supp. Table 6; `inference_w_delay.py:127-192`. | No. |
| Checkpoint selection | Paper is silent. | No contrary setting found. | Saves each scheduled epoch and selects lowest aggregate validation loss as `bestval`; inference uses user-selected/resumed checkpoint. | Same. | PAPER_SILENT_CODE_DEFINED | MEDIUM | `train_w_delay.py:164-205`; `inference_w_delay.py:87-105`. | Yes—state exact checkpoint-selection rule in formal protocol. |
| Postprocess: score/NMS/range | Paper is silent. | No contrary setting found. | score .20, NMS .15, range filtering. | Same. | PAPER_SILENT_CODE_DEFINED | LOW | `stage2.yaml:65-87`; `voxel_postprocessor.py:379-400`. | No, but preserve values. |
| Manual postprocessor filters and reported AP | Paper reports its results but does not describe this filter. | No contrary setting found. | Filters large boxes and abnormal-z boxes before NMS. Frozen README says removing these filters gives the slightly better results reported in the paper, while retaining filters improves consistency with other papers. | Same. | SCIENTIFIC_CONFIGURATION_DISCREPANCY | HIGH | `voxel_postprocessor.py:371-384`; frozen `README.md:85-90`. | Yes—paper-faithful AP requires removal; released-code-faithful evaluation retains them. Do not mix the two. |
| Visualization/output options | Paper visualizes results. | No contrary setting found. | Optional NPY/BEV visualization writes outputs after AP inputs are formed. | Same except worker count. | REPORTING_ONLY_DISCREPANCY | NONE | `inference_w_delay.py:193-237`. | No. |
| GT-flow construction | Paper describes object-ID tracks, transform between boxes, and zero flow for points outside boxes. | No contrary setting found. | Tracks `gt_object_ids`, projects world boxes into current ego coordinates, derives 2-D rigid transform from box center/yaw, rasterizes transformed points; initialized maps are zero. | Same logic; only local input paths changed. | ALIGNED | LOW | Paper §4.1; `flow_label_gen_v2x-smi.py:97-104,137-177,222-238,240-294,297-329`. | No; retain generator/version and maps as immutable inputs. |
| Current-fork DataLoader change | Paper/supplement silent. | Silent. | Four workers and prefetch factor 2. | Zero workers and prefetch factor None; pin_memory unchanged. | RUNTIME_ONLY_DISCREPANCY | NONE | `git diff frozen..HEAD -- opencood/tools/{train,inference}_w_delay.py`. | No. |
| Current-fork import/operator compatibility | Paper/supplement silent. | Silent. | Imports positional encoding from MMDetection. | Uses repository-local equivalent; PPU operator/runtime edits are outside scientific YAML. | RUNTIME_ONLY_DISCREPANCY | LOW | `git diff frozen..HEAD -- opencood/models/meta_flow.py opencood/models/sub_modules/*encoder*.py`. | No, subject to completed forward/backward smoke tests. |
| Current-fork slot/pairwise/history repair | Paper requires feature/flow correspondence but does not specify padding-slot implementation. | Silent. | Calibration storage is keyed by physical CAV buckets and pairwise history shape is inconsistent with agent-slot consumption; boundary `None` crashes. | Remote calibration is slot-indexed with valid masks; pairwise matrix is (max_cav,max_cav); only valid `N-1` remote slots are consumed; boundary returns existing rejected-sample `None`. | SCIENTIFIC_CONFIGURATION_DISCREPANCY | HIGH | `git diff frozen..HEAD -- intermediate_latency_fusion_dataset.py transformation_utils.py defor_encoder_multi_scale.py v2xsim_basedataset.py`; current `defor_encoder_multi_scale.py:304-318`. | Yes—formal Stage-2 must declare compatibility-corrected data/slot semantics, not claim bitwise frozen-code reproduction. |

## Material discrepancies requiring a researcher decision

### 1. V2X-Sim latency protocol and units — HIGH

The paper and supplement report V2X-Sim at 0/200/400/600 ms because the dataset is 5 Hz. The released inference loop instead creates six settings indexed 0..5 and labels them 0/100/200/300/400/500 ms. This changes both latency coordinates and the terminal condition; it can change the reported curve and makes a numerical comparison to the paper's V2X-Sim table invalid without a declared mapping.

- **Stage 1:** does not alter completed Stage-1 architecture/optimizer, but its delay conditioning uses the same `index*100` code unit.
- **Stage 2:** directly affects flow time encoding and final evaluation labels/protocol.
- **Paper-faithful:** 5-Hz / 200-ms V2X-Sim framing and reported 0/200/400/600-ms evaluation.
- **Code-faithful:** integer delay index with `*100` code units, evaluation indices 0..5 / labels 0..500 ms.
- **Decision:** required before formal training/evaluation; no automatic conversion is justified by evidence.

### 2. Stage-2 learning rate — HIGH

The paper states 5e-4; frozen and current YAML specify 2e-4. The upstream history available locally contains no later config explanation. LR is a direct optimization setting and can change convergence and AP.

- **Stage 1:** not affected (its YAML remains .002).
- **Stage 2:** directly affected.
- **Paper-faithful:** Adam flow LR 5e-4.
- **Code-faithful:** Adam flow LR 2e-4 with released scheduler.
- **Decision:** required; select one and preserve the other as a documented comparator rather than silently editing it.

### 3. Manual filtering vs reported AP — HIGH

The active released postprocessor removes large and abnormal-z boxes before NMS. The frozen README explicitly says removing that manual filtering yields the slightly better results reported in the paper. This is therefore an evaluation configuration difference that can affect reported AP.

- **Stage 1:** affects Stage-1 AP evaluation, not learned weights.
- **Stage 2:** affects final AP evaluation, not flow-loss optimization.
- **Paper-faithful:** remove filters for paper-result protocol.
- **Code-faithful:** retain active filters for released-code protocol.
- **Decision:** required before claiming numerical reproduction. Do not report results from one protocol as the other.

### 4. Compatibility-corrected slot/pairwise semantics — HIGH

The current fork makes remote calibration tensors follow each sample's feature-slot order, masks padded remote slots, makes pairwise transforms agent-to-agent, and propagates documented history rejection. These repairs make heterogeneous batches and PPU execution work, but they are not bitwise behavior of frozen implementation and can change which offset/pose a remote feature consumes.

- **Stage 1:** the same slot and GT-offset route was exercised by completed Stage-1 compatibility gates; results are not bitwise frozen-code results.
- **Stage 2:** directly affects predicted-flow input/target correspondence and calibrated reference points.
- **Paper-faithful:** semantically matched remote feature, pose, and flow slot; paper does not prescribe padding implementation.
- **Code-faithful:** frozen physical-CAV bucket logic and its shape/ordering behavior.
- **Decision:** required to label formal run accurately as a compatibility-corrected reproduction rather than an unmodified upstream reproduction.

## Non-blocking provenance notes

- Source code does not specify exact original Stage-1 checkpoint contents or data/offset-map hashes. The formal run must pin checkpoint SHA and PKL/offset-map hashes.
- Optional visualization and AP@.3 computation do not change AP@.5/.7 inputs; they are not scientific discrepancies for paper tables.
- The fork's `num_workers=0`/`prefetch_factor=None` changes are runtime throughput safeguards and are not method changes.

## Required freeze decisions before a formal Stage-2 run

1. Choose one latency protocol/label mapping.
2. Choose Stage-2 LR: paper 5e-4 or released code 2e-4.
3. Choose paper-result postprocessing or released-code postprocessing.
4. Approve compatibility-corrected slot/pairwise semantics and exact Stage-1 initialization checkpoint.

## Researcher Freeze Decisions

- **D1 — primary target:** compatibility-corrected, released-code-faithful reproduction. It is neither a bitwise upstream reproduction nor an exact unmodified-code reproduction.
- **D2 — Stage-2 LR:** primary LR is **2e-4**, retaining the frozen released Stage-2 YAML baseline. Paper LR 5e-4 is a paper-faithful sensitivity configuration only and is not authorized in this run.
- **D3 — temporal semantics:** retain the released internal MetaFlow encoding `time_index * 100`. Physical V2X-Sim reporting is 5-Hz: index 0/1/2/3 means 0/200/400/600 ms. Internal log labels 100/200/300 ms are not physical V2X-Sim milliseconds for paper comparison.
- **D4 — compatibility corrections:** approved and retained: slot-indexed remote calibration, padded-slot valid mask, agent-to-agent pairwise transform, history-boundary rejection, PPU changes, and `num_workers=0` safeguard.
- **D5 — postprocessing:** Stage-2 training is not blocked. The final checkpoint must be evaluated twice with the *same checkpoint*: C retains manual filtering; P removes only the README-identified manual filtering. Results must remain separate.
- **D6 — initialization:** use `/workspace/LRCP/opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_02_15_37_55/net_epoch_bestval_at13.pth`; SHA256 `dbd713f67122d11bc0b7606177658712e3ab729ca095623eb0e8d4a2367c7f20`.

`STAGE2_RECONCILIATION_STATUS = FROZEN_FOR_PRIMARY_CODE_FAITHFUL_PREFLIGHT`
