# Validation report

## Completed

- Python syntax compilation passed for all three changed files.
- Pairwise diagnostic passed: shape `(5, 5, 4, 4)`, finite values, ego in slot 0, and remote transforms derived from delayed `params_single` poses.
- `git diff --check` was run after the repair; no repair-introduced whitespace error remains.

## Train batch invariant (after HISTORY_BOUNDARY_GUARD)

After sourcing `/usr/local/PPU_SDK/envsetup.sh`, a real shuffled train batch of eight valid samples passed the slot/batch invariant on the first attempt. `record_len = [2, 5, 4, 4, 4, 3, 5, 3]`; pairwise shape was `(8, 5, 5, 4, 4)`; every remote-slot offset shape was `(8, 80, 80, 2)`; merged offsets had shape `(8, 6400, 4, 2)`; and each slot's `valid_mask` exactly matched `remote_slot < record_len`. The prior `KeyError: None` did not recur.

## Blocked / not passed

The history-boundary failure is resolved by the authorized guard and the real train batch=8 invariant now passes. PPU Stage-1 forward (batch 1 / batch 8), validation invariant coverage, and one-batch backward/optimizer smoke have **not** yet been run. The PPU environment is now reproducible after sourcing `/usr/local/PPU_SDK/envsetup.sh`, but this alone is not a forward-pass acceptance result. No full training was started.

`STAGE1_COMPATIBILITY_REPAIR = FAIL` pending completion of Gates D–G in a working data/PPU environment.

## M1–M3 acceptance

- `VALIDATION_SLOT_INVARIANT = PASS`: real validation samples at indices 7, 107, and 207 covered 3, 4, and 5 agents. In each case ego was CAV1 in slot 0; remote physical-ID metadata followed `cav_id_list`; CAV1 was absent from remote slots; CAV5 occupied its correct remote slot when present; pairwise had shape `(1, 5, 5, 4, 4)`; offsets had shape `(1, 80, 80, 2)` for every remote slot; and valid masks matched `remote_slot < record_len`.
- `PPU_STAGE1_BATCH1_FORWARD = PASS`: after the prescribed PPU setup, a real 5-agent validation sample (index 207, order `[1,2,3,4,5]`) completed no-grad Stage-1 forward on `cuda:0` / `PPU-ZW810E`. `cls_preds`, `reg_preds`, `cls_preds_single`, `reg_preds_single`, `dir_preds`, and `dir_preds_single` were all finite.
- `PPU_STAGE1_HETERO_BATCH8_FORWARD = PASS`: a real shuffled train batch with `record_len=[2,5,4,4,4,3,5,3]` passed structural checks—pairwise `(8,5,5,4,4)`, all slot offsets `(8,80,80,2)`, merged offsets `(8,6400,4,2)`—and completed no-grad PPU forward with all six required output tensors finite.

No AP evaluation, backward pass, optimizer step, or full training was run in this phase.
