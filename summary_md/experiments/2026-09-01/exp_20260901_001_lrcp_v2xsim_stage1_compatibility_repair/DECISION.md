# Decision

## Status

`STAGE1_COMPATIBILITY_REPAIR = FAIL`

The code-level compatibility repair is implemented. The authorized HISTORY_BOUNDARY_GUARD resolved the pre-collation `KeyError: None`, and the real train batch=8 slot/batch invariant now passes. Validation coverage and both required real PPU forwards now pass. Full acceptance remains incomplete because the one-batch Stage-1 loss/backward/optimizer smoke has not yet run. Under the task contract, full training must remain stopped.

`STAGE2_COMPATIBILITY = UNRESOLVED`

## Interpretation boundary

The public V2X-Sim Stage-1 implementation has been given an evidence-based, slot-consistent compatibility interpretation aligned with the published LRCP pipeline. Exact identity with authors' private/internal training code cannot be established from the public repository alone.
