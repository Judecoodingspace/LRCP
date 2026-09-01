# Decision

## Required status

```text
CODE_OFF_INSTANCE_BACKUP = PASS
EXPERIMENT_DOCS_BACKUP = PASS
ENVIRONMENT_MANIFEST_BACKUP = PASS
INFO_PKL_BACKUP = FAIL
OFFSET_MAPS_BACKUP = FAIL
RAW_V2XSIM_BACKUP = FAIL
CHECKPOINT_BACKUP = FAIL
LOG_BACKUP = PASS
PERSISTENT_STORAGE_TARGET = CONFIRMED_FOR_GITHUB_ELIGIBLE_ASSETS
INSTANCE_RELEASE_READINESS = FAIL
```

## Basis

No terminal-visible persistent volume, platform dataset storage mount, or object-storage endpoint was identified. `/workspace`, datasets, checkpoints, reports, and `/workspace/lrcp_recovery` reside on the root container overlay. The 2.2T apparent free capacity is not lifecycle evidence.

No large copy was attempted. This is intentional: copying the 171G dataset tree into another overlay directory would not create a backup. The user/platform administrator must provide a destination with explicit post-release retention and sufficient space. Plan for at least the 171G dataset plus repository, checkpoints, logs, and a 10–20% margin.

The recovery bundle is locally prepared but not backed up. The repair is uncommitted and unpushed; remote verification shows only upstream `main` at the frozen commit.

## GitHub backup completion

GitHub remote `https://github.com/Judecoodingspace/LRCP.git` now contains `main` at `e17cd28bad24b50d1d5e3c1f3f15d1f7bc0d6499`, confirmed through `git ls-remote`. It is an off-instance target for the eligible small source/documentation/manifest/patch/log assets. It is not used for datasets, PKLs, offsets, or checkpoints.
