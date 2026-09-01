# Decision

## Required status

```text
CODE_OFF_INSTANCE_BACKUP = FAIL
EXPERIMENT_DOCS_BACKUP = FAIL
ENVIRONMENT_MANIFEST_BACKUP = FAIL
INFO_PKL_BACKUP = FAIL
OFFSET_MAPS_BACKUP = FAIL
RAW_V2XSIM_BACKUP = FAIL
CHECKPOINT_BACKUP = FAIL
LOG_BACKUP = FAIL
PERSISTENT_STORAGE_TARGET = NEEDS_PLATFORM_UI_CONFIRMATION
INSTANCE_RELEASE_READINESS = FAIL
```

## Basis

No terminal-visible persistent volume, platform dataset storage mount, or object-storage endpoint was identified. `/workspace`, datasets, checkpoints, reports, and `/workspace/lrcp_recovery` reside on the root container overlay. The 2.2T apparent free capacity is not lifecycle evidence.

No large copy was attempted. This is intentional: copying the 171G dataset tree into another overlay directory would not create a backup. The user/platform administrator must provide a destination with explicit post-release retention and sufficient space. Plan for at least the 171G dataset plus repository, checkpoints, logs, and a 10–20% margin.

The recovery bundle is locally prepared but not backed up. The repair is uncommitted and unpushed; remote verification shows only upstream `main` at the frozen commit.
