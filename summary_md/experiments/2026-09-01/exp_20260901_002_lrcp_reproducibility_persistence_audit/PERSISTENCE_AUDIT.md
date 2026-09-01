# Reproducibility & Persistence Audit

## Scope and evidence

This is a reproducibility and persistence audit, not a training or algorithm-change task. Topology evidence is saved locally at `/workspace/lrcp_recovery/storage_topology.txt`.

`/workspace` and `/workspace/datasets` have no separate `findmnt` entry. They resolve to the container root `overlay` filesystem, whose upper directory is under a containerd snapshot path. This is evidence of instance-local container storage, not evidence of an independently retained volume.

| Classification | Paths | Evidence | Conclusion |
|---|---|---|---|
| INSTANCE_LOCAL / EPHEMERAL | `/workspace/LRCP`, `/workspace/datasets`, `/workspace/lrcp_recovery` | Root `overlay`; no independent mount at `/workspace` or `/workspace/datasets` | Expected to disappear with the instance/container lifecycle. |
| PERSISTENT_VOLUME | None confirmed | No independent persistent mount is visible | Not found from terminal evidence. |
| PLATFORM_DATASET_STORAGE | None confirmed | No platform-specific mount/object-store endpoint is visible | Not proven. |
| UNKNOWN | Any platform UI-managed disk or dataset service not mounted in this container | Terminal cannot inspect platform lifecycle policy | Requires platform UI confirmation. |

Observed capacity is 3.4T root overlay with 2.2T free. This is capacity only, not persistence evidence. `/dev/shm` is a separate 64MiB tmpfs and is unrelated to artifact preservation.

## Required platform confirmation

`PERSISTENT_STORAGE_TARGET = NEEDS_PLATFORM_UI_CONFIRMATION`.

In the compute-platform UI, identify a volume, dataset storage, or object-storage location that explicitly survives instance release. Confirm its mount/path, retention policy, and available capacity. Do not treat another directory beneath `/workspace` as a backup target.

## Local recovery staging

`/workspace/lrcp_recovery` contains small manifests, environment files, hashes, and patches. It is useful for a copy operation after a target is confirmed, but is itself INSTANCE_LOCAL and is **not** an off-instance backup.
