# GitHub Remote Backup Verification

- Remote: `https://github.com/Judecoodingspace/LRCP.git`
- Remote branch: `main`
- Verified on: 2026-09-01
- Commit: `e17cd28bad24b50d1d5e3c1f3f15d1f7bc0d6499`
- Verification command: `git ls-remote --heads persistence-backup main`
- Result: remote `main` exactly matched the commit above.

The pushed commit includes source/worktree changes, Stage-1 compatibility-repair documentation, persistence-audit documentation, local recovery manifests/patches, environment text manifests, and small TensorBoard/text logs. It intentionally excludes raw V2X-Sim data, generated PKLs, offset maps, and native `.pth` checkpoints.
