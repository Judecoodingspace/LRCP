#!/usr/bin/env bash
set -euo pipefail
P4_C1_RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
if [[ -e "$P4_C1_RUN_DIR/baseline_validation.json" || -e "$P4_C1_RUN_DIR/counterfactual_metrics.json" ]]; then
  echo 'Refuse overwrite/restart: copy code, inputs.json, protocol and selection CSVs to a fresh P4_C1 directory.' >&2
  exit 64
fi
mkdir -p "$P4_C1_RUN_DIR/baseline" "$P4_C1_RUN_DIR/counterfactual" "$P4_C1_RUN_DIR/figures"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/verify_inputs.py"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/run_baseline.py"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/run_counterfactual.py"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/validate_c1.py"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/make_evidence_and_figures.py"
/workspace/venvs/lrcp/bin/python -B "$P4_C1_RUN_DIR/write_report.py"
