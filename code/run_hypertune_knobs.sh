#!/bin/bash
# 超参微调批次5 自动链 (2026-08-19): bs32 → cosine → maxepoch250, 单种子42, 验证集选优
cd "D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"
CHAIN_LOG="hypertune_knobs_chain.log"

log() { echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$CHAIN_LOG"; }

run_job() {
  local name="$1"; shift
  log "START $name (extra args: $*)"
  "$PY" main.py --data biosnap --split random --hop 2 --seeds 42 "$@" --tag "$name" > "hypertune_${name}.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  echo "$name $rc" >> hypertune_status.txt
}

log "KNOBS QUEUE START"
run_job tune_bs32 --batch-size 32
run_job tune_cosine --lr 1e-4 --cosine
run_job tune_maxepoch250 --max-epoch 250
log "KNOBS JOBS DONE — generating report"
"$PY" make_tune_report.py
log "KNOBS REPORT WRITTEN — tune_report.md"
