#!/bin/bash
# 超参微调批次4 自动链 (2026-08-19): lr1e-4 → hidden256, 单种子42, 验证集选优
cd "D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"
CHAIN_LOG="hypertune_batch4_chain.log"

log() { echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$CHAIN_LOG"; }

run_job() {
  local name="$1"; shift
  log "START $name (extra args: $*)"
  "$PY" main.py --data biosnap --split random --hop 2 --seeds 42 "$@" --tag "$name" > "hypertune_${name}.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  echo "$name $rc" >> hypertune_status.txt
}

log "BATCH4 QUEUE START"
run_job tune_lr1e4 --lr 1e-4
run_job tune_hidden256 --hidden 256
log "BATCH4 JOBS DONE — generating report"
"$PY" make_tune_report.py
log "BATCH4 REPORT WRITTEN — tune_report.md"
