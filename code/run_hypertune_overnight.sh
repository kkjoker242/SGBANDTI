#!/bin/bash
# 超参微调批次1+2 自动链 (2026-08-18)
# 三屏单种子42: baseline → dropout0.2+wd1e-4 → heads4
# 全部验证集选优; 报告后停; 批次3(5种子确认 ~13h)留人工
cd "D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"
CHAIN_LOG="hypertune_chain.log"

log() { echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$CHAIN_LOG"; }

run_job() {
  local name="$1"; shift
  log "START $name (extra args: $*)"
  "$PY" main.py --data biosnap --split random --hop 2 --seeds 42 "$@" --tag "$name" > "hypertune_${name}.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  echo "$name $rc" >> hypertune_status.txt
  if [ $rc -ne 0 ]; then
    log "FAILED $name — see hypertune_${name}.log"
  fi
}

log "QUEUE START"
run_job tune_baseline
run_job tune_dropout02_wd1e4 --dropout 0.2 --weight-decay 1e-4
run_job tune_heads4 --heads 4
log "ALL JOBS DONE — generating report"
"$PY" make_tune_report.py
log "REPORT WRITTEN — tune_report.md"
