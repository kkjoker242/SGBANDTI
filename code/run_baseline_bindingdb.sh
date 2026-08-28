#!/bin/bash
# MGNDTI / TransformerCPI / INGNN 的 BindingDB 5 种子跑 (2026-08-20)
# 本地 GPU 空闲时顺序跑, 单卡串行
BASE="D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"
LOG="$BASE/baseline_bindingdb_logs"
mkdir -p "$LOG"

log() { echo "[run] $(date '+%Y-%m-%d %H:%M:%S') $1"; }

log "START MGNDTI bindingdb (5 seeds)"
cd "$BASE/comprare/MGNDTI-main/code"
"$PY" main.py --data bindingdb --split random --seeds 42,52,62,72,82 > "$LOG/mgndti_bindingdb.log" 2>&1
echo "mgndti_bindingdb $?" >> "$LOG/status.txt"
log "END MGNDTI rc=$?"

log "START TransformerCPI bindingdb (5 seeds)"
cd "$BASE/comprare/transformerCPI-master/Human,C.elegans"
"$PY" main_glu.py --data bindingdb --seeds 42,52,62,72,82 > "$LOG/transformer_cpi_bindingdb.log" 2>&1
echo "transformer_cpi_bindingdb $?" >> "$LOG/status.txt"
log "END TransformerCPI rc=$?"

log "START INGNN bindingdb (5 seeds)"
cd "$BASE/comprare/iNGNN-DTI-master"
"$PY" main.py > "$LOG/ingnn_bindingdb.log" 2>&1
echo "ingnn_bindingdb $?" >> "$LOG/status.txt"
log "END INGNN rc=$?"

log "ALL THREE DONE — 状态见 $LOG/status.txt"
