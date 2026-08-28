#!/bin/bash
# MGNDTI 完成后自动衔接(2026-08-21): 停本地链 → 跑 3 个 CPU 任务
BASE="D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"
MG_DONE="$BASE/comprare/MGNDTI-main/code/output/bindingdb/random/seed_82/result_metrics.pt"

log() { echo "[auto] $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 1) 等 MGNDTI seed_82 完成
log "等待 MGNDTI seed_82 完成..."
until [ -f "$MG_DONE" ]; do sleep 30; done
log "MGNDTI 全部 5 种子完成"

# 2) 停本地基线链(避免本地重复跑 TransCPI/INGNN, 那些在实验室)
log "停本地基线链(run_baseline_bindingdb.sh 及其 python 子进程)"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'transformerCPI|iNGNN-DTI|MGNDTI' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
pkill -f run_baseline_bindingdb.sh 2>/dev/null
sleep 2

# 3) 三个 CPU 任务
cd "$BASE"

log "任务1/3: RF 统一指标校验"
"$PY" rf_eval_metrics.py

log "任务2/3: BioSNAP 数据集统计"
"$PY" biosnap_dataset_stats.py

log "任务3/3: RF 冷启动(unseen_drug/unseen_target, 5 种子)"
cd comprare/RF
"$PY" "$BASE/rf_coldstart.py"

log "三个 CPU 任务全部完成"
