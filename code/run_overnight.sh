#!/bin/bash
# 整夜自动执行链：unseen_drug(单种子，跑中) → unseen_target(单种子) → BindingDB SGBANDTI(单种子)
cd "D:/vs word/SGBANDTI-main/SGBANDTI-main"
PY="C:/Users/Jack/Miniconda3/envs/sgbandti/python.exe"

echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') start. waiting for unseen_drug single seed (seed42)..."

# 1) 等当前 unseen_drug 单种子完成（最长等 8h，每 60s 查一次结果文件）
for i in $(seq 1 480); do
  if [ -f "result/biosnap_unseen_drug_hop2/seed_42/result_metrics.pt" ]; then
    echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') unseen_drug single-seed DONE (result_metrics.pt found)"
    break
  fi
  sleep 60
done
if [ ! -f "result/biosnap_unseen_drug_hop2/seed_42/result_metrics.pt" ]; then
  echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') WARNING: unseen_drug result_metrics.pt not found after 8h wait"
fi

# 2) unseen_target 单种子（首跑自动重建 cache）
echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') launching unseen_target single seed..."
"$PY" main.py --data biosnap --split unseen_target --hop 2 --seeds 42 > coldstart_unseen_target.log 2>&1
echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') unseen_target exit=$?"

# 3) BindingDB SGBANDTI 单种子（cache 已建则直接训练；不一致则自动重建）
echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') launching BindingDB SGBANDTI single seed..."
"$PY" main.py --data bindingdb --split random --hop 2 --seeds 42 > bindingdb_sgbandti.log 2>&1
echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') bindingdb exit=$?"

echo "[chain] $(date '+%Y-%m-%d %H:%M:%S') OVERNIGHT QUEUE COMPLETE"
