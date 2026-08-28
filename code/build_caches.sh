#!/bin/bash
# 重建全部子图缓存(一次性, 约 1-3 小时)。数据见 ../data/。
# 用法: 先确认环境(check_env.py), 再 bash build_caches.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python"
export MPLBACKEND=Agg

# biosnap/random: nested(子图) + flat(整分子图, 消融 no_subgraph/no_both 用)
"$PY" -u build_subgraph_cache.py --data biosnap --split random      --hop 2 --use-nested nested --num-workers 4
"$PY" -u build_subgraph_cache.py --data biosnap --split random      --hop 2 --use-nested flat   --num-workers 4
# 冷启动(nested)
"$PY" -u build_subgraph_cache.py --data biosnap --split unseen_drug   --hop 2 --use-nested nested --num-workers 4
"$PY" -u build_subgraph_cache.py --data biosnap --split unseen_target --hop 2 --use-nested nested --num-workers 4
# BindingDB(nested)
"$PY" -u build_subgraph_cache.py --data bindingdb --split random    --hop 2 --use-nested nested --num-workers 4

echo "[cache] 全部缓存完成"
