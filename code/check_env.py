"""环境自检: 验证依赖 + CUDA + SGBANDTI 可实例化。跑任何训练前先执行。"""
import sys
import torch

ok = True
print(f"Python: {sys.version.split()[0]}")
for mod in ["torch", "dgl", "dgllife", "torch_scatter", "rdkit", "sklearn", "yacs", "prettytable", "tqdm", "pandas", "numpy"]:
    try:
        m = __import__(mod)
        print(f"  [OK] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {mod}: {e}")

print(f"CUDA available: {torch.cuda.is_available()}"
      + (f", device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

try:
    from configs import get_cfg_defaults
    from models import SGBANDTI
    m = SGBANDTI(**get_cfg_defaults())
    n = sum(p.numel() for p in m.parameters())
    print(f"  [OK] SGBANDTI 实例化, params={n} (应为 1,070,342)")
    if n != 1070342:
        ok = False
except Exception as e:
    ok = False
    print(f"  [FAIL] SGBANDTI 实例化: {e}")

print("环境自检:", "通过 ✅" if ok else "有缺失 ❌ 按 requirements.txt / environment.yml 补装")
