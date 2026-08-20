#!/usr/bin/env bash
# 创建/更新 metrics/venv 并安装依赖（幂等，可重复执行）
# 用法： bash setup_venv.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"

# Windows Git Bash 下 venv 的 python 位于 Scripts/，Linux/macOS 位于 bin/
if [ -f "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"
elif [ -f "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
else
  PY=""
fi

if [ -z "$PY" ]; then
  echo "==> 创建虚拟环境: $VENV"
  python -m venv "$VENV"
  if [ -f "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"
  else
    PY="$VENV/bin/python"
  fi
fi

echo "==> 升级 pip"
"$PY" -m pip install --upgrade pip

echo "==> 安装 torch / torchvision（体积较大，请耐心等待，pip 会显示进度）"
"$PY" -m pip install torch torchvision

echo "==> 安装其余依赖（numpy/scikit-image/pandas/Pillow/lpips/flip_evaluator）"
"$PY" -m pip install -r "$ROOT/requirements.txt"

echo "==> 完成。使用方式： $PY 脚本.py ..."
