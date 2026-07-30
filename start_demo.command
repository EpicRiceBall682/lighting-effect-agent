#!/bin/zsh

set -eu
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ -x ".venv-module3/bin/python" ]]; then
  PYTHON_BIN=".venv-module3/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  echo "未找到 .venv-module3 或 .venv。请先按 README 创建运行环境。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if curl --silent --fail --max-time 2 "http://127.0.0.1:7860/" >/dev/null; then
  open "http://127.0.0.1:7860/"
  exit 0
fi

exec "$PYTHON_BIN" \
  -m modules.module_06_demo_evaluation.src.app \
  --inbrowser
