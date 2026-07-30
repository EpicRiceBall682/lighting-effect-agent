#!/bin/zsh

set -u

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h:h}
cd "$PROJECT_ROOT" || exit 1

echo "模块二：校验 synthetic_v3 的 180 条视觉描述"
echo "当前发布数据已完成视觉描述，本步骤只核对描述与图片哈希，不调用外部 API。"
echo

python3 -m modules.module_02_model_finetuning.src.vision_caption \
  --source-metadata modules/module_02_model_finetuning/artifacts/synthetic_v3/synthetic_metadata.jsonl \
  --output-metadata modules/module_02_model_finetuning/artifacts/synthetic_v3/vision_metadata.jsonl \
  --cache-only
STATUS=$?

echo
if [[ $STATUS -eq 0 ]]; then
  echo "视觉描述校验通过：180 张图片都已有对应描述。"
else
  echo "视觉描述不完整或文件校验失败，请把上方错误信息发给 Codex。"
fi
read "REPLY?按回车关闭窗口。"
exit $STATUS
