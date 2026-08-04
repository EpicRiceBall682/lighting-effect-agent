# 模块二：数据集构建与文生图模型微调

## 职责

清洗现有图文训练对，程序化扩充光效数据，并在 Google Colab 上对预训练文生图模型进行 LoRA 微调。

## 输入

- `reference_data/训练集.xlsx`
- `reference_data/训练集图片/`
- 程序化生成的渐变、流体光晕和天空光效图片

## 输出

- 规范化训练集清单
- 训练集、验证集划分
- LoRA 权重文件，例如 `light_effect_lora.safetensors`
- 训练参数、日志和验证图片

## 现有参考

- 根目录 `simulate_graph_one.py`
- 根目录 `simulate_graph_more.py`
- 根目录 `generate_sky_light_texture.py`

## Colab 前的本地工具

- `src/audit_dataset.py`：检查 Excel、图片完整性、真实格式和重复内容
- `src/generate_synthetic.py`：统一调用三类参考方法，生成宽幅光效，并对最终整张图片做像素级色域与局部色斑检查
- `src/vision_caption.py`：生成或复用可续跑、可追溯的图片视觉描述
- `src/prepare_dataset.py`：去除完全重复图片，按来源和配方/场景组生成可复现的训练/验证划分，并打包为 Colab ZIP
- `tests/`：小尺寸离线测试

原始 `reference_data` 数据不会被修改。准备后的图片会统一转换成真正的 RGB PNG。

## 1. 审计原始数据

从项目根目录运行：

```bash
python3 -m modules.module_02_model_finetuning.src.audit_dataset \
  --output modules/module_02_model_finetuning/artifacts/source_audit.json
```

## 2. 生成宽幅合成数据

默认生成 180 张 `1056×320` 图片，对应约 3.3:1 的灯具比例：

```bash
python3 -m modules.module_02_model_finetuning.src.generate_synthetic \
  --output-dir modules/module_02_model_finetuning/artifacts/synthetic_v4_regenerated_uncaptioned
```

默认包含：

- 60 张双色线性渐变
- 60 张多色流体光晕
- 60 张天空云彩光效

合成数据现在允许绿色、青色及其他完整色相。审计仍会拒绝过暗图片和局部
高频色斑，但不会再把正常的绿色或蓝黄之间的绿色过渡判为不合格。

这项审计只影响以后重新生成、重新训练的数据；当前模块三已经加载的 LoRA
权重不会因为代码更新自动改变。

## 3. 校验视觉描述

本次 v4 数据包保留已逐图验证的 v3 图片与 180 条视觉描述，并为它们回填
确定性的配方分组。重新生成的候选图因为哈希发生变化，保存在
`synthetic_v4_regenerated_uncaptioned`，在重新完成视觉描述前不会进入训练包。
下面的命令只校验正式 v4 描述与图片哈希，不调用外部 API，也不会下载模型。

运行：

```bash
python3 -m modules.module_02_model_finetuning.src.vision_caption \
  --source-metadata modules/module_02_model_finetuning/artifacts/synthetic_v4/synthetic_metadata.jsonl \
  --output-metadata modules/module_02_model_finetuning/artifacts/synthetic_v4/vision_metadata.jsonl \
  --cache-only
```

命令成功时会显示 `reused: 180` 和 `remaining_uncaptioned: 0`。也可以直接双击 `run_vision_caption.command`。

## 4. 生成 Colab 数据包

```bash
python3 -m modules.module_02_model_finetuning.src.prepare_dataset \
  --synthetic-metadata modules/module_02_model_finetuning/artifacts/synthetic_v4/vision_metadata.jsonl \
  --output-dir modules/module_02_model_finetuning/artifacts/v4_release/colab_dataset \
  --zip-output modules/module_02_model_finetuning/artifacts/v4_release/module_02_colab_dataset_v4.zip
```

正式打包默认会拒绝模板描述；如果仍有合成图片没有视觉模型描述，命令会直接报出具体行号，避免错误数据被带入训练。

输出目录符合 Hugging Face ImageFolder 格式：

```text
colab_dataset/
├── train/
│   ├── metadata.jsonl
│   └── *.png
├── validation/
│   ├── metadata.jsonl
│   └── *.png
├── dataset_summary.json
└── source_audit.json
```

每张图片对应的训练提示词保存在相同分区的 `metadata.jsonl` 中：

```json
{"file_name": "synthetic_linear_0001.png", "text": "Wide panoramic lighting with ..."}
```

- `file_name`：图片文件名
- `text`：训练时送给文本编码器的提示词
- `scene_prompt`：原始场景描述，仅供追溯，不作为本次训练的 caption
- `original_name`：图片在整理前的名称，仅供追溯
- `caption_source`：`vision_model`、`organizer_original` 等描述来源
- `caption_model`：生成视觉描述时使用的模型，便于复现
- `recipe_id`：合成图的生成配方
- `split_group`：不能跨训练/验证集合的配方或原始场景组
- `palette_family`、`layout_id`：色板家族与布局审计信息

划分先按 `source` 分层，再把相同 `split_group` 的记录作为整体放入训练集
或验证集。这样既保持来源可控，又避免同一合成配方或同一原始场景的近邻样本
同时出现在两个集合中。

合成图片生成阶段的模板描述位于 `artifacts/synthetic_v4/synthetic_metadata.jsonl`，最终视觉描述位于 `artifacts/synthetic_v4/vision_metadata.jsonl`。最终是否进入训练集或验证集，以 `artifacts/v4_release/colab_dataset/train/metadata.jsonl` 和 `validation/metadata.jsonl` 为准。

当前 v4 发布包包含 254 张训练图和 42 张验证图，ZIP SHA256 为
`b8d6e3ed2ac2c46fa3923f255fb0577190b2e12e5f8eca944d3af0984b6c7840`。
训练/验证之间的 `split_group` 和图片内容哈希交叉均为 0。

进入 Colab 后可以直接使用：

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="/content/colab_dataset")
```

## 5. 在 Colab T4 上训练 LoRA

需要上传到 Colab 的两个本地文件：

- `artifacts/v4_release/module_02_colab_dataset_v4.zip`
- `notebooks/train_sd15_lora_t4_colab.ipynb`

打开 [Google Colab](https://colab.research.google.com/)，选择“文件 → 上传笔记本”，上传 Notebook。然后选择“代码执行程序 → 更改运行时类型 → T4 GPU”，从上到下运行每个单元格。

Notebook 默认执行 SD 1.5 LoRA 正式训练，并补丁官方 Diffusers
`train_text_to_image_lora.py` 的图像变换以保留宽幅构图：

- 训练画布：768×232（约 3.31:1，宽高均可被 8 整除）
- 步数：1000
- LoRA rank：8
- 混合精度：FP16
- PyTorch AdamW 与梯度检查点：开启
- 检查点间隔：250 步

输出保存在 Google Drive 的：

```text
MyDrive/lighting_lora/sd15_light_effect_lora_full/
```

最终供模块三加载的文件为：

```text
light_effect_lora.safetensors
```

## 离线测试

```bash
python3 -m unittest discover -s modules/module_02_model_finetuning/tests -v
```
