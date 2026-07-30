# 模块三：自动化推理生图

## 职责

加载 Stable Diffusion 1.5 和模块二训练的 LoRA，接收模块一 JSON 中的 `effect` 提示词，输出未经色域映射的 Raw 光效图与可复现参数 JSON。

## 当前文件

- `weights/light_effect_lora.safetensors`：模块二训练完成的正式 LoRA 权重
- `src/model_loader.py`：自动选择 CUDA、Apple MPS 或 CPU，加载基础模型与 LoRA
- `src/image_geometry.py`：把灯具毫米尺寸换算成保持比例且能被扩散模型接受的像素尺寸
- `src/generator.py`：运行 LoRA、调用结构化渲染、执行质量门禁并保存参数清单
- `src/structured_gradient.py`：从提示词构建 2～3 节点横向主渐变，并仅保留受限的 LoRA 低频亮度残差
- `src/quality.py`：记录平均亮度、近黑比例、禁用色、突变像素和局部色斑比例
- `src/cli.py`：命令行入口，可直接读取模块一 JSON
- `tests/`：不下载基础模型、不占用 GPU 的离线测试

当前权重信息：

```text
文件大小：约 6.1 MB
LoRA rank：8
张量数量：256
SHA256：32140f4d8750e8b6b43f6440e7e28fa7ab5bb7840de68f36fb4733c81ba2ddd0
基础模型：stable-diffusion-v1-5/stable-diffusion-v1-5
基础模型 revision：451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

当前正式权重来自 v4 宽幅重训：使用 SHA256 已固定的分组数据集、768×232
宽幅 transform、1000 个训练步和 rank 8。训练配置、数据集哈希与验证状态记录在
`weights/weight_provenance.json`；替换前的 legacy 权重保存在 `weights/archive/`。

## 安装

建议创建单独的 Python 环境，然后从项目根目录运行：

```bash
python3 -m pip install -r modules/module_03_image_generation/requirements.txt
```

MacBook Air M4 会自动使用 MPS；Colab T4 会自动使用 CUDA。首次运行需要从 Hugging Face 下载 SD 1.5 基础模型。

## 一条命令完成中文场景到图片

这是最适合最终用户的入口，只输入一次中文：

```bash
python3 -m modules.module_03_image_generation.src.scene_to_image_cli \
  --scene "傍晚的海边餐厅，顾客正在安静地用餐，希望灯光呈现温暖、浪漫、放松的日落氛围。" \
  --width-mm 1220 \
  --height-mm 370 \
  --space-size-m2 35 \
  --device mps \
  --seed 20260722 \
  --output-dir outputs/scene_to_image/seaside
```

程序内部会自动完成：

```text
中文场景 → DeepSeek V4 Flash 英文提示词 → SD 1.5 + LoRA → Raw 光效图
```

输出目录同时保留英文提示词、图片和推理参数，便于复现。

## 分两步运行（调试用）

先让模块一保存结果：

```bash
python3 -m modules.module_01_prompt_agent.src.cli \
  --scene "当前场景是一个飘香的咖啡厅，人们在这里能够惬意地品尝咖啡。" \
  --width-mm 1220 \
  --height-mm 370 \
  --output outputs/module_01/coffee.json
```

再交给模块三：

```bash
python3 -m modules.module_03_image_generation.src.cli \
  --prompt-json outputs/module_01/coffee.json \
  --fixture-width-mm 1220 \
  --fixture-height-mm 370 \
  --seed 20260719 \
  --output-dir outputs/module_03/coffee
```

1220×370 mm 会换算为接近相同比例的 1024×312 像素。输出包括：

```text
outputs/module_03/coffee/
├── raw_light_effect_seed_20260719_diffusion_raw.png
├── raw_light_effect_seed_20260719_guided.png
├── raw_light_effect_seed_20260719.png
└── raw_light_effect_seed_20260719.json
```

三个 PNG 分别是扩散模型原图、结构化主渐变与弱 LoRA 亮度合成图，以及通过
质量门禁的最终 Raw 图。JSON 会记录模块一的完整五项属性、原始提示词、实际增强后的提示词、
由属性转换得到的 `module_01_prompt_controls`、尺寸、seed、推理步数、
LoRA 强度、基础模型、权重哈希和基础图像质量报告。

`density` 当前控制光区数量和画面层次；主光、重点光和环境光强度会共同
决定整体明亮程度、重点光晕与环境填充方式。它们不再只是保存到 JSON 的
元数据，而会真正进入扩散模型的有效提示词。

默认生成模式为 `structured_gradient`。颜色和空间排布由提示词中的
左、中心、右颜色构建为确定性的水平渐变；LoRA 不再贡献色相，只提供默认
`0.10` 强度的低频亮度残差，对应最大约 `±4.5%` 的线性光增益。增益会在
不裁剪单个 RGB 通道的范围内生效，避免亮部产生额外偏色。若纵向颜色波动
超过阈值，系统会按 `0.10 → 0.08 → 0.04 → 0` 自动降低纹理强度。
渐变节点、主色、请求/实际纹理强度、横向结构解释率与纵向颜色波动均写入
`prompt_color_guidance`。旧颜色混合逻辑仍作为 `legacy_diffusion` 兼容路径保留。

无论颜色锚定是否启用，模块三都会检测相对于周围低频色场异常突出的局部
色斑。没有检测到色斑时逐字节保留图像；检测到异常区域时只在羽化掩膜内
混合低通结果，不再对整张图片做基础模糊。检测比例、修复范围和平均变化都会
写入 `artifact_cleanup`，最终剩余色斑比例写入 `quality`。

最终 Raw 图必须通过平均亮度、近黑像素、禁用色、突变像素和局部色斑五项
质量门禁，其中带有效空间颜色锚点的 Prompt 还必须通过布局误差门禁。未通过
时模块三自动更换确定性种子重试，默认最多两次；所有尝试、
失败原因和最终采用的种子都会记录在 manifest 中，全部失败则明确报错。

## 直接输入英文提示词

```bash
python3 -m modules.module_03_image_generation.src.cli \
  --prompt "Warm pale yellow to soft orange gradient with a gentle glow." \
  --width 1024 \
  --height 320 \
  --device mps
```

常用参数：

- `--device auto`：默认，优先 CUDA，其次 MPS，最后 CPU
- `--steps 30`：推理步数
- `--guidance-scale 7.0`：提示词引导强度
- `--lora-scale 1.0`：LoRA 强度，可尝试 0.7～1.0
- `--lora-texture-strength 0.10`：结构化模式中保留的 LoRA 低频亮度强度
- `--legacy-diffusion`：临时切回旧版扩散颜色输出，便于 A/B 对比
- `--seed`：固定后可以复现相同结果
- `--no-enrich`：不追加宽幅光效纹理与模块一属性控制描述

## 离线测试

```bash
python3 -m unittest discover -s modules/module_03_image_generation/tests -v
```

离线测试不会下载 SD 1.5。真正第一次生图时才会下载基础模型；Mac 上建议关闭其他占用大量内存的软件。
