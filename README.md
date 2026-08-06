# 宽幅光效智能体

本项目把同一中文场景分别转换成自然概念图提示词和独立光效提示词，两条链路
分别生成带实体场景的概念图与宽幅光色图。系统只在双方主色严重冲突时，才用
概念图色板纠正光效图，最后映射到目标 SDL 硬件色域。核心链路为：

```text
中文场景 ┬→ 自然概念提示词 → 实体概念图 ─┐
        └→ 独立光效提示词 → 宽幅渐变 ───┴→ 严重偏色检测 → 模块五增强 → SDL 输出
```

模块五以低频连续主题光场增强接入模块三与模块四之间；质量不通过时会
自动降低强度或安全绕过，不会阻断 SDL 输出。

默认网页采用四步 LCM 概念图推理，只运行一次扩散模型；概念图保持生成模型的
自然材质与场景颜色，光色图默认保留模块一独立判断的配色。只有概念图主色具有
足够色度且双方主色达到严重偏色阈值时，才启用概念图色板兜底。模型加载在网页开放前
完成，目标是在支持 CUDA 的 Windows 设备上于
6 秒内同时显示概念图和光色图。绿色、青色及其他色相均允许，硬件边界仍由
SDL 映射处理。宽幅渐变还会按场景语义执行动态亮度下限：活力、霓虹和运动
场景避免出现大片暗区，明确暗调、微光和助眠场景则保留较低亮度。该步骤为
确定性像素处理，不增加扩散模型调用；它同时检查 RGB 通道能量，不会仅因红色
或蓝色的相对亮度权重较低就自动混白。CPU 可以运行，但不承诺 6 秒目标。

## Windows 快速开始

支持 Windows 10/11 与 Python 3.11～3.13。下载并解压仓库后，双击：

```text
start_demo_windows.bat
```

脚本会自动创建 `.venv`、安装依赖、检测并优先使用 NVIDIA CUDA、询问当前进程
使用的 DeepSeek API Key，并打开 `http://127.0.0.1:7860/`。默认快速模式在用户
无论用户是否写明颜色，模块一大模型都会在一次调用中分别生成自然场景概念提示词
和独立光效提示词，等待上限为 3.0 秒；概念图色板只参与严重偏色检测，不再默认
覆盖光效配色。大模型不可用时，本地双提示词会按时间、自然现象、空间和风格语义
选择多样化应急配色。API Key 不会写入项目文件。
检测到 CUDA 时会显示显卡名称并明确使用 `--device cuda`；否则会提示
CPU 回退。如果电脑有 NVIDIA 驱动但当前 PyTorch 不支持 CUDA，脚本会给出官方
CUDA 安装入口。

如需启用模块四，把已获授权的 `SDL2_0.txt` 放到：

```text
reference_data\颜色信息\SDL2_0.txt
```

也可以把该文件直接拖到 `start_demo_windows.bat` 上。公开仓库不包含私有色表；
缺少色表时 Demo 仍会运行模块一、三、五，并明确标记为“预览模式”，不会把结果
误报为满足 SDL 硬件色域。

PowerShell 手动启动方式：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
.\.venv\Scripts\python.exe -m modules.module_06_demo_evaluation.src.app --inbrowser
```

首次启动会从 Hugging Face 下载 Stable Diffusion 1.5 基础模型和概念图 LCM
加速适配器。没有可用的
NVIDIA CUDA 环境时会自动使用 CPU，功能可用但生成速度会明显变慢。Windows
CUDA 版 PyTorch 应根据显卡驱动从 `https://pytorch.org/get-started/locally/`
选择合适版本，不建议盲目固定 CUDA wheel。

## macOS / Linux 快速开始

支持 Python 3.11～3.13。在项目根目录创建环境并安装完整运行依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

需要严格复现当前本地验证环境时，改用：

```bash
python -m pip install -r requirements-lock.txt
```

设置 DeepSeek API Key：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

启动本地界面：

```bash
python -m modules.module_06_demo_evaluation.src.app --inbrowser
```

macOS 也可以双击 `start_demo.command`；该脚本优先使用现有的
`.venv-module3` 环境，否则使用按上文创建的 `.venv` 环境。

可通过任意平台的 `--sdl-path` 指定私有色表；需要强制完整模块四输出时增加
`--require-sdl`，缺少文件会立即退出：

```bash
python -m modules.module_06_demo_evaluation.src.app \
  --sdl-path "reference_data/颜色信息/SDL2_0.txt" \
  --require-sdl \
  --inbrowser
```

也可以使用根目录兼容入口：

```bash
python demo.py
```

## 批量评测与提交表

对本地 `reference_data/测试集.txt` 的场景运行可续跑评测：

```bash
python -m modules.module_06_demo_evaluation.src.evaluator \
  --output-dir outputs/evaluation
```

每条结果会立即追加到 `batch_results.jsonl`。再次执行会跳过已成功的场景，
失败场景会被重新尝试。全部成功后生成比赛提交表：

```bash
python -m modules.module_06_demo_evaluation.src.exporter \
  --results outputs/evaluation/batch_results.jsonl \
  --output outputs/evaluation/测试集提交结果.xlsx
```

导出器默认拒绝包含失败项的不完整批次，并按官方模板写入英文 Prompt、
中文 Prompt、嵌入图片和原始 Scene。

## 训练与推理

- 数据审计、合成、分组划分和 Colab LoRA 训练：
  `modules/module_02_model_finetuning/README.md`
- LoRA 推理、三阶段图像与质量门禁：
  `modules/module_03_image_generation/README.md`
- SDL 色域映射：`modules/module_04_gamut_mapping/README.md`
- Demo、批量评测和 Excel 导出：
  `modules/module_06_demo_evaluation/README.md`

正式权重位于
`modules/module_03_image_generation/weights/light_effect_lora.safetensors`。
首次推理仍需从 Hugging Face 下载 SD 1.5 基础模型。

## 测试

```bash
python -m unittest discover -s modules -p "test_*.py" -v
```

离线测试不会下载基础模型，也不会调用 DeepSeek API。

GitHub Actions 会在 `main` 分支推送和 Pull Request 时自动执行 Linux 完整测试，
并在 Windows 上验证安装、源码编译、Demo 流水线和 SDL 缺失降级逻辑。

## 可复现性

正式 LoRA 权重、基础模型 revision、训练数据 ZIP 哈希、训练尺寸、步数和数据
划分信息记录在：

```text
modules/module_03_image_generation/weights/weight_provenance.json
```

批量评测的恢复缓存会同时校验推理配置、权重、SDL 表、核心源码哈希和结果文件
是否存在。发布代码变化后不会错误复用旧批次结果。

## GitHub 提交

- [SUBMISSION.md](SUBMISSION.md)：最终 45 条评测、Excel 导出和 Release 清单
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：基础模型及主办方资料说明
- [LICENSE](LICENSE)：比赛评审用途许可

Git 仓库不包含主办方资料。原始训练图片、训练工作簿、评测数据、提交模板、
历史输出、准备后的数据集 ZIP、虚拟环境和旧模型权重均由 `.gitignore` 排除。
正式提交 Excel、批量 JSONL 和可选训练包应作为 GitHub Release 附件提供，
避免进入 Git 历史。

## 仓库说明

- `reference_data/` 与根目录赛题文档属于本地资料，不进入 Git 历史。
- 批量评测、提交表导出和 SDL 映射需要用户在本地提供相应资料。
- `outputs/`、虚拟环境和模块生成的 `artifacts/` 已在 `.gitignore` 中排除。
- 仓库采用比赛评审用途许可，不授予通用开源或商业使用权。
