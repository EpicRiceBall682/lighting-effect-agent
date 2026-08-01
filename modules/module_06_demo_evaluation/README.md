# 模块六：本地 Demo 与评价

## 职责

当前使用本地 Gradio 界面串联模块一、三、四，展示未经物理约束的 Raw
光效、模块五低频主题增强与 SDL 高保真结果。

## 输入

- 用户场景和灯具信息
- 模块一、三、四的接口
- 本地 `reference_data/测试集.txt`
- 本地 `reference_data/测试集提交格式.xlsx`

## 输出

- Gradio Demo
- Raw 图与 SDL 图 A/B 对比
- 单次可复现报告和完整结果 ZIP
- 可续跑批量评价报告、测试集提交表和结果图片

## 已实现

- 中文场景、灯具尺寸和可选空间面积的一次输入
- DeepSeek V4 Flash 英文光效提示词
- 对绿色/植被等受限场景显示明确的色彩转换说明
- 英文提示词可编辑，并可跳过 DeepSeek 直接重新生成
- SD 1.5 + LoRA Raw 光效图
- 模块三局部色斑检测、轻度平滑和局部羽化修复
- 模块三 Raw 与模块四 SDL 的赛题要求 A/B 对比
- SDL 连续色域视觉预览、严格控制图和超色域掩膜
- 关键质量指标、JSON 报告和 ZIP 下载
- 模型惰性加载与复用，生成任务串行执行
- 45 条官方测试场景的逐条 JSONL 批量运行与断点续跑
- 保持官方四列表头并嵌入结果图片的 Excel 提交表导出

## 启动

Windows 10/11 可在项目根目录双击：

```text
start_demo_windows.bat
```

脚本自动创建 `.venv` 并安装运行依赖。启动前会检测 PyTorch CUDA：可用时明确
选择 `--device cuda` 并显示 NVIDIA 显卡名称，不可用时提示并回退 CPU；检测到
NVIDIA 驱动但 PyTorch 不支持 CUDA 时，会显示官方安装入口。把授权的 SDL 色表
拖到 BAT 文件上，可以从任意本地路径启动完整模块四；不提供色表时会进入
模块三/五预览模式。

在 Finder 中双击项目根目录的 `start_demo.command`，浏览器会自动打开：

```text
http://127.0.0.1:7860
```

也可在项目根目录运行：

```bash
.venv-module3/bin/python -m modules.module_06_demo_evaluation.src.app --inbrowser
```

DeepSeek 密钥优先读取 `DEEPSEEK_API_KEY`；在 macOS 上，如果它已经通过
`launchctl` 配置，界面会自动读取，不会在页面或日志中显示密钥。

第一次点击生成时需要加载 Stable Diffusion 模型；后续生成会复用同一个模型。
每次运行的全部文件保存在 `outputs/demo/<时间戳_随机编号>/`。

跨平台命令行参数：

```text
--sdl-path <文件>   指定本地 SDL 色表
--require-sdl       缺少 SDL 色表时拒绝启动
--device auto       自动选择 CUDA、MPS 或 CPU
```

公开仓库不包含授权资料。未配置 SDL 色表时，页面会显示模块三 Raw、模块五主题
结果和明确的不可用控制图占位符；报告中的模块四状态为
`skipped_missing_sdl_table`，不会产生虚假的硬件合规结论。

界面生成的是受赛题配色和硬件色域约束的“抽象灯具光效”，不是场景照片。
完成一次自动生成后，可以直接修改页面中的英文提示词，再点击
“使用修改后的提示词重新生成”；这次不会再次调用 DeepSeek，但编辑后的
提示词仍必须通过颜色、英文长度和空间结构校验。

模块四接收模块五增强图；模块五未通过质量门时会自动绕过，因此其输出与
模块三 Raw 图逐像素相同。若 Raw 图全部
位于 SDL 色域内，页面会明确提示“无需改变视觉预览”，而不是把相同结果
误解为处理失败。

默认种子模式为“场景派生”：界面中的基础种子会与中文场景共同计算有效种子，
因此不同场景不会再从完全相同的扩散噪声开始，同一场景仍能复现。高级参数中
勾选“固定随机种子”后，才会严格使用输入数值。流水线还会把新图与上一张做
低分辨率感知差异比较；若不同场景的差异低于阈值，会自动更换种子重试一次，
并把比较结果和重试记录写入 `pipeline_report.json`。固定种子模式不会执行
任何换种子重试；若质量未通过，会保留可复现语义并明确报错。

## 批量测试

```bash
.venv-module3/bin/python \
  -m modules.module_06_demo_evaluation.src.evaluator \
  --output-dir outputs/evaluation
```

运行器按 `测试集.txt` 原始顺序逐条处理，每条完成后立即追加到
`batch_results.jsonl`，并更新 `batch_summary.json`。再次执行时会跳过成功
记录并重试失败记录；单条异常不会丢失整批进度。恢复缓存同时校验推理配置、
模型和权重哈希、核心源码哈希以及产物文件是否仍然存在；任一变化都会重新生成，
不会复用旧配置的结果。

当前模块一只生成英文 Prompt，因此批量记录和提交表的 `中文 Prompt` 保守地
保存原始中文 Scene，不伪造未经过模型校验的翻译。

全部成功后导出官方模板：

```bash
.venv-module3/bin/python \
  -m modules.module_06_demo_evaluation.src.exporter \
  --results outputs/evaluation/batch_results.jsonl \
  --output outputs/evaluation/测试集提交结果.xlsx
```

默认嵌入模块三最终 Raw 图。也可通过 `--image-field sdl_preview_path` 导出
SDL 视觉预览。若批次仍有失败项，导出器默认拒绝生成不完整提交表。

## 代码

- `src/app.py`：Gradio 界面
- `src/pipeline.py`：串联各模块
- `src/evaluator.py`：可续跑批量测试与 JSONL/摘要输出
- `src/exporter.py`：官方模板 Excel 填充与图片嵌入
- `tests/test_module_06.py`：流水线和界面冒烟测试
