# 模块一：场景理解与 Prompt 转译

## 职责

接收用户的场景描述和灯具规格，调用大语言模型，将自然语言转成可校验的光效参数 JSON 和英文生图 Prompt。

## 输入

- 场景描述
- 灯具宽度、高度
- 可选的空间面积

## 输出

- 结构化光效参数 JSON
- JSON 中的 `effect` 字段即英文生图 Prompt

Negative Prompt、目标宽高比和建议生成尺寸由模块三负责，避免两个模块分别维护同一套推理参数。

## 现有参考

- 根目录 `demo.py`
- 本地 `reference_data/生图光效提示词.xlsx`

## 当前实现

- `src/prompt_builder.py`：基于根目录 `demo.py` 构建系统提示词和用户提示词
- `src/client.py`：调用 DeepSeek V4 Chat Completions API，并对空响应和临时错误自动重试
- `src/schemas.py`：校验 density、三类亮度和英文 effect
- `src/agent.py`：串联调用、校验和一次自动纠错
- `src/cli.py`：命令行入口
- `tests/`：不消耗 API 额度的离线测试

`effect` 不是场景照片描述，而是参考训练集风格的横向配色 caption。
当前要求它用 30～50 个英文单词明确表达左侧辅助色、中心至右侧主色、
水平渐变和统一的纵向颜色。默认只使用 2～3 种颜色，活力场景优先使用
清晰高饱和配色，不再使用 `misty glow`、`diffused bloom`、
`luminous center` 等会诱发随机光斑和雾团的合成数据词汇。

为减少扩散模型产生局部彩色斑点，`effect` 还会拒绝 `spots`、`dots`、
`clusters`、`accents`、`streaks`、`beams` 等容易诱发小面积硬结构的词。
所有颜色区域应描述为宽阔、连续且无局部纹理的低频横向渐变。

当输入包含草原、森林等绿色意象时，模块仍遵守赛题禁用绿色的要求，
但必须保留上下层次、开阔感和自然氛围，以浅蓝、象牙白、淡黄或暖粉转换，
不能直接删除场景结构。前端会向用户显示这项颜色转换说明。

## DeepSeek API Key

代码默认调用：

```text
Endpoint: https://api.deepseek.com/chat/completions
Model:    deepseek-v4-flash
```

请先在 DeepSeek Platform 创建 API Key，并在运行前设置环境变量。不要把真实 Key 写进代码或提交到仓库。

macOS / Linux：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

也可以通过以下环境变量覆盖默认配置：

```bash
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_ENDPOINT="https://api.deepseek.com/chat/completions"
```

模块一默认使用速度更快、成本更低的 `deepseek-v4-flash`。只有在复杂场景下需要更强推理时，才建议把 `DEEPSEEK_MODEL` 改为 `deepseek-v4-pro`。代码会显式关闭思考模式，并使用 JSON Output；模型输出仍需通过本地字段、颜色、亮度和英文校验。

## 运行

从项目根目录执行：

```bash
python3 -m modules.module_01_prompt_agent.src.cli \
  --scene "酒店套房客厅，浪漫氛围" \
  --width-mm 1220 \
  --height-mm 370 \
  --space-size-m2 31.8
```

可使用 `--output outputs/result.json` 保存结果。

预期输出格式：

```json
{
  "density": "middle",
  "m_intensity": 70,
  "k_intensity": 90,
  "a_intensity": 60,
  "effect": "Wide panoramic organizer-style color field with light peach on the left and dominant warm amber across the center and right, forming a clean smooth horizontal gradient with uniform vertical color, balanced illumination, welcoming warmth, and an uninterrupted surface throughout."
}
```

## 离线测试

```bash
python3 -m unittest discover -s modules/module_01_prompt_agent/tests -v
```

离线测试不会调用 DeepSeek API，也不会消耗额度。
