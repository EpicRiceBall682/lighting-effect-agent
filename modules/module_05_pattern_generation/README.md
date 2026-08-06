# 模块五：低频主题光场增强

## 职责

模块五位于模块三 Raw 光效与模块四 SDL 映射之间，用确定性的连续低频光场
加强场景节奏，但不绘制花瓣、圆点、线条或照片式物体。

```text
模块一场景参数 → 模块三基础 Raw → 模块五主题增强 → 模块四 SDL 输出
```

## 三类增强

- `breathing`：平缓呼吸光场，适合酒店、休闲和睡眠场景。
- `flowing`：横向或斜向流动光场，适合水波、品牌、运动和动态场景。
- `radiant`：宽范围中心扩散，适合入口、橱窗、日出和唤醒场景。

默认强度为 `0.07～0.11`，用户可在 `0～0.18` 范围内调整。模块五会检查：

增强会先对已有横向色板做轻微的低频非对称位移，再在线性 RGB 中加入平滑亮度
起伏；它只移动模块一已经选择的颜色，不引入新的限制色。增益会在触及显示色域
边界前受限，因此不会因为单通道裁剪而改变主渐变色相。

- 平均像素变化；
- 平均亮度变化；
- 局部色斑增量；
- 中尺度宽区域色斑增量。

如果质量不通过，强度会自动减半重试，最多三次；仍不通过时输出模块三原图，
并在 `module_05_pattern.json` 中记录 `quality_status=bypassed` 和原因。

## 独立运行

```bash
.venv-module3/bin/python \
  -m modules.module_05_pattern_generation.src.cli \
  --input outputs/module_03/example.png \
  --scene "酒店入口的日出唤醒光" \
  --strength 0.10 \
  --seed 42 \
  --output-dir outputs/module_05/radiant
```

## 测试

```bash
.venv-module3/bin/python -m unittest \
  modules.module_05_pattern_generation.tests.test_module_05
```
