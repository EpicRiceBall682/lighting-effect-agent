# 模块四：SDL 物理色域约束

## 职责

接收模块三生成的 Raw 光效图，使用本地提供的 SDL CIE xy/RGB 表进行色域检测，
输出连续色域视觉预览、严格硬件控制图、超色域遮罩和质量报告。

## 实现

- `src/color_spaces.py`：D65 sRGB、XYZ、xyY 和 Lab 双向转换
- `src/sdl_palette.py`：严格解析 SDL 表、构建 xy 凸包并校验 RGB 表成员
- `src/mapper.py`：仅压缩超色域像素的连续视觉预览，以及官方最近邻/平滑有序抖动严格控制图
- `src/metrics.py`：超域比例、严格合规、色差、亮度、突变和色带平台指标
- `src/cli.py`：单张图片命令行入口
- `tests/`：颜色转换、SDL 解析、严格合规、确定性和透明通道测试

当前 `SDL2_0.txt` 包含 1024 条 xy/RGB 记录、990 个唯一 RGB，xy 外边界为三角形。

## 两条独立输出路径

### 连续视觉预览

供人眼比较的预览图不会再把所有像素量化到 SDL 离散 RGB 表：

- 已经位于 SDL xy 色域内的像素与 Raw 图逐字节一致。
- 只有超出色域的 xy 色度会连续投影到最近边界，并尽量保持原始亮度。
- 预览图用于观察最终色彩趋势，不保证每个 RGB 像素都是硬件控制码。

这样可以避免“图像原本完全在色域内，却仍被量化出明显硬边界”的问题。
因此，如果报告中的 `before_xy_out_of_gamut_fraction` 为 `0`，连续视觉
预览与 Raw 图完全相同是正确结果，不代表模块四没有运行。

### 严格硬件控制图

带 `_control` 的文件仍执行离散 SDL 映射，每个像素必须精确属于 SDL RGB 表。
控制图可能存在离散纹理，它是硬件数据，不作为人眼最终成图。

严格控制图提供两种方法：

#### `nearest`

复现赛题方参考方法：在 CIE Lab 中把每种输入颜色替换成距离最近的 SDL RGB。优点是简单且严格合规，缺点是渐变中可能产生色带。

#### `smooth`（默认）

先对输入做轻度高斯预平滑，再在 Lab 中找到两个最接近的 SDL 颜色，通过确定性的 8×8 有序抖动进行空间混合。每个最终像素仍是 SDL 表中的精确 RGB，但在正常观看距离下能形成更连续的视觉渐变。

SDL 表中的 RGB 是接近满功率的颜色控制码。严格控制图保留完整 SDL
原始控制码；连续视觉预览与其分开计算，避免控制码的离散量化污染人眼预览。

## 运行

从项目根目录执行：

```bash
python3 -m modules.module_04_gamut_mapping.src.cli \
  --input outputs/scene_to_image/deepseek_coffee_test/raw_light_effect_seed_20260724.png \
  --output-dir outputs/module_04/deepseek_coffee_test \
  --method smooth \
  --save-baseline
```

输出：

```text
outputs/module_04/deepseek_coffee_test/
├── raw_light_effect_seed_20260724_sdl_smooth.png
├── raw_light_effect_seed_20260724_sdl_smooth_control.png
├── raw_light_effect_seed_20260724_sdl_nearest_baseline.png
├── raw_light_effect_seed_20260724_sdl_nearest_baseline_control.png
├── raw_light_effect_seed_20260724_out_of_gamut_mask.png
└── raw_light_effect_seed_20260724_sdl_report.json
```

其中不带 `_control` 的图片是连续色域展示预览，带 `_control` 的图片是
发送给后续硬件层的严格 SDL 控制图。

报告中的 `strict_invalid_pixel_count` 必须为 `0`，表示控制图的所有输出像素都能在 SDL RGB 表中找到。

除了严格表成员校验，默认质量策略还限制超色域比例、预览 P95 色差、亮度变化、
控制图突变、梯度不连续和 smooth 控制图平台比例。报告会写入
`quality_policy`、`quality_status` 和 `quality_failures`；被拒绝的结果仍保留
预览、控制图、掩膜和报告供诊断，但不会进入 Demo 下载 ZIP。

`control_flat_neighbor_fraction` 用于观察离散控制图造成的大面积相同颜色平台。对于原本连续的渐变图，该数值越高，出现可见色带的风险越大。平滑方法以细粒度空间混色换取更低的平台比例，因此报告同时保留预览图和控制图的突变、二阶梯度指标，方便观察这种取舍。

## 常用参数

- `--method smooth`：默认平滑映射
- `--method nearest`：官方参考基线
- `--dither-strength 1.0`：相邻 SDL 颜色混合强度
- `--smooth-radius 0.6`：映射前的轻度平滑半径
- `--save-baseline`：同时输出最近邻结果，便于 A/B 对比

## 自动测试

```bash
python3 -m unittest discover -s modules/module_04_gamut_mapping/tests -v
```
