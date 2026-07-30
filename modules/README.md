# 项目模块目录

本目录按照六个功能模块组织代码。本地 `reference_data/` 资料保持只读，并由
`.gitignore` 排除。

| 目录 | 功能 |
| --- | --- |
| `module_01_prompt_agent` | 场景理解与光效 Prompt 转译 |
| `module_02_model_finetuning` | 数据集构建与文生图 LoRA 微调 |
| `module_03_image_generation` | 加载基础模型和 LoRA，自动生成 Raw 光效图 |
| `module_04_gamut_mapping` | SDL 色域检测、映射与平滑处理 |
| `module_05_pattern_generation` | 低频主题光场增强、质量降级与安全绕过 |
| `module_06_demo_evaluation` | Demo、批量测试、指标评价与结果导出 |

建议各模块逐步采用统一结构：

```text
module_xx_name/
├── README.md       # 模块说明
├── src/            # 正式实现
├── tests/          # 单元测试
└── configs/        # 可调整配置（需要时创建）
```
