"""Local Gradio interface for the lighting-effect pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from modules.module_04_gamut_mapping.src.sdl_palette import DEFAULT_SDL_PATH

from .pipeline import DEFAULT_OUTPUT_ROOT, LightingDemoPipeline


APP_TITLE = "AI 光效生成实验台"

CSS = """
:root {
  --app-ink: #251b18;
  --app-muted: #746965;
  --app-coral: #df5b45;
  --app-amber: #f3ad57;
  --app-cream: #fffaf2;
}
.gradio-container {
  color-scheme: light;
  --body-background-fill: #f7f3ed;
  --background-fill-primary: #f7f3ed;
  --background-fill-secondary: #fffaf2;
  --block-background-fill: rgba(255,255,255,.86);
  --block-border-color: rgba(72, 49, 42, .12);
  --block-label-background-fill: transparent;
  --block-label-border-color: transparent;
  --block-label-text-color: #514641;
  --block-title-background-fill: transparent;
  --block-title-border-color: transparent;
  --block-title-text-color: #514641;
  --panel-background-fill: #fffaf2;
  --panel-border-color: rgba(72, 49, 42, .12);
  --input-background-fill: #fffdf9;
  --input-background-fill-focus: #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-border-color: rgba(72, 49, 42, .16);
  --input-border-color-focus: #df5b45;
  --input-border-color-hover: rgba(223, 91, 69, .55);
  --body-text-color: #251b18;
  --body-text-color-subdued: #746965;
  --accordion-text-color: #251b18;
  --border-color-primary: rgba(72, 49, 42, .13);
  --button-secondary-background-fill: #fffaf2;
  --button-secondary-background-fill-hover: #fff1e5;
  --button-secondary-border-color: rgba(72, 49, 42, .14);
  --button-secondary-text-color: #3e322e;
  --button-secondary-text-color-hover: #251b18;
  --code-background-fill: #fffaf2;
  --table-even-background-fill: #fffaf2;
  --table-odd-background-fill: #ffffff;
  --table-text-color: #251b18;
  background:
    radial-gradient(circle at 12% 0%, rgba(255, 213, 154, .32), transparent 34rem),
    radial-gradient(circle at 92% 4%, rgba(232, 115, 91, .16), transparent 28rem),
    #f7f3ed;
  color: var(--app-ink);
}
.app-shell { max-width: 1240px; margin: 0 auto; }
.hero {
  padding: 28px 30px;
  border: 1px solid rgba(72, 49, 42, .10);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(255,255,255,.90), rgba(255,247,235,.88));
  box-shadow: 0 18px 55px rgba(74, 48, 39, .08);
  margin: 18px 0 14px;
}
.eyebrow {
  color: var(--app-coral);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.hero h1 {
  color: var(--app-ink) !important;
  font-size: clamp(30px, 4.5vw, 54px);
  line-height: 1.05;
  letter-spacing: -.035em;
  margin: 9px 0 12px;
}
.hero p { color: var(--app-muted); font-size: 16px; max-width: 760px; margin: 0; }
.step-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.step-chip {
  padding: 7px 11px; border-radius: 999px; background: rgba(240, 154, 75, .12);
  color: #8a4a28; font-size: 12px; font-weight: 650;
}
.panel {
  border: 1px solid rgba(72, 49, 42, .10) !important;
  border-radius: 20px !important;
  background: rgba(255,255,255,.80) !important;
  box-shadow: 0 10px 30px rgba(74, 48, 39, .055);
}
.generate-btn {
  background: linear-gradient(110deg, var(--app-coral), #ee8055) !important;
  color: white !important;
  border: 0 !important;
  min-height: 48px !important;
  font-weight: 750 !important;
  box-shadow: 0 9px 24px rgba(223, 91, 69, .25);
}
.status-card {
  border-left: 4px solid var(--app-amber) !important;
  background: rgba(255, 249, 239, .88) !important;
}
.prompt-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.result-image img { border-radius: 14px; background: #f4eee6; object-fit: contain !important; }
.small-note { color: var(--app-muted); font-size: 12px; }
footer { display: none !important; }
"""


def _format_metrics(
    quality: dict[str, Any],
    raw_quality: dict[str, Any] | None = None,
    color_guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    brightness_metrics: dict[str, Any] = {}
    if color_guidance:
        brightness = dict(color_guidance.get("brightness_floor", {}))
        if brightness:
            policy = dict(brightness.get("policy", {}))
            before = dict(brightness.get("before", {}))
            after = dict(brightness.get("after", {}))
            policy_labels = {
                "standard": "普通场景",
                "energetic": "活力/霓虹场景",
                "intentional_dark": "明确暗场景",
            }
            brightness_metrics = {
                "Raw 亮度策略": policy_labels.get(
                    str(policy.get("mode", "")),
                    str(policy.get("mode", "未知")),
                ),
                "Raw 自动增亮": (
                    "已启用" if brightness.get("applied") else "无需调整"
                ),
                "Raw 平均亮度（处理前）": round(
                    float(before.get("mean_luminance", 0.0)),
                    4,
                ),
                "Raw 平均亮度（最终）": round(
                    float(after.get("mean_luminance", 0.0)),
                    4,
                ),
                "Raw 最终低亮区域比例": round(
                    float(after.get("below_0_20_fraction", 0.0)) * 100,
                    3,
                ),
            }
    if quality.get("mapping_available") is False:
        metrics = {
            "SDL 映射状态": "未运行（缺少本地 SDL 色表）",
            "当前输出": "模块三 Raw 与模块五主题预览",
            "硬件色域保证": "无；配置 SDL 色表后才可验证",
            "输出尺寸": f'{quality.get("pixel_count", 0)} pixels',
        }
        metrics.update(brightness_metrics)
        return metrics
    metrics = {
        "SDL 严格无效像素": int(quality["strict_invalid_pixel_count"]),
        "映射前超出 SDL 色域比例": round(
            float(quality["before_xy_out_of_gamut_fraction"]) * 100, 3
        ),
        "视觉预览平均色差 Delta E 76": round(float(quality["mean_delta_e76"]), 3),
        "视觉预览 95% 色差 Delta E 76": round(float(quality["p95_delta_e76"]), 3),
        "视觉预览平均亮度变化": round(
            float(quality["mean_absolute_luminance_change"]), 6
        ),
        "输出尺寸": f'{quality["pixel_count"]} pixels',
    }
    if raw_quality and "isolated_chroma_fraction" in raw_quality:
        metrics["Raw 局部色斑比例"] = round(
            float(raw_quality["isolated_chroma_fraction"]) * 100,
            4,
        )
    if raw_quality and "broad_chroma_fraction" in raw_quality:
        metrics["Raw 宽区域色斑比例"] = round(
            float(raw_quality["broad_chroma_fraction"]) * 100,
            4,
        )
    if color_guidance and color_guidance.get(
        "post_guidance_anchor_color_error"
    ) is not None:
        metrics["Raw 主色区域误差"] = round(
            float(color_guidance["post_guidance_anchor_color_error"]) * 100,
            3,
        )
    if color_guidance and color_guidance.get("render_mode") == (
        "structured_horizontal_gradient_with_lora_luminance"
    ):
        final_metrics = dict(color_guidance.get("final_metrics", {}))
        if "mean_saturation" in final_metrics:
            metrics["Raw 平均饱和度"] = round(
                float(final_metrics["mean_saturation"]),
                4,
            )
        if "horizontal_structure_explained" in final_metrics:
            metrics["Raw 横向结构解释率"] = round(
                float(final_metrics["horizontal_structure_explained"]) * 100,
                3,
            )
        if "vertical_color_variation" in final_metrics:
            metrics["Raw 纵向颜色波动"] = round(
                float(final_metrics["vertical_color_variation"]) * 100,
                4,
            )
        metrics["LoRA 亮度纹理强度"] = round(
            float(color_guidance.get("effective_texture_strength", 0.0)),
            3,
        )
        plan = dict(color_guidance.get("plan", {}))
        if plan.get("dominant_color"):
            metrics["Raw 主色"] = str(plan["dominant_color"])
    metrics.update(brightness_metrics)
    return metrics


def build_demo(pipeline: LightingDemoPipeline | None = None) -> gr.Blocks:
    """Create the UI; dependency injection keeps construction testable."""

    pipeline = pipeline or LightingDemoPipeline(fast_mode=True)
    startup_status = "### 等待生成\n填写左侧信息后点击“生成完整光效”。"
    if not pipeline.sdl_path.is_file():
        startup_status += (
            "\n\n⚠️ 当前未检测到本地 SDL 色表。界面仍可生成模块三 Raw "
            "与模块五主题预览，但不会声称输出满足硬件色域。"
        )

    def _run_and_format(
        scene: str,
        width_mm: float,
        height_mm: float,
        space_size_m2: float | None,
        seed: int,
        steps: int,
        fixed_seed: bool,
        pattern_enabled: bool,
        pattern_strength: float,
        *,
        prompt_override: str | None = None,
        attributes_override: dict[str, Any] | None = None,
        progress: gr.Progress,
    ) -> tuple[Any, ...]:
        def update(value: float, text: str) -> None:
            progress(value, desc=text)

        try:
            result = pipeline.run(
                scene,
                width_mm,
                height_mm,
                space_size_m2,
                seed,
                steps,
                fixed_seed=fixed_seed,
                prompt_override=prompt_override,
                attributes_override=attributes_override,
                pattern_enabled=pattern_enabled,
                pattern_strength=pattern_strength,
                progress=update,
            )
        except Exception as exc:
            raise gr.Error(f"生成失败：{exc}") from exc

        status = (
            "### 生成完成\n"
            f"已输出 `{result.width} × {result.height}` 光效图；"
            f"本次有效种子为 `{result.effective_seed}`。"
        )
        if result.sdl_available:
            status += (
                f" SDL 严格无效像素为 "
                f"`{result.quality['strict_invalid_pixel_count']}`。"
            )
        if result.prompt_source in {
            "deepseek_concept_prompt",
            "deepseek_concept_prompt_cache",
        }:
            status += (
                "\n\n🎨 模块一大模型分别生成自然场景提示词和独立光效提示词。"
            )
        elif result.prompt_source == "local_concept_prompt_fallback":
            status += (
                "\n\n🎨 模块一大模型不可用或超时，本次使用本地应急概念提示词；"
                "光效提示词仍由独立的本地设计链路生成。"
            )
        elif result.prompt_source == "local_explicit_colors":
            status += "\n\n🎨 已严格保留用户输入中的明确颜色。"
        if result.color_guidance.get("correction_applied"):
            comparison = dict(result.color_guidance.get("color_comparison", {}))
            status += (
                "\n\n⚠️ 两条链路的主色差异超过安全阈值，已用概念图色板纠正光效；"
                f"检测色相差约 `{float(comparison.get('dominant_hue_delta_degrees', 0.0)):.1f}°`。"
            )
        elif result.color_guidance.get("independent_chains"):
            status += "\n\n✅ 两条链路颜色差异可接受，光效图保留模块一的独立配色。"
        if result.color_guidance.get("render_mode") == (
            "structured_horizontal_gradient_with_lora_luminance"
        ):
            plan = dict(result.color_guidance.get("plan", {}))
            texture_strength = float(
                result.color_guidance.get("effective_texture_strength", 0.0)
            )
            status += (
                "\n\n模块三已把独立光效提示词转换为横向主渐变；"
                f"主色为 `{plan.get('dominant_color', '未标注')}`，"
                f"LoRA 仅保留 `{texture_strength:.3f}` 强度的低频亮度纹理。"
            )
        if not result.sdl_available:
            status += f"\n\n⚠️ {result.sdl_notice}"
        else:
            if float(result.quality["before_xy_out_of_gamut_fraction"]) == 0.0:
                status += (
                    "\n\n所有 Raw 颜色均已位于 SDL 色域内，"
                    "模块四无需改变视觉预览。"
                )
            else:
                status += (
                    "\n\n⚠️ Raw 设计色有一部分超出 SDL 硬件色域；"
                    "系统保留鲜艳 Raw 图，并另外输出压缩后的硬件预览"
                    "与严格控制图。"
                )
            if float(result.quality["p95_delta_e76"]) > 45.0:
                status += (
                    " 本次硬件预览与 Raw 的色差较明显，"
                    "请以模块三 Raw 评估设计色，"
                    "以模块四结果评估实际硬件可实现效果。"
                )
            if result.sdl_notice:
                status += f"\n\n⚠️ {result.sdl_notice}"
        if result.artifact_cleanup.get("applied"):
            status += "\n\n模块三已执行局部色斑抑制与轻度平滑。"
        pattern_status = str(result.pattern_report.get("quality_status", ""))
        pattern_strength_used = float(
            result.pattern_report.get("effective_strength", 0.0)
        )
        if pattern_status == "accepted":
            status += (
                "\n\n模块五主题光场增强已通过，实际强度为 "
                f"`{pattern_strength_used:.3f}`。"
            )
        elif pattern_enabled:
            status += "\n\n模块五增强未通过安全检查，已自动绕过并保留模块三原图。"
        if result.similarity_retry_count:
            status += "\n\n检测到新结果与上一张过于相似，已自动更换种子重试。"
            if (
                result.similarity_difference is not None
                and result.similarity_difference < 0.03
            ):
                status += " 重试后仍较相似，建议进一步调整英文提示词的构图结构。"
        if result.sdl_retry_count:
            status += "\n\n首次 SDL 色域质量未通过，系统已更换种子并自动恢复。"
        total_seconds = float(result.timings.get("total_seconds", 0.0))
        if total_seconds:
            if result.deadline_met:
                status += f"\n\n⚡ 双图已在 `{total_seconds:.2f}` 秒内完成。"
            else:
                status += (
                    f"\n\n⚠️ 本次双图耗时 `{total_seconds:.2f}` 秒，超过 "
                    f"`{pipeline.time_budget_seconds:.1f}` 秒目标；详见性能报告。"
                )
        palette_notice = (
            f"⚠️ **色彩转换提示：** {result.palette_notice}"
            if result.palette_notice
            else "✅ 支持绿色、青色及其他色相；最终硬件范围由 SDL 映射处理。"
        )
        return (
            status,
            palette_notice,
            result.prompt,
            result.attributes,
            str(result.concept_image_path),
            str(result.raw_image_path),
            str(result.themed_image_path),
            str(result.sdl_preview_path),
            _format_metrics(
                result.quality,
                result.raw_quality,
                result.color_guidance,
            ),
            str(result.out_of_gamut_mask_path),
            str(result.sdl_control_path),
            str(result.archive_path),
            str(result.report_path),
        )

    def generate(
        scene: str,
        width_mm: float,
        height_mm: float,
        space_size_m2: float | None,
        seed: int,
        steps: int,
        fixed_seed: bool,
        pattern_enabled: bool,
        pattern_strength: float,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> tuple[Any, ...]:
        return _run_and_format(
            scene,
            width_mm,
            height_mm,
            space_size_m2,
            seed,
            steps,
            fixed_seed,
            pattern_enabled,
            pattern_strength,
            progress=progress,
        )

    def regenerate_with_edited_prompt(
        scene: str,
        width_mm: float,
        height_mm: float,
        space_size_m2: float | None,
        seed: int,
        steps: int,
        fixed_seed: bool,
        pattern_enabled: bool,
        pattern_strength: float,
        edited_prompt: str,
        current_attributes: dict[str, Any] | None,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> tuple[Any, ...]:
        return _run_and_format(
            scene,
            width_mm,
            height_mm,
            space_size_m2,
            seed,
            steps,
            fixed_seed,
            pattern_enabled,
            pattern_strength,
            prompt_override=edited_prompt,
            attributes_override=current_attributes,
            progress=progress,
        )

    with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero">
                  <div class="eyebrow">AI Lighting Lab</div>
                  <h1>一句中文，同时生成概念意象与可落地光色。</h1>
                  <p>系统用两条独立链路分别生成实体概念图与宽幅光色图；只有双方主色严重冲突时，才使用概念图色板纠正光效，随后完成主题增强与 SDL 色域处理。</p>
                  <div class="step-strip">
                    <span class="step-chip">01 中文场景 → 双提示词</span>
                    <span class="step-chip">02 独立实体概念图</span>
                    <span class="step-chip">03 独立光效 → 严重偏色兜底</span>
                    <span class="step-chip">05 低频主题光场增强</span>
                    <span class="step-chip">04 SDL 物理色域映射</span>
                  </div>
                </section>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("### 设计输入")
                    scene = gr.Textbox(
                        label="中文场景描述",
                        value="当前场景是一个飘香的咖啡厅，人们在这里能够惬意地品尝咖啡。",
                        lines=4,
                        placeholder="描述空间、氛围和使用者感受……",
                    )
                    with gr.Row():
                        width_mm = gr.Number(
                            label="灯具宽度（mm）", value=1220, minimum=1
                        )
                        height_mm = gr.Number(
                            label="灯具高度（mm）", value=370, minimum=1
                        )
                    space_size_m2 = gr.Textbox(
                        label="空间面积（m²，可选）",
                        value="",
                        placeholder="例如：80；不确定可留空",
                    )
                    with gr.Accordion("高级参数", open=False):
                        seed = gr.Number(
                            label="随机种子",
                            value=20260724,
                            precision=0,
                            minimum=0,
                        )
                        steps = gr.Slider(
                            label="概念图快速推理步数",
                            minimum=2,
                            maximum=8,
                            value=4,
                            step=1,
                        )
                        fixed_seed = gr.Checkbox(
                            label="固定随机种子（复现实验）",
                            value=False,
                            info=(
                                "默认关闭：不同中文场景会得到不同但可复现的有效种子；"
                                "开启后才会严格使用上方数值。"
                            ),
                        )
                        pattern_enabled = gr.Checkbox(
                            label="启用模块五主题光场增强",
                            value=True,
                            info="未通过质量检查时会自动降低强度或安全绕过。",
                        )
                        pattern_strength = gr.Slider(
                            label="模块五增强强度",
                            minimum=0.0,
                            maximum=0.18,
                            value=0.10,
                            step=0.01,
                        )
                    submit = gr.Button(
                        "生成完整光效",
                        variant="primary",
                        elem_classes=["generate-btn"],
                    )
                    gr.Markdown(
                        "<span class='small-note'>模型在网页启动前完成加载；目标是在 6 秒内同时输出概念图和光色图。</span>"
                    )

                with gr.Column(scale=7):
                    status = gr.Markdown(
                        startup_status,
                        elem_classes=["panel", "status-card"],
                    )
                    palette_notice = gr.Markdown(
                        "支持绿色、青色及其他色相；概念图与光色图默认独立配色，仅在严重偏色时自动纠正。"
                    )
                    prompt = gr.Textbox(
                        label="独立光效链路的英文提示词（可编辑）",
                        lines=6,
                        interactive=True,
                        info=(
                            "一键生成后可修改左、中、右颜色和主色，"
                            "再点击下方按钮；不会再次调用 DeepSeek。"
                        ),
                        elem_classes=["prompt-box"],
                    )
                    rerun = gr.Button(
                        "使用修改后的提示词重新生成",
                        variant="secondary",
                    )
                    attributes = gr.JSON(
                        label="模块一 · 结构化参数",
                        open=False,
                    )

            gr.Markdown("## 概念意象与最终光色")
            with gr.Row(equal_height=True):
                concept_image = gr.Image(
                    label="实体场景概念图",
                    type="filepath",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    elem_classes=["result-image"],
                )
                sdl_preview = gr.Image(
                    label="最终宽幅光色图",
                    type="filepath",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    elem_classes=["result-image"],
                )

            with gr.Accordion("查看生成与硬件工程输出", open=False):
                with gr.Row(equal_height=True):
                    raw_image = gr.Image(
                        label="模块三 · 参考风格主渐变 Raw",
                        type="filepath",
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_classes=["result-image"],
                    )
                    themed_image = gr.Image(
                        label="模块五 · 低频主题增强",
                        type="filepath",
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_classes=["result-image"],
                    )

            with gr.Row(equal_height=False):
                metrics = gr.JSON(label="关键质量指标", open=True)
                with gr.Column():
                    archive = gr.File(label="下载本次完整结果（ZIP）")
                    report = gr.File(label="下载可复现报告（JSON）")

            with gr.Accordion("查看模块四工程输出", open=False):
                gr.Markdown(
                    "白色区域表示 Raw 图中超出 SDL xy 色域的像素；"
                    "视觉预览只连续压缩这些超域像素，色域内像素保持 Raw 不变。"
                    "严格控制图的每个像素都来自 SDL RGB 表，因此它可能出现离散纹理，"
                    "用于硬件控制而不是人眼成图。"
                )
                with gr.Row():
                    gamut_mask = gr.Image(
                        label="超色域掩膜",
                        type="filepath",
                        interactive=False,
                        buttons=["download", "fullscreen"],
                    )
                    control_image = gr.Image(
                        label="严格 SDL 控制图（缺少色表时显示不可用占位图）",
                        type="filepath",
                        interactive=False,
                        buttons=["download", "fullscreen"],
                    )

        submit.click(
            fn=generate,
            inputs=[
                scene,
                width_mm,
                height_mm,
                space_size_m2,
                seed,
                steps,
                fixed_seed,
                pattern_enabled,
                pattern_strength,
            ],
            outputs=[
                status,
                palette_notice,
                prompt,
                attributes,
                concept_image,
                raw_image,
                themed_image,
                sdl_preview,
                metrics,
                gamut_mask,
                control_image,
                archive,
                report,
            ],
            concurrency_limit=1,
            concurrency_id="diffusion-model",
            show_progress="full",
            scroll_to_output=True,
        )
        rerun.click(
            fn=regenerate_with_edited_prompt,
            inputs=[
                scene,
                width_mm,
                height_mm,
                space_size_m2,
                seed,
                steps,
                fixed_seed,
                pattern_enabled,
                pattern_strength,
                prompt,
                attributes,
            ],
            outputs=[
                status,
                palette_notice,
                prompt,
                attributes,
                concept_image,
                raw_image,
                themed_image,
                sdl_preview,
                metrics,
                gamut_mask,
                control_image,
                archive,
                report,
            ],
            concurrency_limit=1,
            concurrency_id="diffusion-model",
            show_progress="full",
            scroll_to_output=True,
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--generation-mode",
        choices=("fast", "quality"),
        default="fast",
        help="fast returns a concept image plus a concept-derived light field",
    )
    parser.add_argument("--time-budget-seconds", type=float, default=6.0)
    parser.add_argument(
        "--auto-palette-timeout-seconds",
        type=float,
        default=3.0,
        help="maximum DeepSeek wait when fast mode has no explicit color",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sdl-path",
        type=Path,
        default=DEFAULT_SDL_PATH,
        help="Path to the private SDL color table; preview mode is used when absent",
    )
    parser.add_argument(
        "--require-sdl",
        action="store_true",
        help="Refuse generation when the SDL color table is missing",
    )
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.require_sdl and not args.sdl_path.expanduser().is_file():
        raise SystemExit(
            "SDL color table does not exist: "
            f"{args.sdl_path.expanduser().resolve()}"
        )
    pipeline = LightingDemoPipeline(
        output_root=args.output_root,
        device=args.device,
        sdl_path=args.sdl_path,
        allow_missing_sdl=not args.require_sdl,
        fast_mode=args.generation_mode == "fast",
        time_budget_seconds=args.time_budget_seconds,
        auto_palette_timeout_seconds=args.auto_palette_timeout_seconds,
    )
    pipeline.warmup()
    demo = build_demo(pipeline)
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="stone",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "PingFang SC", "sans-serif"],
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        inbrowser=args.inbrowser,
        show_error=True,
        theme=theme,
        css=CSS,
        allowed_paths=[str(args.output_root.expanduser().resolve())],
        footer_links=[],
    )


if __name__ == "__main__":
    main()
