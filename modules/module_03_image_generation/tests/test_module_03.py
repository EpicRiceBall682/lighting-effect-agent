from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from modules.module_03_image_generation.src.cli import _dimensions, build_parser, prompt_from_json
from modules.module_03_image_generation.src.config import GenerationConfig
from modules.module_03_image_generation.src.generator import (
    attribute_prompt_fragments,
    build_effective_prompt,
    enrich_prompt,
)
from modules.module_03_image_generation.src.generator import LightEffectGenerator
from modules.module_03_image_generation.src.image_geometry import dimensions_from_fixture
from modules.module_03_image_generation.src.prompt_guidance import (
    apply_prompt_color_guidance,
    extract_color_anchors,
)
from modules.module_03_image_generation.src.quality import (
    analyze_image_quality,
    broad_chroma_fraction,
    suppress_broad_chroma_artifacts,
    suppress_isolated_chroma_artifacts,
)
from modules.module_03_image_generation.src.structured_gradient import (
    build_structured_gradient_plan,
    render_base_gradient,
    render_structured_gradient,
    structured_gradient_metrics,
)
from modules.module_03_image_generation.src.scene_to_image_cli import (
    build_parser as build_scene_parser,
    run_scene_pipeline,
)


class GeometryTests(unittest.TestCase):
    def test_fixture_ratio_is_preserved(self):
        width, height = dimensions_from_fixture(1220, 370)
        self.assertEqual(width, 1024)
        self.assertEqual(height % 8, 0)
        self.assertAlmostEqual(width / height, 1220 / 370, delta=0.05)

    def test_invalid_fixture_is_rejected(self):
        with self.assertRaises(ValueError):
            dimensions_from_fixture(1220, 0)

    def test_extreme_fixture_ratio_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            dimensions_from_fixture(1, 100_000)

    def test_invalid_dimension_multiple_is_rejected_cleanly(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            dimensions_from_fixture(1220, 370, multiple=0)


class ConfigTests(unittest.TestCase):
    def test_dimensions_must_be_divisible_by_eight(self):
        with self.assertRaises(ValueError):
            GenerationConfig(width=1000, height=319)

    def test_enrichment_preserves_original_prompt(self):
        prompt = "Warm pale yellow to soft orange gradient"
        self.assertTrue(enrich_prompt(prompt).startswith(prompt))
        self.assertIn("no objects", enrich_prompt(prompt))

    def test_attributes_change_the_effective_prompt(self):
        attributes = {
            "density": "high",
            "m_intensity": 60,
            "k_intensity": 85,
            "a_intensity": 72,
        }
        fragments = attribute_prompt_fragments(attributes)
        effective = enrich_prompt("A structured warm gradient", attributes)
        self.assertIn("broad three-stop horizontal gradient", effective)
        self.assertIn("soft focal contrast", effective)
        self.assertEqual(len(fragments), 2)

    def test_tokenizer_limit_drops_optional_controls_instead_of_truncating(self):
        class FakeTokenizer:
            model_max_length = 12

            def __call__(self, text, **_kwargs):
                return {"input_ids": text.replace(",", "").split()}

        attributes = {
            "density": "high",
            "m_intensity": 60,
            "k_intensity": 85,
            "a_intensity": 72,
        }
        effective, controls, token_count = build_effective_prompt(
            "Warm upper glow with pale yellow lower gradient",
            attributes,
            tokenizer=FakeTokenizer(),
        )
        self.assertLessEqual(token_count, 12)
        self.assertIn("no objects", effective)
        self.assertLess(len(controls), 2)


class CliTests(unittest.TestCase):
    def test_reads_module_one_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.json"
            path.write_text(
                json.dumps(
                    {
                        "density": "middle",
                        "m_intensity": 70,
                        "k_intensity": 80,
                        "a_intensity": 60,
                        "effect": (
                            "Wide panoramic abstract light texture with warm yellow at the "
                            "center, soft pink spreading toward both sides, a broad horizontal "
                            "gradient, misty luminous glow, diffused bloom, evenly bright "
                            "illumination, and a calm welcoming atmosphere."
                        ),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                prompt_from_json(path),
                (
                    "Wide panoramic abstract light texture with warm yellow at the center, "
                    "soft pink spreading toward both sides, a broad horizontal gradient, "
                    "misty luminous glow, diffused bloom, evenly bright illumination, and "
                    "a calm welcoming atmosphere."
                ),
            )

    def test_explicit_dimensions(self):
        args = build_parser().parse_args(["--prompt", "soft gradient", "--width", "1024", "--height", "320"])
        self.assertEqual(_dimensions(args), (1024, 320))

    def test_scene_pipeline_accepts_one_chinese_input(self):
        args = build_scene_parser().parse_args(
            [
                "--scene",
                "傍晚的海边餐厅，希望呈现温暖浪漫的氛围。",
                "--width-mm",
                "1220",
                "--height-mm",
                "370",
            ]
        )
        self.assertIn("海边餐厅", args.scene)
        self.assertEqual(args.lora_scale, 0.8)

    def test_scene_pipeline_passes_all_module_one_attributes_to_generator(self):
        from types import SimpleNamespace
        from modules.module_01_prompt_agent.src.schemas import LightingEffectAttributes

        attributes = LightingEffectAttributes(
            density="middle",
            m_intensity=65,
            k_intensity=80,
            a_intensity=70,
            effect="A pale yellow to warm orange gradient with a soft relaxing luminous glow.",
        )

        class FakeAgent:
            def generate(self, *args, **kwargs):
                return attributes

        captured = {}

        class FakeGenerator:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def generate(self, prompt, **kwargs):
                captured["prompt"] = prompt
                captured["generate"] = kwargs
                output = kwargs["output_dir"]
                return SimpleNamespace(image_path=output / "fake.png", manifest_path=output / "fake.json")

        with tempfile.TemporaryDirectory() as directory:
            args = build_scene_parser().parse_args(
                [
                    "--scene",
                    "温暖的咖啡厅",
                    "--width-mm",
                    "1220",
                    "--height-mm",
                    "370",
                    "--output-dir",
                    directory,
                ]
            )
            _result, prompt_path, returned = run_scene_pipeline(
                args,
                prompt_agent=FakeAgent(),
                generator_factory=FakeGenerator,
            )
            self.assertEqual(returned, attributes)
            self.assertTrue(prompt_path.is_file())
            self.assertEqual(captured["generate"]["source_attributes"], attributes.to_dict())
            self.assertEqual(captured["generate"]["config"].height, 312)


class QualityTests(unittest.TestCase):
    def test_structured_plan_keeps_dominant_color_across_center_and_right(self):
        from PIL import Image

        prompt = (
            "Wide panoramic organizer-style color field with electric purple on the left "
            "and dominant vivid magenta across the center and right, forming a saturated "
            "smooth horizontal gradient with uniform vertical color and clean illumination."
        )
        plan = build_structured_gradient_plan(
            prompt,
            Image.new("RGB", (96, 32), (120, 120, 120)),
        )

        self.assertEqual(plan.direction, "horizontal")
        self.assertEqual(plan.dominant_color, "vivid magenta")
        self.assertEqual([stop.position for stop in plan.stops], [0.0, 0.5, 1.0])
        self.assertEqual(plan.stops[-1].rgb, plan.stops[-2].rgb)

    def test_bright_blue_is_preserved_as_the_dominant_color(self):
        from PIL import Image

        prompt = (
            "Wide panoramic organizer-style color field with pale yellow on the left and "
            "dominant bright blue across the center and right, forming a clean smooth "
            "horizontal gradient with uniform vertical color, calm illumination, clear "
            "identity, and an uninterrupted surface throughout."
        )
        plan = build_structured_gradient_plan(
            prompt,
            Image.new("RGB", (96, 64), (120, 120, 120)),
        )

        self.assertEqual(plan.dominant_color, "bright blue")
        self.assertEqual(plan.stops[0].color_name, "pale yellow")
        self.assertEqual(plan.stops[1].color_name, "bright blue")
        self.assertEqual(plan.stops[2].color_name, "bright blue")
        rendered = render_base_gradient(plan, width=320, height=96)
        quality = analyze_image_quality(rendered)
        self.assertLessEqual(quality.forbidden_hue_fraction, 0.005)
        self.assertGreater(rendered.getpixel((319, 48))[2], 200)

    def test_dominant_modifier_does_not_leak_to_the_previous_color(self):
        from PIL import Image

        prompt = (
            "Wide panoramic organizer-style field with pale yellow left, dominant "
            "bright blue center and right, forming one clean smooth horizontal gradient "
            "with uniform vertical color and an uninterrupted luminous surface throughout."
        )
        plan = build_structured_gradient_plan(
            prompt,
            Image.new("RGB", (128, 64), (200, 200, 200)),
        )
        self.assertEqual(plan.dominant_color, "bright blue")
        self.assertEqual(plan.stops[0].role, "secondary")
        self.assertEqual(plan.stops[1].role, "primary")

    def test_unsupported_blue_term_fails_instead_of_becoming_single_color(self):
        from PIL import Image

        with self.assertRaisesRegex(ValueError, "cannot parse: cerulean blue"):
            build_structured_gradient_plan(
                "Pale yellow on the left and dominant cerulean blue across the right.",
                Image.new("RGB", (96, 64), (120, 120, 120)),
            )

    def test_structured_renderer_rejects_lora_chroma_but_keeps_weak_luminance(self):
        import numpy as np
        from PIL import Image

        yy, xx = np.ogrid[:96, :320]
        source = np.empty((96, 320, 3), dtype=np.uint8)
        source[:] = (110, 130, 150)
        source[((xx - 245) / 62) ** 2 + ((yy - 70) / 24) ** 2 <= 1] = (
            60,
            230,
            90,
        )
        prompt = (
            "Wide panoramic organizer-style color field with electric purple on the left "
            "and dominant vivid magenta across the center and right, forming a saturated "
            "smooth horizontal gradient with uniform vertical color and clean illumination."
        )
        rendered, report = render_structured_gradient(
            Image.fromarray(source, mode="RGB"),
            prompt,
            texture_strength=0.10,
        )
        metrics = structured_gradient_metrics(rendered)
        left = rendered.getpixel((0, 48))
        right = rendered.getpixel((319, 48))

        self.assertGreater(left[2], left[0])
        self.assertGreater(right[0], right[2])
        self.assertGreater(right[2], right[1])
        self.assertLessEqual(metrics["vertical_color_variation"], 0.018)
        self.assertGreater(metrics["horizontal_structure_explained"], 0.98)
        self.assertLessEqual(report["effective_texture_strength"], 0.10)
        self.assertEqual(
            report["render_mode"],
            "structured_horizontal_gradient_with_lora_luminance",
        )

    def test_spatial_color_anchors_create_prompt_faithful_vertical_gradient(self):
        from PIL import Image

        prompt = (
            "Smooth gradient with light blue across the upper area and "
            "pale golden yellow across the lower area."
        )
        anchors = extract_color_anchors(prompt)
        self.assertEqual({anchor.position for anchor in anchors}, {0.0, 1.0})

        guided, report = apply_prompt_color_guidance(
            Image.new("RGB", (64, 32), (190, 175, 180)),
            prompt,
            strength=1.0,
        )
        top = guided.getpixel((32, 0))
        bottom = guided.getpixel((32, 31))
        self.assertGreater(top[2], top[0])
        self.assertGreater(bottom[0], bottom[2])
        self.assertTrue(report["applied"])

    def test_color_guidance_is_not_applied_without_two_spatial_anchors(self):
        from PIL import Image

        source = Image.new("RGB", (32, 16), (120, 140, 160))
        guided, report = apply_prompt_color_guidance(
            source,
            "A soft pale yellow luminous glow.",
        )
        self.assertEqual(guided.tobytes(), source.tobytes())
        self.assertFalse(report["applied"])

    def test_default_color_guidance_keeps_more_lora_texture(self):
        from PIL import Image

        source = Image.new("RGB", (32, 16), (100, 100, 100))
        guided, report = apply_prompt_color_guidance(
            source,
            "Light blue across the upper area and pale yellow across the lower area.",
        )
        self.assertEqual(report["requested_strength"], 0.64)
        self.assertGreater(report["effective_strength"], 0.0)
        self.assertLessEqual(report["effective_strength"], 0.64)
        self.assertNotEqual(guided.tobytes(), source.tobytes())

    def test_matching_raw_layout_avoids_unnecessary_color_overlay(self):
        import numpy as np
        from PIL import Image

        top = np.asarray((151, 203, 238), dtype=np.float32)
        bottom = np.asarray((250, 231, 153), dtype=np.float32)
        line = np.linspace(top, bottom, 32)
        pixels = np.repeat(line[:, None, :], 64, axis=1).astype(np.uint8)
        source = Image.fromarray(pixels, mode="RGB")
        guided, report = apply_prompt_color_guidance(
            source,
            "Light blue across the upper area and pale yellow across the lower area.",
        )

        self.assertEqual(report["effective_strength"], 0.0)
        self.assertEqual(guided.tobytes(), source.tobytes())

    def test_soft_purple_is_recognized_as_a_color_anchor(self):
        anchors = extract_color_anchors(
            "Light blue across the upper area, soft purple near the center, "
            "and pale pink across the lower area."
        )
        self.assertIn("soft purple", {anchor.color_name for anchor in anchors})

    def test_broad_main_color_outweighs_local_diffusion_color(self):
        from PIL import Image

        prompt = (
            "Light blue across the upper area, soft ivory cloud-like diffusion "
            "in the upper center, fading to pale yellow across the lower area."
        )
        anchors = extract_color_anchors(prompt)
        blue = next(anchor for anchor in anchors if anchor.color_name == "light blue")
        ivory = next(anchor for anchor in anchors if anchor.color_name == "ivory")
        self.assertEqual(blue.scope, "broad")
        self.assertEqual(ivory.scope, "local")
        self.assertGreater(blue.weight, ivory.weight * 4)

        guided, report = apply_prompt_color_guidance(
            Image.new("RGB", (64, 32), (150, 150, 160)),
            prompt,
            strength=1.0,
        )
        top = guided.getpixel((32, 0))
        self.assertGreater(top[2], top[0] + 35)
        self.assertLess(
            report["post_guidance_anchor_color_error"],
            report["pre_guidance_anchor_color_error"],
        )

    def test_nearly_black_image_is_detected(self):
        from PIL import Image

        report = analyze_image_quality(Image.new("RGB", (64, 64), (0, 0, 0)))
        self.assertGreaterEqual(report.near_black_fraction, 0.98)

    def test_green_image_is_reported(self):
        from PIL import Image

        report = analyze_image_quality(Image.new("RGB", (64, 64), (80, 220, 120)))
        self.assertGreater(report.forbidden_hue_fraction, 0.9)
        self.assertTrue(report.warnings)

    def test_isolated_chroma_spot_is_detected_and_reduced(self):
        import numpy as np
        from PIL import Image

        pixels = np.full((96, 256, 3), (220, 175, 165), dtype=np.uint8)
        pixels[38:48, 190:200] = (255, 30, 180)
        source = Image.fromarray(pixels, mode="RGB")
        before = analyze_image_quality(source)
        cleaned, cleanup = suppress_isolated_chroma_artifacts(source)
        after = analyze_image_quality(cleaned)

        self.assertGreater(before.isolated_chroma_fraction, 0.001)
        self.assertTrue(cleanup["applied"])
        self.assertLess(
            after.isolated_chroma_fraction,
            before.isolated_chroma_fraction,
        )

    def test_artifact_cleanup_preserves_broad_gradient_direction(self):
        import numpy as np
        from PIL import Image

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        pixels = np.empty((96, 256, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.rint(180 + 60 * x)
        pixels[:, :, 1] = np.rint(130 + 35 * x)
        pixels[:, :, 2] = np.rint(170 - 50 * x)
        cleaned, _report = suppress_isolated_chroma_artifacts(
            Image.fromarray(pixels, mode="RGB")
        )

        left = cleaned.getpixel((8, 48))
        right = cleaned.getpixel((247, 48))
        self.assertGreater(right[0], left[0] + 40)
        self.assertLess(right[2], left[2] - 30)

    def test_broad_chroma_blotch_is_detected_without_rejecting_smooth_gradient(self):
        import numpy as np
        from PIL import Image

        x = np.linspace(0.0, 1.0, 320, dtype=np.float32)
        smooth = np.empty((120, 320, 3), dtype=np.uint8)
        smooth[:, :, 0] = np.rint(175 + 55 * x)
        smooth[:, :, 1] = np.rint(130 + 35 * x)
        smooth[:, :, 2] = np.rint(185 - 45 * x)
        blotchy = smooth.copy()
        yy, xx = np.ogrid[:120, :320]
        patch = ((xx - 245) / 55) ** 2 + ((yy - 86) / 24) ** 2 <= 1
        blotchy[patch] = (250, 92, 145)

        smooth_report = analyze_image_quality(Image.fromarray(smooth, mode="RGB"))
        blotchy_report = analyze_image_quality(Image.fromarray(blotchy, mode="RGB"))

        self.assertLessEqual(smooth_report.broad_chroma_fraction, 0.003)
        self.assertGreater(blotchy_report.broad_chroma_fraction, 0.003)
        self.assertGreater(
            broad_chroma_fraction(Image.fromarray(blotchy, mode="RGB")),
            broad_chroma_fraction(Image.fromarray(smooth, mode="RGB")),
        )

        cleaned, cleanup = suppress_broad_chroma_artifacts(
            Image.fromarray(blotchy, mode="RGB")
        )
        self.assertTrue(cleanup["applied"])
        self.assertLessEqual(broad_chroma_fraction(cleaned), 0.003)

    def test_artifact_cleanup_is_byte_exact_when_nothing_is_detected(self):
        from PIL import Image

        source = Image.new("RGB", (128, 64), (220, 175, 165))
        cleaned, report = suppress_isolated_chroma_artifacts(source)
        self.assertEqual(cleaned.tobytes(), source.tobytes())
        self.assertEqual(report["detected_fraction"], 0.0)
        self.assertEqual(report["changed_pixel_fraction"], 0.0)
        self.assertFalse(report["applied"])

    def test_generator_manifest_keeps_module_one_attributes_and_quality(self):
        from types import SimpleNamespace
        from PIL import Image

        class FakePipeline:
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), (255, 180, 100))])

        attributes = {
            "density": "middle",
            "m_intensity": 70,
            "k_intensity": 80,
            "a_intensity": 60,
            "effect": "A warm luminous gradient.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "fake.safetensors"
            weight.write_bytes(b"fake weight")
            generator = LightEffectGenerator(
                lora_path=weight,
                pipeline=FakePipeline(),
                selected_device="test",
            )
            result = generator.generate(
                attributes["effect"],
                output_dir=root,
                config=GenerationConfig(width=64, height=64),
                source_attributes=attributes,
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["module_01_attributes"], attributes)
            self.assertIn("balanced three-stop horizontal gradient", manifest["effective_prompt"])
            self.assertTrue(manifest["module_01_prompt_controls"])
            self.assertIsNone(manifest["effective_prompt_token_count"])
            self.assertIn("mean_luminance", manifest["quality"])
            self.assertIn("artifact_cleanup", manifest)
            self.assertIn("isolated_chroma_fraction", manifest["quality"])
            self.assertEqual(manifest["quality_status"], "accepted")
            self.assertEqual(manifest["generation_mode"], "structured_gradient")
            self.assertIn("plan", manifest["prompt_color_guidance"])
            self.assertTrue(Path(manifest["diffusion_raw_path"]).is_file())
            self.assertTrue(Path(manifest["guided_image_path"]).is_file())
            self.assertEqual(result.seed, manifest["seed"])

    def test_generator_retries_a_quality_rejected_image(self):
        from types import SimpleNamespace
        from PIL import Image

        class FakePipeline:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                color = (80, 220, 120) if self.calls == 1 else (255, 180, 100)
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), color)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "fake.safetensors"
            weight.write_bytes(b"fake weight")
            pipeline = FakePipeline()
            generator = LightEffectGenerator(
                lora_path=weight,
                pipeline=pipeline,
                selected_device="test",
            )
            result = generator.generate(
                "Warm luminous gradient.",
                output_dir=root,
                config=GenerationConfig(width=64, height=64, seed=42),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(pipeline.calls, 2)
            self.assertEqual(result.quality_retry_count, 1)
            self.assertEqual(len(manifest["quality_attempts"]), 2)
            self.assertTrue(manifest["quality_attempts"][0]["failures"])
            self.assertFalse(manifest["quality_attempts"][1]["failures"])

    def test_vivid_structured_palette_does_not_reroll_for_expected_luminance(self):
        from types import SimpleNamespace
        from PIL import Image

        class FakePipeline:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    images=[Image.new("RGB", (96, 64), (125, 125, 125))]
                )

        prompt = (
            "Wide panoramic organizer-style color field with electric purple on the left "
            "and dominant vivid magenta across the center and right, forming a saturated "
            "smooth horizontal gradient with uniform vertical color, clean illumination, "
            "energetic rhythm, strong visual identity, and an uninterrupted surface."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "fake.safetensors"
            weight.write_bytes(b"fake weight")
            pipeline = FakePipeline()
            generator = LightEffectGenerator(
                lora_path=weight,
                pipeline=pipeline,
                selected_device="test",
            )
            result = generator.generate(
                prompt,
                output_dir=root,
                config=GenerationConfig(width=96, height=64, seed=42),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(pipeline.calls, 1)
            self.assertEqual(result.quality_retry_count, 0)
            self.assertEqual(
                manifest["quality_policy"]["minimum_mean_luminance"],
                0.22,
            )

    def test_generator_reports_every_failed_quality_attempt(self):
        from types import SimpleNamespace
        from PIL import Image

        class FakePipeline:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    images=[Image.new("RGB", (64, 64), (80, 220, 120))]
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "fake.safetensors"
            weight.write_bytes(b"fake weight")
            pipeline = FakePipeline()
            generator = LightEffectGenerator(
                lora_path=weight,
                pipeline=pipeline,
                selected_device="test",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                r"after 3 attempts: attempt 0 seed 42: .*"
                r"\| attempt 1 seed 130405: .*"
                r"\| attempt 2 seed 260768:",
            ):
                generator.generate(
                    "Warm luminous gradient.",
                    output_dir=root,
                    config=GenerationConfig(width=64, height=64, seed=42),
                )
            self.assertEqual(pipeline.calls, 3)
            self.assertFalse(list(root.glob("raw_light_effect*.png")))


if __name__ == "__main__":
    unittest.main()
