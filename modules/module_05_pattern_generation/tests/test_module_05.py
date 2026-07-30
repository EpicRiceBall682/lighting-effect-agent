from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from modules.module_05_pattern_generation.src.generator import (
    PatternGenerator,
    PatternQualityPolicy,
)
from modules.module_05_pattern_generation.src.pattern_renderer import render_pattern
from modules.module_05_pattern_generation.src.theme_extractor import extract_theme


MODULE_01 = {
    "density": "middle",
    "m_intensity": 70,
    "k_intensity": 85,
    "a_intensity": 60,
    "effect": (
        "Wide panoramic abstract light texture with light blue in the upper area, "
        "pale pink near the center, and pale golden yellow toward the lower right."
    ),
}


class ThemeExtractorTests(unittest.TestCase):
    def test_scene_types_receive_distinct_visual_signatures(self):
        sea = extract_theme("沙滩和大海", MODULE_01)
        coffee = extract_theme("温暖惬意的咖啡厅", MODULE_01)
        sports = extract_theme("湖人篮球比赛的热血时刻", MODULE_01)
        entrance = extract_theme("酒店入口的日出唤醒光", MODULE_01)
        self.assertEqual(sea.motif, "flowing")
        self.assertEqual(coffee.motif, "breathing")
        self.assertEqual(sports.motif, "flowing")
        self.assertEqual(entrance.motif, "radiant")
        self.assertEqual(len({sea.motif, coffee.motif, entrance.motif}), 3)

    def test_palette_uses_only_safe_bright_colors(self):
        attributes = extract_theme("绿色森林", MODULE_01)
        for red, green, blue in attributes.palette:
            self.assertGreater(max(red, green, blue), 150)
            self.assertFalse(green > red * 1.25 and green > blue * 1.15)

    def test_strength_validation_rejects_overpowering_pattern(self):
        with self.assertRaises(ValueError):
            extract_theme("浪漫婚礼", MODULE_01, pattern_strength=0.9)


class PatternRendererTests(unittest.TestCase):
    def setUp(self):
        self.base = Image.new("RGB", (320, 96), (205, 190, 180))

    def test_same_seed_is_deterministic(self):
        attributes = extract_theme("温暖惬意的咖啡厅", MODULE_01)
        first, _ = render_pattern(self.base, attributes, seed=42)
        second, _ = render_pattern(self.base, attributes, seed=42)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_different_themes_produce_different_images(self):
        coffee, _ = render_pattern(
            self.base,
            extract_theme("温暖惬意的咖啡厅", MODULE_01),
            seed=42,
        )
        sports, _ = render_pattern(
            self.base,
            extract_theme("热血篮球比赛", MODULE_01),
            seed=42,
        )
        difference = np.mean(
            np.abs(
                np.asarray(coffee, dtype=np.float32)
                - np.asarray(sports, dtype=np.float32)
            )
        )
        self.assertGreater(difference, 0.1)

    def test_zero_strength_preserves_every_pixel(self):
        attributes = extract_theme(
            "浪漫婚礼",
            MODULE_01,
            pattern_strength=0.0,
        )
        rendered, report = render_pattern(self.base, attributes, seed=42)
        self.assertEqual(rendered.tobytes(), self.base.tobytes())
        self.assertFalse(report["applied"])

    def test_enhancement_preserves_rgb_chromaticity(self):
        base = Image.new("RGB", (320, 96), (120, 160, 200))
        rendered, report = render_pattern(
            base,
            extract_theme("动态品牌流动光", MODULE_01, pattern_strength=0.12),
            seed=42,
        )
        pixels = np.asarray(rendered, dtype=np.float32)
        normalized = pixels / np.maximum(pixels.sum(axis=2, keepdims=True), 1.0)
        expected = np.asarray((120, 160, 200), dtype=np.float32)
        expected /= expected.sum()
        self.assertLess(float(np.max(np.abs(normalized - expected))), 0.006)
        self.assertEqual(
            report["color_mode"],
            "chromaticity_preserving_linear_rgb_gain",
        )

    def test_brightening_does_not_clip_one_channel_and_shift_hue(self):
        from modules.module_04_gamut_mapping.src.color_spaces import rgb8_to_xyy

        base = Image.new("RGB", (320, 96), (250, 231, 153))
        rendered, _report = render_pattern(
            base,
            extract_theme("温暖中心聚焦入口", MODULE_01, pattern_strength=0.18),
            seed=42,
        )
        before_xy = rgb8_to_xyy(np.asarray(base, dtype=np.uint8))[..., :2]
        after_xy = rgb8_to_xyy(np.asarray(rendered, dtype=np.uint8))[..., :2]
        shift = np.linalg.norm(after_xy - before_xy, axis=2)
        self.assertLess(float(np.max(shift)), 0.0015)

    def test_generator_writes_image_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PatternGenerator().generate(
                self.base,
                scene="沙滩和大海",
                module_01_attributes=MODULE_01,
                seed=42,
                output_dir=Path(directory),
            )
            self.assertTrue(result.image_path.is_file())
            with Image.open(result.image_path) as opened:
                self.assertEqual(opened.size, self.base.size)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["attributes"]["theme"], "flowing")
            self.assertTrue(manifest["render"]["applied"])

    def test_generator_reduces_strength_or_bypasses_when_policy_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PatternGenerator().generate(
                self.base,
                scene="动态品牌流动光",
                module_01_attributes=MODULE_01,
                seed=42,
                output_dir=Path(directory),
                pattern_strength=0.18,
                quality_policy=PatternQualityPolicy(
                    maximum_mean_absolute_change=0.000001,
                    maximum_mean_luminance_change=0.000001,
                ),
            )
            self.assertEqual(result.attributes.pattern_strength, 0.0)
            self.assertEqual(result.render_report["quality_status"], "bypassed")
            with Image.open(result.image_path) as opened:
                self.assertEqual(opened.convert("RGB").tobytes(), self.base.tobytes())


if __name__ == "__main__":
    unittest.main()
