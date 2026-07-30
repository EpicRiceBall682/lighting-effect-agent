from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from modules.module_04_gamut_mapping.src.color_spaces import (
    lab_to_rgb8,
    rgb8_to_lab,
    srgb_to_xyz,
    xyz_to_srgb,
)
from modules.module_04_gamut_mapping.src.cli import main as cli_main
from modules.module_04_gamut_mapping.src.mapper import GamutMapper
from modules.module_04_gamut_mapping.src.metrics import MappingQualityPolicy
from modules.module_04_gamut_mapping.src.sdl_palette import (
    DEFAULT_SDL_PATH,
    SDLPalette,
    points_in_convex_polygon,
)
from modules.module_04_gamut_mapping.tests.sample_palette import (
    write_sample_sdl_palette,
)


class ColorSpaceTests(unittest.TestCase):
    def test_srgb_xyz_round_trip(self):
        srgb = np.array([[0.1, 0.4, 0.9], [1.0, 0.5, 0.0]], dtype=np.float64)
        restored = xyz_to_srgb(srgb_to_xyz(srgb))
        np.testing.assert_allclose(restored, srgb, atol=1e-6)

    def test_rgb_lab_round_trip(self):
        rgb = np.array([[12, 90, 240], [255, 180, 40]], dtype=np.uint8)
        restored = lab_to_rgb8(rgb8_to_lab(rgb))
        np.testing.assert_allclose(restored, rgb, atol=1)


class SDLPaletteTests(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_SDL_PATH.is_file(),
        "authorized local SDL table is not present",
    )
    def test_official_table_is_complete(self):
        palette = SDLPalette.from_file(DEFAULT_SDL_PATH)
        self.assertEqual(len(palette.xy_samples), 1024)
        self.assertEqual(len(palette.rgb), 990)
        self.assertEqual(len(palette.hull_xy), 3)

    def test_malformed_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("(0.2,0.3),(255, 200, 100)\nnot valid\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                SDLPalette.from_file(path)

    def test_table_requires_two_unique_rgb_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-rgb.txt"
            path.write_text(
                "(0.20, 0.20), (255, 200, 100)\n"
                "(0.40, 0.20), (255, 200, 100)\n"
                "(0.30, 0.40), (255, 200, 100)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique RGB"):
                SDLPalette.from_file(path)

    def test_convex_polygon_includes_boundary(self):
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        points = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        np.testing.assert_array_equal(
            points_in_convex_polygon(points, triangle),
            np.array([True, True, False]),
        )


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._palette_directory = tempfile.TemporaryDirectory()
        palette_path = Path(cls._palette_directory.name) / "sample_sdl.txt"
        write_sample_sdl_palette(palette_path)
        cls.palette = SDLPalette.from_file(palette_path)
        cls.mapper = GamutMapper(cls.palette, batch_size=128)
        x = np.linspace(0, 1, 64, dtype=np.float32)
        gradient = np.empty((16, 64, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.rint(255 * x)
        gradient[:, :, 1] = np.rint(80 + 150 * x)
        gradient[:, :, 2] = np.rint(255 * (1 - x))
        cls.gradient = gradient

    @classmethod
    def tearDownClass(cls):
        cls._palette_directory.cleanup()

    def test_nearest_mapping_is_strictly_in_table(self):
        mapped = self.mapper.map_rgb_array(self.gradient, method="nearest")
        self.assertTrue(self.palette.strict_table_mask(mapped).all())

    def test_smooth_mapping_is_deterministic_and_strict(self):
        first = self.mapper.map_rgb_array(
            self.gradient,
            method="smooth",
            smooth_radius=0.4,
        )
        second = self.mapper.map_rgb_array(
            self.gradient,
            method="smooth",
            smooth_radius=0.4,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(self.palette.strict_table_mask(first).all())

    def test_smooth_mapping_reduces_palette_plateaus_on_gradient(self):
        image = Image.fromarray(self.gradient, mode="RGB")
        nearest = self.mapper.map_image(image, method="nearest")
        smooth = self.mapper.map_image(
            image,
            method="smooth",
            smooth_radius=0.4,
        )
        self.assertLess(
            smooth.quality.control_flat_neighbor_fraction,
            nearest.quality.control_flat_neighbor_fraction,
        )

    def test_alpha_channel_is_preserved(self):
        alpha = np.tile(np.arange(64, dtype=np.uint8), (16, 1))
        image = Image.fromarray(np.dstack((self.gradient, alpha)), mode="RGBA")
        result = self.mapper.map_image(image)
        np.testing.assert_array_equal(np.asarray(result.image)[:, :, 3], alpha)
        np.testing.assert_array_equal(np.asarray(result.control_image)[:, :, 3], alpha)
        control_rgb = np.asarray(result.control_image)[:, :, :3]
        self.assertTrue(self.palette.strict_table_mask(control_rgb).all())
        self.assertEqual(result.quality.strict_invalid_pixel_count, 0)

    def test_preview_preserves_source_luminance(self):
        image = Image.fromarray(self.gradient, mode="RGB")
        result = self.mapper.map_image(image)
        self.assertLess(result.quality.mean_absolute_luminance_change, 0.01)

    def test_visual_preview_does_not_quantize_in_gamut_pixels(self):
        source = np.full((12, 20, 3), (255, 255, 0), dtype=np.uint8)
        self.assertTrue(self.palette.chromaticity_is_inside(source).all())
        result = self.mapper.map_image(Image.fromarray(source, mode="RGB"))
        np.testing.assert_array_equal(np.asarray(result.image), source)
        self.assertEqual(result.quality.mean_delta_e76, 0.0)
        self.assertEqual(result.quality.mean_absolute_luminance_change, 0.0)

    def test_out_of_gamut_preview_is_continuously_compressed(self):
        source = np.full((12, 20, 3), (0, 255, 0), dtype=np.uint8)
        self.assertFalse(self.palette.chromaticity_is_inside(source).any())
        result = self.mapper.map_image(Image.fromarray(source, mode="RGB"))
        preview = np.asarray(result.image)
        self.assertFalse(np.array_equal(preview, source))
        self.assertTrue(self.palette.chromaticity_is_inside(preview).all())
        control = np.asarray(result.control_image)
        self.assertTrue(self.palette.strict_table_mask(control).all())

    def test_out_of_gamut_mask_matches_image_size(self):
        image = Image.fromarray(self.gradient, mode="RGB")
        result = self.mapper.map_image(image)
        self.assertEqual(result.out_of_gamut_mask.size, image.size)
        self.assertEqual(result.quality.pixel_count, 16 * 64)

    def test_quality_policy_warns_for_excessive_out_of_gamut_input(self):
        source = np.full((12, 20, 3), (0, 255, 0), dtype=np.uint8)
        result = self.mapper.map_image(Image.fromarray(source, mode="RGB"))
        self.assertFalse(
            any(
                "before_xy_out_of_gamut_fraction" in failure
                for failure in result.quality_failures
            )
        )
        self.assertTrue(
            any(
                "before_xy_out_of_gamut_fraction" in failure
                for failure in result.quality_policy.advisories(result.quality)
            )
        )
        self.assertFalse(
            any("p95_delta_e76" in failure for failure in result.quality_failures)
        )
        self.assertTrue(
            any(
                "p95_delta_e76" in advisory
                for advisory in result.quality_policy.advisories(result.quality)
            )
        )

    def test_quality_policy_accepts_representative_warm_gradient(self):
        x = np.linspace(0, 1, 64, dtype=np.float32)
        source = np.empty((16, 64, 3), dtype=np.uint8)
        source[:, :, 0] = 245
        source[:, :, 1] = np.rint(180 + 35 * x)
        source[:, :, 2] = np.rint(170 - 55 * x)
        result = self.mapper.map_image(Image.fromarray(source, mode="RGB"))
        self.assertTrue(result.accepted, result.quality_failures)


class CliTests(unittest.TestCase):
    def test_cli_writes_preview_control_mask_and_report(self):
        x = np.linspace(0, 1, 64, dtype=np.float32)
        gradient = np.empty((16, 64, 3), dtype=np.uint8)
        gradient[:, :, 0] = 245
        gradient[:, :, 1] = np.rint(180 + 35 * x)
        gradient[:, :, 2] = np.rint(170 - 55 * x)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw.png"
            palette_path = root / "sample_sdl.txt"
            output_dir = root / "mapped"
            Image.fromarray(gradient, mode="RGB").save(input_path)
            write_sample_sdl_palette(palette_path)
            return_code = cli_main(
                [
                    "--input",
                    str(input_path),
                    "--sdl",
                    str(palette_path),
                    "--output-dir",
                    str(output_dir),
                    "--batch-size",
                    "128",
                    "--save-baseline",
                ]
            )
            self.assertEqual(return_code, 0)
            self.assertTrue((output_dir / "raw_sdl_smooth.png").is_file())
            self.assertTrue((output_dir / "raw_sdl_smooth_control.png").is_file())
            self.assertTrue((output_dir / "raw_out_of_gamut_mask.png").is_file())
            report = json.loads(
                (output_dir / "raw_sdl_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["quality"]["strict_invalid_pixel_count"], 0)
            self.assertEqual(report["quality_status"], "accepted")
            self.assertEqual(report["quality_failures"], [])
            self.assertIn("input_sha256", report)
            self.assertIn("sdl_sha256", report)


if __name__ == "__main__":
    unittest.main()
