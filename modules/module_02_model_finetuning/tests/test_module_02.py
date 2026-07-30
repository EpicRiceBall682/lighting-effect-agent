from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from PIL import Image

from modules.module_02_model_finetuning.src.generate_synthetic import (
    color_name,
    generate_synthetic_dataset,
    palette_is_allowed,
)
from modules.module_02_model_finetuning.src.image_palette import audit_rgb_pixels
from modules.module_02_model_finetuning.src.prepare_dataset import (
    _write_split,
    infer_synthetic_grouping_fields,
    load_synthetic_records,
    split_records,
)
from modules.module_02_model_finetuning.src.vision_caption import (
    VisionCaptionError,
    caption_metadata,
    validate_caption,
)


class SyntheticGeneratorTests(unittest.TestCase):
    def test_color_names_are_human_readable(self):
        import numpy as np

        self.assertEqual(color_name(np.array([255, 180, 30], dtype=np.float32)), "golden amber")

    def test_rejects_dark_blue_but_accepts_light_blue(self):
        import numpy as np

        self.assertFalse(palette_is_allowed([np.array([20, 40, 255], dtype=np.float32)]))
        self.assertTrue(palette_is_allowed([np.array([160, 200, 255], dtype=np.float32)]))

    def test_final_pixel_audit_rejects_green_middle_of_blue_yellow_gradient(self):
        import numpy as np

        blue = np.array([40, 150, 230], dtype=np.float32)
        yellow = np.array([250, 245, 30], dtype=np.float32)
        blend = np.linspace(blue, yellow, 256, dtype=np.float32)[None, :, :]
        self.assertFalse(audit_rgb_pixels(blend).allowed)

    def test_final_pixel_audit_accepts_warm_gradient(self):
        import numpy as np

        pink = np.array([255, 180, 200], dtype=np.float32)
        orange = np.array([255, 170, 70], dtype=np.float32)
        blend = np.linspace(pink, orange, 256, dtype=np.float32)[None, :, :]
        self.assertTrue(audit_rgb_pixels(blend).allowed)

    def test_final_pixel_audit_rejects_isolated_chroma_spot(self):
        import numpy as np

        image = np.full((96, 256, 3), (230, 185, 165), dtype=np.uint8)
        image[40:47, 190:197] = (255, 30, 180)
        audit = audit_rgb_pixels(image, max_isolated_chroma_fraction=0.0005)
        self.assertGreater(audit.isolated_chroma_fraction, 0.0005)
        self.assertFalse(audit.allowed)

    def test_generates_all_requested_types(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = generate_synthetic_dataset(
                output,
                linear_count=1,
                fluid_count=1,
                sky_count=1,
                width=64,
                height=64,
                seed=7,
            )
            self.assertTrue(metadata.is_file())
            self.assertEqual(len(metadata.read_text(encoding="utf-8").splitlines()), 3)
            generated_records = [
                json.loads(line)
                for line in metadata.read_text(encoding="utf-8").splitlines()
            ]
            for record in generated_records:
                self.assertTrue(record["recipe_id"])
                self.assertTrue(record["split_group"])
                self.assertTrue(record["palette_family"])
                self.assertTrue(record["layout_id"])
            for path in (output / "images").glob("*.png"):
                with Image.open(path) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertTrue(audit_rgb_pixels(__import__("numpy").asarray(image)).allowed)


class SplitTests(unittest.TestCase):
    def test_legacy_v3_metadata_receives_deterministic_grouping(self):
        linear = infer_synthetic_grouping_fields(
            {
                "source": "synthetic_linear",
                "template_text": (
                    "Wide panoramic lighting with a seamless horizontal gradient "
                    "transitioning from light blue to golden amber, a soft central "
                    "mist, gentle luminosity, and clean continuous color blending."
                ),
            }
        )
        fluid = infer_synthetic_grouping_fields(
            {
                "source": "synthetic_fluid",
                "generator_mode": "hybrid",
                "template_text": (
                    "Wide panoramic hybrid gradient lighting blending light blue, "
                    "soft lavender, with soft diffused glows."
                ),
            }
        )

        self.assertEqual(linear["layout_id"], "horizontal")
        self.assertEqual(
            linear["split_group"],
            "synthetic_linear:linear:light blue->golden amber",
        )
        self.assertEqual(fluid["layout_id"], "hybrid")
        self.assertEqual(fluid["palette_family"], "light blue|soft lavender")

    def test_split_is_deterministic_and_keeps_each_source(self):
        records = [
            {"source": "a", "content_sha256": f"a{i}"} for i in range(10)
        ] + [{"source": "b", "content_sha256": f"b{i}"} for i in range(10)]
        train1, validation1 = split_records(records, 0.1, 42)
        train2, validation2 = split_records(records, 0.1, 42)
        self.assertEqual(train1, train2)
        self.assertEqual(validation1, validation2)
        self.assertEqual({item["source"] for item in validation1}, {"a", "b"})

    def test_recipe_groups_never_cross_train_and_validation(self):
        records = [
            {
                "source": "synthetic",
                "content_sha256": f"hash-{index}",
                "split_group": f"recipe-{index // 2}",
            }
            for index in range(12)
        ]

        train, validation = split_records(records, 0.25, 42)
        train_groups = {item["split_group"] for item in train}
        validation_groups = {item["split_group"] for item in validation}

        self.assertTrue(validation)
        self.assertTrue(train)
        self.assertTrue(train_groups.isdisjoint(validation_groups))

    def test_imagefolder_metadata_has_only_one_file_name_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (8, 8), (255, 200, 120)).save(source)
            records = [
                {
                    "source_path": source,
                    "text": "soft warm gradient",
                    "scene_prompt": "",
                    "source": "synthetic_linear",
                    "caption_source": "vision_model",
                    "caption_model": "fake/vision",
                    "original_file_name": "linear_0001.png",
                    "content_sha256": "abc123",
                    "recipe_id": "linear:warm pink->warm orange",
                    "split_group": "synthetic_linear:linear:warm pink->warm orange",
                    "palette_family": "warm orange|warm pink",
                    "layout_id": "horizontal",
                }
            ]

            _write_split(root / "train", records)
            metadata = json.loads((root / "train" / "metadata.jsonl").read_text(encoding="utf-8"))

            self.assertEqual(metadata["file_name"], "synthetic_linear_0001.png")
            self.assertEqual(metadata["original_name"], "linear_0001.png")
            self.assertEqual(metadata["caption_source"], "vision_model")
            self.assertEqual(metadata["layout_id"], "horizontal")
            self.assertTrue(metadata["split_group"].startswith("synthetic_linear:"))
            self.assertEqual(
                [key for key in metadata if key == "file_name" or key.endswith("_file_name")],
                ["file_name"],
            )

    def test_final_package_rejects_template_only_synthetic_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (8, 8), (255, 200, 120)).save(image_dir / "one.png")
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "file_name": "images/one.png",
                        "text": "A template caption for a warm synthetic gradient image.",
                        "caption_source": "template",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_synthetic_records([metadata])


class VisionCaptionTests(unittest.TestCase):
    def test_rejects_forbidden_color_in_caption(self):
        with self.assertRaises(VisionCaptionError):
            validate_caption(
                "A seamless green and yellow panoramic gradient with a soft luminous center and flowing atmospheric texture."
            )

    def test_caption_metadata_writes_grounded_caption_and_can_resume(self):
        class FakeClient:
            model = "fake/vision"
            provider = "fake_local"

            def __init__(self):
                self.calls = 0

            def caption(self, image_path):
                self.calls += 1
                return "A pale yellow to warm orange horizontal gradient with soft central haze and smooth luminous blending across the panorama."

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (64, 64), (255, 190, 80)).save(image_dir / "one.png")
            source = root / "source.jsonl"
            output = root / "vision.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "file_name": "images/one.png",
                        "text": "template caption",
                        "caption_source": "template",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            client = FakeClient()
            first = caption_metadata(source, output, client, request_interval_seconds=0)
            second = caption_metadata(source, output, client, request_interval_seconds=0)
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(first["newly_captioned"], 1)
            self.assertEqual(second["reused"], 1)
            self.assertEqual(client.calls, 1)
            self.assertEqual(record["caption_source"], "vision_model")
            self.assertEqual(record["caption_provider"], "fake_local")
            self.assertEqual(record["template_text"], "template caption")

    def test_caption_cache_is_reused_by_image_hash_after_filename_changes(self):
        class FailIfCalledClient:
            model = "fake/vision"
            provider = "fake"

            def caption(self, image_path):
                raise AssertionError("identical cached image should not be captioned again")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_images = root / "old" / "images"
            new_images = root / "new" / "images"
            old_images.mkdir(parents=True)
            new_images.mkdir(parents=True)
            image = Image.new("RGB", (16, 16), (255, 180, 100))
            image.save(old_images / "old.png")
            image.save(new_images / "new.png")
            digest = __import__(
                "modules.module_02_model_finetuning.src.vision_caption",
                fromlist=["sha256_file"],
            ).sha256_file(old_images / "old.png")
            cache = root / "old" / "vision.jsonl"
            cache.write_text(
                json.dumps(
                    {
                        "file_name": "images/old.png",
                        "text": "A pale yellow to warm orange gradient with soft luminous haze and smooth horizontal blending across the panoramic texture.",
                        "caption_source": "vision_model",
                        "caption_model": "fake/vision",
                        "caption_provider": "fake",
                        "caption_image_sha256": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source = root / "new" / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "file_name": "images/new.png",
                        "text": "new template",
                        "caption_source": "template",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "new" / "vision.jsonl"
            summary = caption_metadata(
                source,
                output,
                FailIfCalledClient(),
                cache_metadata_paths=(cache,),
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["reused"], 1)
            self.assertEqual(record["file_name"], "images/new.png")
            self.assertEqual(record["template_text"], "new template")


if __name__ == "__main__":
    unittest.main()
