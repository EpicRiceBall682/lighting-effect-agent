from __future__ import annotations

import json
import os
import re
import unittest
from unittest.mock import patch

from modules.module_01_prompt_agent.src.agent import (
    LightingPromptAgent,
    density_for_space_size,
)
from modules.module_01_prompt_agent.src.client import (
    DeepSeekClient,
    ModelConfigurationError,
    decode_json_content,
)
from modules.module_01_prompt_agent.src.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
)
from modules.module_01_prompt_agent.src.fast_compiler import (
    FastPromptCompiler,
    has_explicit_color_cue,
)
from modules.module_01_prompt_agent.src.schemas import (
    LightingEffectAttributes,
    LightingEffectValidationError,
)
from modules.color_vocabulary import SUPPORTED_COLOR_NAMES


VALID_RESPONSE = {
    "density": "middle",
    "m_intensity": 70,
    "k_intensity": 90,
    "a_intensity": 60,
    "effect": (
        "Wide panoramic abstract light texture with pale yellow across the upper area, "
        "soft pink spreading toward the lower right, a broad diagonal gradient, gentle "
        "misty glow, diffused bloom, evenly bright illumination, and a relaxing atmosphere."
    ),
}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class SchemaTests(unittest.TestCase):
    def test_accepts_numeric_strings_from_original_demo_shape(self):
        raw = dict(VALID_RESPONSE, m_intensity="70")
        parsed = LightingEffectAttributes.from_mapping(raw)
        self.assertEqual(parsed.m_intensity, 70)

    def test_allows_dark_hue_modifiers(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with a dark pink gradient across "
                "the upper area, gentle misty glow, diffused bloom, evenly bright "
                "illumination, and a romantic welcoming atmosphere throughout the panel."
            ),
        )
        self.assertIn("dark pink", LightingEffectAttributes.from_mapping(raw).effect)

    def test_allows_green_cyan_and_teal(self):
        for color in ("bright green", "bright cyan", "teal"):
            raw = dict(
                VALID_RESPONSE,
                effect=(
                    f"Wide panoramic abstract light texture with a bright {color} gradient "
                    "across the upper area, a luminous center, diffused bloom, evenly bright "
                    "illumination, and a clean translucent welcoming atmosphere throughout "
                    "the continuous panel."
                ),
            )
            with self.subTest(color=color):
                self.assertIn(color, LightingEffectAttributes.from_mapping(raw).effect)

    def test_rejects_unexpected_json_fields(self):
        raw = dict(VALID_RESPONSE, explanation="extra model commentary")
        with self.assertRaises(LightingEffectValidationError):
            LightingEffectAttributes.from_mapping(raw)

    def test_rejects_mixed_chinese_effect(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with pale yellow across the upper "
                "area and soft pink below，营造温暖放松的氛围。"
            ),
        )
        with self.assertRaises(LightingEffectValidationError):
            LightingEffectAttributes.from_mapping(raw)

    def test_allows_light_and_unspecified_blue(self):
        allowed = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with pale yellow across the lower "
                "area, light blue above, a broad vertical gradient, cloud-like diffusion, "
                "soft luminous blending, evenly bright illumination, and an airy atmosphere."
            ),
        )
        rejected = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with pale yellow across the lower "
                "area, blue above, a broad vertical gradient, cloud-like diffusion, soft "
                "luminous blending, evenly bright illumination, and an airy atmosphere."
            ),
        )
        self.assertIn("light blue", LightingEffectAttributes.from_mapping(allowed).effect)
        self.assertIn("blue", LightingEffectAttributes.from_mapping(rejected).effect)

    def test_bright_blue_is_supported_by_the_shared_renderer_vocabulary(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic organizer-style color field with pale yellow on the left "
                "and dominant bright blue across the center and right, forming a clean "
                "smooth horizontal gradient with uniform vertical color, calm illumination, "
                "clear identity, and an uninterrupted surface throughout."
            ),
        )
        parsed = LightingEffectAttributes.from_mapping(raw)
        self.assertIn("bright blue", parsed.effect)

    def test_rejects_artifact_prone_small_shapes(self):
        for term in ("spots", "clusters", "accents", "beams"):
            raw = dict(
                VALID_RESPONSE,
                effect=(
                    "Wide panoramic abstract light texture with pale yellow across the "
                    f"upper area, several luminous {term} near the right side, a broad "
                    "horizontal gradient, misty glow, diffused bloom, evenly bright "
                    "illumination, and an uplifting atmosphere."
                ),
            )
            with self.subTest(term=term), self.assertRaisesRegex(
                LightingEffectValidationError, "artifact-prone"
            ):
                LightingEffectAttributes.from_mapping(raw)

    def test_bright_does_not_count_as_right_spatial_placement(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "Bright abstract luminous texture with pale yellow warmth and soft orange "
                "tones creating gentle diffused bloom, soothing radiance, smooth softness, "
                "calm atmosphere, elegant illumination, balanced warmth, translucent color, "
                "and a welcoming comfortable mood throughout."
            ),
        )
        with self.assertRaisesRegex(
            LightingEffectValidationError, "spatial color placement"
        ):
            LightingEffectAttributes.from_mapping(raw)

    def test_requires_at_least_twenty_english_words(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "Pale yellow across the upper area with a horizontal gradient, "
                "misty glow, soft orange warmth, and calm illumination."
            ),
        )
        with self.assertRaisesRegex(LightingEffectValidationError, "20 to 60"):
            LightingEffectAttributes.from_mapping(raw)

    def test_rejects_direction_that_conflicts_with_color_placement(self):
        vertical_layout = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with light blue across the upper "
                "area and pale pink across the lower area, a broad horizontal gradient, "
                "misty glow, diffused bloom, evenly bright illumination, clean softness, "
                "and a calm futuristic atmosphere throughout."
            ),
        )
        horizontal_layout = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic abstract light texture with pale yellow along the left "
                "side and warm peach along the right side, a broad vertical gradient, "
                "misty glow, diffused bloom, evenly bright illumination, clean softness, "
                "and a calm welcoming atmosphere throughout."
            ),
        )
        for raw in (vertical_layout, horizontal_layout):
            with self.subTest(effect=raw["effect"]), self.assertRaisesRegex(
                LightingEffectValidationError, "requires"
            ):
                LightingEffectAttributes.from_mapping(raw)


class PromptTests(unittest.TestCase):
    def test_includes_fixture_aspect_ratio(self):
        prompt = build_user_prompt(
            "酒店套房客厅，浪漫氛围",
            hardware_width_mm=1220,
            hardware_height_mm=370,
        )
        self.assertIn("1220 mm × 370 mm", prompt)
        self.assertIn("3.297:1", prompt)

    def test_system_prompt_requires_organizer_aligned_horizontal_caption(self):
        prompt = build_system_prompt()
        self.assertIn("30 to 50 English words", prompt)
        self.assertIn("dominant color", prompt)
        self.assertIn("left-to-right horizontal gradient", prompt)
        self.assertIn("one to four named colors", prompt)
        self.assertIn("vertical axis uniform", prompt)
        self.assertIn("Do not apply fixed mappings", prompt)
        self.assertNotIn("Energetic disco", prompt)
        self.assertNotIn("Cozy coffee shop", prompt)
        self.assertIn("Do not introduce mist", prompt)
        self.assertTrue(all(name in prompt for name in SUPPORTED_COLOR_NAMES))
        self.assertIn("concept_prompt", prompt)

    def test_fast_compiler_builds_a_natural_unwashed_concept_prompt(self):
        result = FastPromptCompiler().generate(
            "清晨花海，粉色、黄色、绿色和蓝色，清新自然"
        )
        self.assertIn("bright green", result.effect)
        self.assertIn("flowers", result.concept_prompt)
        self.assertIn("physically plausible", result.concept_prompt)
        self.assertIn("without a uniform color wash", result.concept_prompt)

    def test_blue_sky_and_clouds_compile_to_an_outdoor_sky_scene(self):
        result = FastPromptCompiler().generate("蓝天白云")

        self.assertIn("vast blue sky", result.concept_prompt)
        self.assertIn("sky fills the entire frame", result.concept_prompt)
        self.assertIn("no ground, water", result.concept_prompt)
        self.assertNotIn("matching the described activity", result.concept_prompt)
        self.assertIn("bright blue", result.effect)
        self.assertIn("soft white", result.effect)

    def test_fast_compiler_recognizes_representative_scene_categories(self):
        cases = {
            "雪山日照": "mountain landscape",
            "安静的湖边": "waterside landscape",
            "城市街道夜景": "urban outdoor setting",
            "机场候机大厅": "transport hub",
            "医院候诊区": "healthcare environment",
            "博物馆展览": "museum or gallery",
            "学校图书馆": "learning space",
            "工厂生产线": "industrial workspace",
        }

        for scene, expected in cases.items():
            with self.subTest(scene=scene):
                result = FastPromptCompiler().generate(scene)
                self.assertIn(expected, result.concept_prompt)

    def test_color_detection_distinguishes_explicit_color_from_style(self):
        self.assertTrue(has_explicit_color_cue("蓝色赛博朋克"))
        self.assertTrue(has_explicit_color_cue("blue cyberpunk"))
        self.assertFalse(has_explicit_color_cue("赛博朋克"))
        self.assertFalse(has_explicit_color_cue("未来主义城市"))
        self.assertFalse(has_explicit_color_cue("地中海黄昏样式光"))

    def test_local_fallback_does_not_add_a_companion_to_one_explicit_color(self):
        result = FastPromptCompiler().generate("纯红色主导的宽幅光效")

        self.assertIn("dominant warm red across the entire panel", result.effect)
        self.assertNotIn("orange", result.effect)
        self.assertNotIn("pink", result.effect)

    def test_local_cyberpunk_fallback_is_cool_neon_not_warm(self):
        result = FastPromptCompiler().generate("赛博朋克")

        self.assertIn("electric purple", result.effect)
        self.assertIn("vivid magenta", result.effect)
        self.assertIn("bright cyan", result.effect)
        self.assertIn("futuristic city", result.concept_prompt)
        self.assertNotIn("warm yellow", result.effect)

    def test_local_fallback_uses_semantic_palettes_for_ambiguous_effects(self):
        compiler = FastPromptCompiler()
        sunset = compiler.generate("酒店特色餐厅，地中海黄昏样式光")
        neon = compiler.generate("服装零售陈列区，霓虹样式光")
        water = compiler.generate("酒店泳池，波光、深度、宁静")

        self.assertIn("warm orange", sunset.effect)
        self.assertIn("vivid purple", sunset.effect)
        self.assertIn("electric purple", neon.effect)
        self.assertIn("bright cyan", neon.effect)
        self.assertIn("deep navy", water.effect)
        self.assertIn("light blue", water.effect)

    def test_local_generic_fallback_varies_by_scene_deterministically(self):
        compiler = FastPromptCompiler()
        scenes = (
            "酒店大堂主入口，用于营造归属感",
            "酒店大堂休息区，用于营造社交舒适",
            "酒店前台，用于营造品牌第一印象",
        )
        first = [compiler.generate(scene).effect for scene in scenes]
        second = [compiler.generate(scene).effect for scene in scenes]

        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_rejects_caption_without_spatial_structure(self):
        raw = dict(
            VALID_RESPONSE,
            effect=(
                "A warm pale yellow and soft pink gradient with a gentle misty glow, "
                "diffused bloom, clean translucent illumination, peaceful energy, and an "
                "inviting atmosphere suitable for relaxed social interaction, quiet comfort, "
                "soft radiance, welcoming warmth, and elegant visual continuity throughout."
            ),
        )
        with self.assertRaisesRegex(
            LightingEffectValidationError, "spatial color placement"
        ):
            LightingEffectAttributes.from_mapping(raw)


class ClientParsingTests(unittest.TestCase):
    def test_decodes_accidental_json_fence(self):
        decoded = decode_json_content('```json\n{"density":"low"}\n```')
        self.assertEqual(decoded["density"], "low")


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class DeepSeekClientTests(unittest.TestCase):
    def test_requires_deepseek_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelConfigurationError):
                DeepSeekClient()

    def test_sends_v4_flash_json_request_with_thinking_disabled(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHTTPResponse(
                {"choices": [{"message": {"content": json.dumps(VALID_RESPONSE)}}]}
            )

        client = DeepSeekClient(token="test-secret", max_retries=0)
        with patch(
            "modules.module_01_prompt_agent.src.client.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client.complete_json([{"role": "user", "content": "Return JSON."}])

        request = captured["request"]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(result, VALID_RESPONSE)

    def test_retries_when_deepseek_returns_empty_content(self):
        responses = [
            FakeHTTPResponse({"choices": [{"message": {"content": ""}}]}),
            FakeHTTPResponse(
                {"choices": [{"message": {"content": json.dumps(VALID_RESPONSE)}}]}
            ),
        ]
        client = DeepSeekClient(token="test-secret", max_retries=1)
        with (
            patch(
                "modules.module_01_prompt_agent.src.client.urlopen",
                side_effect=responses,
            ) as mocked_urlopen,
            patch("modules.module_01_prompt_agent.src.client.time.sleep"),
        ):
            result = client.complete_json([{"role": "user", "content": "Return JSON."}])

        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual(result, VALID_RESPONSE)


class AgentTests(unittest.TestCase):
    def test_returns_validated_attributes(self):
        agent = LightingPromptAgent(FakeClient([VALID_RESPONSE]))
        result = agent.generate("咖啡厅，温暖放松")
        self.assertEqual(result.density, "middle")

    def test_retries_once_after_validation_failure(self):
        invalid = dict(VALID_RESPONSE, m_intensity=130)
        client = FakeClient([invalid, VALID_RESPONSE])
        result = LightingPromptAgent(client, validation_retries=1).generate("酒店大堂，晨雾天光")
        self.assertEqual(result.m_intensity, 70)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("failed validation", client.calls[1][-1]["content"])

    def test_repairs_near_boundary_effect_length_without_api_retry(self):
        too_short = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic field with red left and blue right, smooth horizontal "
                "gradient, uniform vertical color, uninterrupted surface."
            ),
        )
        client = FakeClient([too_short])
        result = LightingPromptAgent(client).generate("酒店大堂，晨雾天光")

        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", result.effect)
        self.assertEqual(len(words), 20)
        self.assertEqual(len(client.calls), 1)

    def test_density_is_deterministic_for_arbitrary_space_size(self):
        self.assertEqual(density_for_space_size(2), "lowest")
        self.assertEqual(density_for_space_size(20), "low")
        self.assertEqual(density_for_space_size(35), "middle")
        self.assertEqual(density_for_space_size(80), "high")

    def test_retries_when_density_does_not_match_space_size(self):
        corrected = dict(VALID_RESPONSE, density="high")
        client = FakeClient([VALID_RESPONSE, corrected])
        result = LightingPromptAgent(client, validation_retries=1).generate(
            "大型酒店大堂，明亮欢迎氛围",
            space_size_m2=75,
        )
        self.assertEqual(result.density, "high")
        self.assertIn("density must be high", client.calls[1][-1]["content"])

    def test_retries_when_effect_duplicates_an_existing_batch_prompt(self):
        corrected = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic organizer-style color field with light peach on the left "
                "and dominant soft orange across the center and right, forming a clean "
                "smooth horizontal gradient with uniform vertical color, warm illumination, "
                "quiet hospitality, and an uninterrupted surface throughout."
            ),
        )
        client = FakeClient([VALID_RESPONSE, corrected])
        result = LightingPromptAgent(client, validation_retries=1).generate(
            "酒店套房客厅，休闲氛围",
            forbidden_effects=[VALID_RESPONSE["effect"]],
        )

        self.assertEqual(result.effect, corrected["effect"])
        self.assertEqual(len(client.calls), 2)
        self.assertIn("duplicates a prompt", client.calls[1][-1]["content"])

    def test_same_ordered_color_design_is_allowed_when_the_model_still_prefers_it(self):
        rephrased_same_design = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic organizer-style color field with pale yellow on the left "
                "and dominant soft pink across the center and right, forming a clean "
                "smooth horizontal gradient with uniform vertical color, calm illumination, "
                "gentle identity, and an uninterrupted surface throughout."
            ),
        )
        client = FakeClient([rephrased_same_design])
        result = LightingPromptAgent(client, validation_retries=1).generate(
            "酒店套房客厅，休闲氛围",
            forbidden_design_effects=[VALID_RESPONSE["effect"]],
        )

        self.assertEqual(result.effect, rephrased_same_design["effect"])
        self.assertEqual(len(client.calls), 1)

    def test_agent_does_not_override_the_models_scene_palette(self):
        model_choice = dict(
            VALID_RESPONSE,
            effect=(
                "Wide panoramic organizer-style color field with pale yellow on the left "
                "and dominant warm orange across the center and right, forming a clean "
                "smooth horizontal gradient with uniform vertical color, clear illumination, "
                "open atmosphere, and an uninterrupted surface throughout."
            ),
        )
        client = FakeClient([model_choice])
        result = LightingPromptAgent(client, validation_retries=1).generate(
            "湛蓝的天空飘着洁白的云朵"
        )
        self.assertEqual(result.effect, model_choice["effect"])
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
