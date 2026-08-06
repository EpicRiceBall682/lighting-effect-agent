"""Shared color vocabulary for scene prompting and deterministic rendering."""

from __future__ import annotations

import re


COLOR_RGB: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("deep navy", (28, 50, 96)),
    ("navy blue", (38, 67, 123)),
    ("navy", (38, 67, 123)),
    ("indigo", (75, 63, 181)),
    ("electric purple", (106, 35, 238)),
    ("vivid purple", (139, 37, 226)),
    ("bright purple", (157, 64, 225)),
    ("light purple", (197, 180, 225)),
    ("soft purple", (202, 183, 226)),
    ("pale lavender", (211, 199, 230)),
    ("lavender", (194, 174, 219)),
    ("purple", (157, 82, 211)),
    ("vivid magenta", (241, 0, 169)),
    ("bright magenta", (235, 24, 174)),
    ("soft magenta", (221, 83, 168)),
    ("magenta", (226, 31, 160)),
    ("vivid pink", (248, 38, 154)),
    ("bright pink", (247, 62, 164)),
    ("pale pink", (242, 190, 207)),
    ("soft pink", (239, 177, 203)),
    ("light pink", (243, 190, 210)),
    ("pink", (242, 112, 174)),
    ("pale golden yellow", (246, 218, 126)),
    ("bright yellow", (255, 226, 70)),
    ("pure yellow", (255, 224, 76)),
    ("warm yellow", (250, 205, 82)),
    ("soft yellow", (249, 224, 142)),
    ("pale yellow", (250, 231, 153)),
    ("yellow", (252, 215, 82)),
    ("vivid green", (54, 205, 104)),
    ("bright green", (74, 214, 112)),
    ("light green", (157, 224, 167)),
    ("pale green", (197, 235, 195)),
    ("emerald green", (42, 183, 122)),
    ("mint green", (157, 232, 201)),
    ("lime green", (170, 224, 92)),
    ("olive green", (153, 166, 91)),
    ("green", (76, 190, 115)),
    ("bright cyan", (61, 213, 224)),
    ("light cyan", (157, 231, 235)),
    ("pale cyan", (196, 239, 239)),
    ("cyan", (78, 207, 218)),
    ("teal", (60, 171, 166)),
    ("turquoise", (72, 210, 196)),
    ("bright blue", (72, 156, 235)),
    ("light blue", (151, 203, 238)),
    ("sky blue", (145, 199, 235)),
    ("soft blue", (165, 208, 235)),
    ("pale blue", (184, 218, 239)),
    ("blue", (80, 145, 220)),
    ("bright orange", (250, 132, 49)),
    ("soft orange", (242, 174, 112)),
    ("warm orange", (238, 154, 91)),
    ("light orange", (245, 183, 121)),
    ("pale orange", (247, 195, 139)),
    ("orange", (245, 148, 75)),
    ("warm peach", (242, 183, 157)),
    ("light peach", (244, 197, 170)),
    ("peach", (244, 184, 151)),
    ("light tan", (205, 172, 137)),
    ("warm beige", (226, 204, 173)),
    ("beige", (222, 205, 180)),
    ("warm brown", (154, 100, 66)),
    ("brown", (125, 78, 45)),
    ("pure red", (255, 0, 0)),
    ("vivid red", (250, 28, 45)),
    ("bright red", (248, 48, 55)),
    ("deep red", (166, 24, 38)),
    ("warm red", (239, 62, 67)),
    ("red", (231, 115, 109)),
    ("coral", (237, 151, 139)),
    ("amber", (237, 178, 84)),
    ("ivory", (250, 241, 211)),
    ("soft white", (248, 245, 235)),
    ("white", (250, 248, 242)),
)

SUPPORTED_COLOR_NAMES = tuple(name for name, _rgb in COLOR_RGB)
COLOR_FAMILY_TERMS = (
    "blue",
    "yellow",
    "orange",
    "pink",
    "magenta",
    "purple",
    "lavender",
    "peach",
    "tan",
    "beige",
    "brown",
    "red",
    "coral",
    "amber",
    "ivory",
    "white",
    "green",
    "cyan",
    "teal",
    "turquoise",
    "mint",
    "lime",
    "olive",
    "emerald",
    "navy",
    "indigo",
)


def matched_color_spans(text: str) -> list[tuple[int, int, str, tuple[int, int, int]]]:
    """Return longest-first, non-overlapping supported color matches."""

    lowered = text.casefold()
    matches: list[tuple[int, int, str, tuple[int, int, int]]] = []
    occupied: list[tuple[int, int]] = []
    for color_name, rgb in sorted(COLOR_RGB, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(rf"\b{re.escape(color_name)}\b", lowered):
            if any(start < match.end() and match.start() < end for start, end in occupied):
                continue
            matches.append((match.start(), match.end(), color_name, rgb))
            occupied.append(match.span())
    return sorted(matches, key=lambda item: item[0])


def unsupported_color_terms(text: str) -> tuple[str, ...]:
    """Find color-family words that are not covered by a supported color name."""

    lowered = text.casefold()
    covered = [(start, end) for start, end, _name, _rgb in matched_color_spans(lowered)]
    unsupported: list[str] = []
    family_pattern = "|".join(re.escape(term) for term in COLOR_FAMILY_TERMS)
    for match in re.finditer(rf"\b(?:[a-z]+\s+)?(?:{family_pattern})\b", lowered):
        family_word = re.search(rf"(?:{family_pattern})\b", match.group(0))
        if family_word is None:
            continue
        family_start = match.start() + family_word.start()
        if any(start <= family_start < end for start, end in covered):
            continue
        term = match.group(0).strip()
        if term not in unsupported:
            unsupported.append(term)
    return tuple(unsupported)
