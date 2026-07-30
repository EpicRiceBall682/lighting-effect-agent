import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


STYLES = {
    "blue": {
        "gradient": [
            (0.00, (42, 190, 230)),
            (0.48, (98, 224, 242)),
            (1.00, (224, 252, 255)),
        ],
        "glows": [
            ((0.58, 0.52), (0.72, 0.58), (225, 255, 255), 0.42),
            ((0.12, 0.90), (0.40, 0.20), (255, 255, 255), 0.48),
        ],
        "cloud_color": (252, 255, 255),
        "shadow_color": (86, 205, 229),
        "clouds": [
            (0.12, 0.80, 0.26, 0.13, 230),
            (0.32, 0.88, 0.30, 0.12, 210),
            (0.74, 0.55, 0.26, 0.10, 120),
            (0.92, 0.90, 0.18, 0.08, 160),
        ],
        "shadow_clouds": [
            (0.17, 0.69, 0.23, 0.10, 86),
            (0.34, 0.74, 0.25, 0.08, 68),
        ],
    },
    "sunset": {
        "gradient": [
            (0.00, (65, 154, 229)),
            (0.28, (143, 164, 231)),
            (0.55, (230, 151, 197)),
            (1.00, (255, 206, 184)),
        ],
        "glows": [
            ((0.52, 0.62), (0.82, 0.52), (255, 222, 205), 0.36),
            ((0.16, 0.46), (0.46, 0.26), (255, 153, 187), 0.32),
        ],
        "cloud_color": (255, 171, 192),
        "shadow_color": (132, 151, 224),
        "highlight_color": (255, 228, 204),
        "clouds": [
            (0.16, 0.43, 0.34, 0.14, 180),
            (0.43, 0.52, 0.38, 0.15, 160),
            (0.77, 0.66, 0.36, 0.13, 150),
            (0.24, 0.87, 0.34, 0.10, 125),
            (0.60, 0.86, 0.36, 0.10, 120),
        ],
        "shadow_clouds": [
            (0.18, 0.24, 0.32, 0.10, 105),
            (0.58, 0.20, 0.30, 0.09, 90),
        ],
        "highlight_clouds": [
            (0.35, 0.76, 0.28, 0.06, 125),
            (0.72, 0.72, 0.25, 0.06, 95),
        ],
    },
    "dawn": {
        "gradient": [
            (0.00, (113, 188, 230)),
            (0.36, (190, 206, 229)),
            (0.68, (255, 206, 175)),
            (1.00, (255, 238, 208)),
        ],
        "glows": [
            ((0.52, 0.72), (0.70, 0.38), (255, 238, 204), 0.44),
            ((0.88, 0.20), (0.40, 0.28), (179, 218, 245), 0.22),
        ],
        "cloud_color": (255, 240, 216),
        "shadow_color": (159, 193, 224),
        "highlight_color": (255, 252, 236),
        "clouds": [
            (0.14, 0.70, 0.30, 0.12, 160),
            (0.48, 0.76, 0.34, 0.11, 150),
            (0.84, 0.82, 0.25, 0.09, 120),
            (0.25, 0.93, 0.36, 0.11, 145),
        ],
        "shadow_clouds": [
            (0.30, 0.54, 0.24, 0.08, 78),
            (0.64, 0.58, 0.28, 0.08, 70),
        ],
        "highlight_clouds": [
            (0.18, 0.84, 0.22, 0.06, 125),
            (0.58, 0.82, 0.28, 0.06, 100),
        ],
    },
    "lavender": {
        "gradient": [
            (0.00, (109, 154, 229)),
            (0.38, (162, 170, 233)),
            (0.72, (217, 190, 238)),
            (1.00, (239, 230, 255)),
        ],
        "glows": [
            ((0.50, 0.50), (0.72, 0.50), (231, 220, 255), 0.40),
            ((0.15, 0.82), (0.46, 0.22), (255, 248, 255), 0.36),
        ],
        "cloud_color": (246, 240, 255),
        "shadow_color": (135, 154, 226),
        "highlight_color": (255, 255, 255),
        "clouds": [
            (0.16, 0.78, 0.32, 0.12, 190),
            (0.44, 0.86, 0.35, 0.11, 165),
            (0.78, 0.62, 0.30, 0.10, 125),
            (0.84, 0.92, 0.25, 0.08, 130),
        ],
        "shadow_clouds": [
            (0.24, 0.46, 0.30, 0.09, 75),
            (0.66, 0.42, 0.26, 0.08, 65),
        ],
        "highlight_clouds": [
            (0.26, 0.84, 0.24, 0.06, 110),
            (0.52, 0.74, 0.24, 0.05, 90),
        ],
    },
    "mint": {
        "gradient": [
            (0.00, (76, 206, 218)),
            (0.45, (151, 235, 224)),
            (1.00, (232, 255, 244)),
        ],
        "glows": [
            ((0.54, 0.55), (0.76, 0.52), (226, 255, 244), 0.46),
            ((0.12, 0.84), (0.38, 0.22), (255, 255, 255), 0.34),
        ],
        "cloud_color": (247, 255, 248),
        "shadow_color": (92, 210, 208),
        "clouds": [
            (0.14, 0.78, 0.28, 0.12, 195),
            (0.40, 0.88, 0.32, 0.11, 170),
            (0.78, 0.60, 0.28, 0.10, 118),
            (0.86, 0.90, 0.22, 0.08, 130),
        ],
        "shadow_clouds": [
            (0.18, 0.64, 0.24, 0.08, 70),
            (0.48, 0.68, 0.24, 0.07, 58),
        ],
    },
    "golden": {
        "gradient": [
            (0.00, (94, 177, 232)),
            (0.42, (180, 221, 235)),
            (0.74, (255, 224, 162)),
            (1.00, (255, 244, 204)),
        ],
        "glows": [
            ((0.50, 0.72), (0.82, 0.38), (255, 230, 155), 0.48),
            ((0.20, 0.28), (0.45, 0.30), (170, 224, 244), 0.24),
        ],
        "cloud_color": (255, 247, 218),
        "shadow_color": (135, 196, 218),
        "highlight_color": (255, 255, 238),
        "clouds": [
            (0.15, 0.76, 0.30, 0.12, 185),
            (0.46, 0.84, 0.36, 0.12, 170),
            (0.80, 0.72, 0.30, 0.10, 145),
            (0.28, 0.96, 0.35, 0.10, 130),
        ],
        "shadow_clouds": [
            (0.25, 0.56, 0.28, 0.08, 68),
            (0.64, 0.58, 0.25, 0.07, 58),
        ],
        "highlight_clouds": [
            (0.38, 0.78, 0.24, 0.05, 105),
            (0.76, 0.80, 0.20, 0.05, 95),
        ],
    },
    "deep": {
        "gradient": [
            (0.00, (31, 92, 162)),
            (0.42, (69, 132, 202)),
            (0.74, (136, 190, 229)),
            (1.00, (213, 245, 255)),
        ],
        "glows": [
            ((0.54, 0.58), (0.72, 0.46), (190, 232, 255), 0.36),
            ((0.12, 0.88), (0.35, 0.20), (255, 255, 255), 0.38),
        ],
        "cloud_color": (235, 248, 255),
        "shadow_color": (57, 133, 198),
        "clouds": [
            (0.12, 0.80, 0.28, 0.12, 190),
            (0.36, 0.88, 0.32, 0.11, 165),
            (0.72, 0.64, 0.28, 0.10, 110),
            (0.92, 0.90, 0.18, 0.08, 120),
        ],
        "shadow_clouds": [
            (0.22, 0.58, 0.30, 0.09, 72),
            (0.58, 0.52, 0.25, 0.08, 60),
        ],
    },
}


def clamp(x, lo=0, hi=255):
    return max(lo, min(hi, int(x)))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(a, b, t):
    return tuple(clamp(lerp(a[i], b[i], t)) for i in range(3))


LAYOUTS = [
    "vertical",
    "horizontal",
    "diagonal",
    "anti_diagonal",
    "corner",
    "radial",
    "patchy",
]


def resolve_layout(layout, seed):
    if layout == "auto":
        return random.Random(seed + 101).choice(LAYOUTS)
    return layout


def warped_stops(stops, rng):
    colors = [color for _, color in stops]
    if len(colors) <= 2:
        return stops

    points = [0.0]
    remaining = 1.0
    for _ in range(len(colors) - 2):
        step = rng.uniform(0.14, 0.46) * remaining
        points.append(min(0.92, points[-1] + step))
        remaining = 1.0 - points[-1]
    points.append(1.0)

    return list(zip(points, colors))


def make_layout_params(layout, seed):
    rng = random.Random(seed)
    params = {}
    if layout == "corner":
        params["corner"] = rng.choice([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    elif layout == "radial":
        params["center"] = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75))
        params["scale"] = rng.uniform(0.62, 0.92)
    elif layout in {"diagonal", "anti_diagonal"}:
        params["mix"] = rng.uniform(0.35, 0.70)
    return params


def gradient_value(x, y, w, h, layout, params):
    u = x / max(1, w - 1)
    v = y / max(1, h - 1)

    if layout == "vertical":
        t = v
    elif layout == "horizontal":
        t = u
    elif layout == "diagonal":
        mix = params.get("mix", 0.58)
        t = (u * mix + v * (1.0 - mix))
    elif layout == "anti_diagonal":
        mix = params.get("mix", 0.58)
        t = ((1.0 - u) * mix + v * (1.0 - mix))
    elif layout == "corner":
        corner = params.get("corner", (0.0, 0.0))
        t = ((u - corner[0]) ** 2 + (v - corner[1]) ** 2) ** 0.5 / 1.414
    elif layout == "radial":
        cx, cy = params.get("center", (0.5, 0.5))
        t = ((u - cx) ** 2 + (v - cy) ** 2) ** 0.5 / params.get("scale", 0.78)
    else:
        t = v

    return max(0.0, min(1.0, t))


def color_from_stops(stops, t):
    left = stops[0]
    right = stops[-1]
    for i in range(len(stops) - 1):
        if stops[i][0] <= t <= stops[i + 1][0]:
            left = stops[i]
            right = stops[i + 1]
            break
    span = max(1e-6, right[0] - left[0])
    return lerp_color(left[1], right[1], (t - left[0]) / span)


def make_gradient(size, stops, layout, seed):
    w, h = size
    rng = random.Random(seed)
    stops = warped_stops(sorted(stops, key=lambda x: x[0]), rng)

    if layout == "patchy":
        return make_patchy_gradient(size, stops, seed)

    params = make_layout_params(layout, seed + 17)
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    if layout == "vertical":
        t = vv
    elif layout == "horizontal":
        t = uu
    elif layout == "diagonal":
        mix = params.get("mix", 0.58)
        t = uu * mix + vv * (1.0 - mix)
    elif layout == "anti_diagonal":
        mix = params.get("mix", 0.58)
        t = (1.0 - uu) * mix + vv * (1.0 - mix)
    elif layout == "corner":
        cx, cy = params.get("corner", (0.0, 0.0))
        t = np.sqrt((uu - cx) ** 2 + (vv - cy) ** 2) / 1.414
    elif layout == "radial":
        cx, cy = params.get("center", (0.5, 0.5))
        t = np.sqrt((uu - cx) ** 2 + (vv - cy) ** 2) / params.get("scale", 0.78)
    else:
        t = vv

    t = np.clip(t, 0.0, 1.0)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        left_pos, left_color = stops[i]
        right_pos, right_color = stops[i + 1]
        if i == len(stops) - 2:
            mask = (left_pos <= t) & (t <= right_pos)
        else:
            mask = (left_pos <= t) & (t < right_pos)
        local = np.clip((t - left_pos) / max(1e-6, right_pos - left_pos), 0.0, 1.0)
        left = np.array(left_color, dtype=np.float32)
        right = np.array(right_color, dtype=np.float32)
        color = left + (right - left) * local[..., None]
        arr[mask] = color[mask]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def make_patchy_gradient(size, stops, seed):
    rng = random.Random(seed)
    w, h = size
    base_color = stops[rng.randrange(len(stops))][1]
    img = Image.new("RGB", size, base_color)

    blob_count = rng.randint(4, 8)
    for _ in range(blob_count):
        _, color = rng.choice(stops)
        cx = rng.randint(-w // 8, w + w // 8)
        cy = rng.randint(-h // 8, h + h // 8)
        rx = rng.randint(w // 5, w // 2)
        ry = rng.randint(h // 6, h // 2)
        strength = rng.uniform(0.34, 0.72)
        mask = elliptical_mask(size, (cx, cy), (rx, ry), strength, power=rng.uniform(1.1, 2.1))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=rng.randint(w // 55, w // 24)))
        img = composite_color(img, color, mask)

    return img


def elliptical_mask(size, center, radius, strength, power=1.75):
    w, h = size
    cx, cy = center
    rx, ry = radius
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dx = (xx - cx) / max(1, rx)
    dy = (yy - cy) / max(1, ry)
    t = np.clip(1.0 - np.sqrt(dx * dx + dy * dy), 0.0, 1.0)
    arr = np.clip((t ** power) * 255 * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def composite_color(base, color, mask):
    return Image.composite(Image.new("RGB", base.size, color), base, mask)


def draw_cloud_mask(size, clouds, blur):
    w, h = size
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for cx, cy, rx, ry, alpha in clouds:
        d.ellipse(
            (
                int((cx - rx) * w),
                int((cy - ry) * h),
                int((cx + rx) * w),
                int((cy + ry) * h),
            ),
            fill=int(alpha),
        )
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, int(blur * w))))


def random_clouds(style, seed, count):
    rng = random.Random(seed)
    clouds = []
    lower_bias = 0.70 if style not in {"sunset", "lavender"} else 0.62
    for _ in range(count):
        clouds.append(
            (
                rng.uniform(0.02, 0.98),
                min(1.05, max(0.12, rng.gauss(lower_bias, 0.22))),
                rng.uniform(0.06, 0.20),
                rng.uniform(0.025, 0.085),
                rng.randint(20, 62),
            )
        )
    return clouds


def add_soft_texture(img, seed, opacity=16):
    w, h = img.size
    noise = Image.effect_noise((max(32, w // 6), max(32, h // 6)), 72)
    noise = ImageOps.autocontrast(noise)
    noise = noise.resize(img.size, Image.Resampling.BICUBIC)
    noise = noise.filter(ImageFilter.GaussianBlur(radius=max(8, w // 120)))
    white = Image.new("RGB", img.size, (255, 255, 255))
    return Image.composite(white, img, noise.point(lambda p: clamp((p - 128) * opacity / 128)))


def resolve_style(style, seed):
    names = sorted(STYLES)
    if style == "random":
        return random.Random(seed).choice(names)
    return style


def generate_style(size, seed, style, layout):
    style = resolve_style(style, seed)
    layout = resolve_layout(layout, seed)
    cfg = STYLES[style]
    img = make_gradient(size, cfg["gradient"], layout, seed + 3)
    w, h = size

    for center, radius, color, strength in cfg.get("glows", []):
        mask = elliptical_mask(
            size,
            (int(center[0] * w), int(center[1] * h)),
            (int(radius[0] * w), int(radius[1] * h)),
            strength,
        )
        img = composite_color(img, color, mask)

    shadow_clouds = cfg.get("shadow_clouds", []) + random_clouds(style, seed + 7, 4)
    shadow_mask = draw_cloud_mask(size, shadow_clouds, blur=0.034)
    img = composite_color(img, cfg["shadow_color"], shadow_mask)

    cloud_mask = draw_cloud_mask(size, cfg["clouds"] + random_clouds(style, seed + 13, 7), blur=0.030)
    img = composite_color(img, cfg["cloud_color"], cloud_mask)

    if "highlight_color" in cfg:
        highlight_mask = draw_cloud_mask(size, cfg.get("highlight_clouds", []) + random_clouds(style, seed + 19, 3), blur=0.018)
        img = composite_color(img, cfg["highlight_color"], highlight_mask)

    img = add_soft_texture(img, seed + 31, opacity=10)
    return img.filter(ImageFilter.GaussianBlur(radius=0.45))


def generate_sky_light_texture(output_path, width=1456, height=1088, seed=11, style="blue", layout="auto"):
    if style not in set(STYLES) | {"random"}:
        raise ValueError(f"style must be one of: {', '.join(sorted(STYLES))}, random")
    if layout not in set(LAYOUTS) | {"auto"}:
        raise ValueError(f"layout must be one of: {', '.join(LAYOUTS)}, auto")

    img = generate_style((width, height), seed, style, layout)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/mnt/a/lighting/sky_light_texture.png")
    parser.add_argument("--width", type=int, default=1456)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--style",
        choices=sorted(STYLES) + ["random"],
        default="blue",
        help="Sky texture preset. Use random for a seed-dependent preset.",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS + ["auto"],
        default="auto",
        help="Color distribution layout. auto chooses one based on seed.",
    )
    args = parser.parse_args()

    output = generate_sky_light_texture(
        args.output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        style=args.style,
        layout=args.layout,
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
