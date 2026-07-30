import colorsys
from pathlib import Path

import numpy as np
from PIL import Image


def hsv_to_rgb255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0))
    return np.array([r * 255, g * 255, b * 255], dtype=np.float32)


def mix_color(c1, c2, t):
    return c1 * (1 - t) + c2 * t


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_atmosphere_tint(base_h, rng):
    # 整张图共享的“空气感”色调，低饱和高明度
    sat = rng.uniform(0.10, 0.22)
    val = rng.uniform(0.92, 0.99)
    return hsv_to_rgb255(base_h, sat, val)


def random_palette(rng, n_colors=4):
    """
    生成更和谐的颜色组：
    1. 类比色 analogous
    2. 分裂互补 split complementary
    3. 柔和三角配色 soft triadic
    并统一向 atmosphere tint 靠拢，减少冲突感
    """
    base_h = rng.random()
    scheme = rng.choice(["analogous", "split_complementary", "soft_triad"], p=[0.50, 0.30, 0.20])

    if scheme == "analogous":
        offsets = rng.choice(
            [
                [-0.10, -0.04, 0.03, 0.09],
                [-0.12, -0.05, 0.02, 0.08],
                [-0.08, -0.02, 0.04, 0.10],
            ]
        )
    elif scheme == "split_complementary":
        offsets = rng.choice(
            [
                [-0.06, 0.00, 0.42, 0.50],
                [-0.04, 0.03, 0.45, 0.53],
                [-0.08, 0.02, 0.40, 0.48],
            ]
        )
    else:
        offsets = rng.choice(
            [
                [-0.03, 0.00, 0.28, 0.58],
                [-0.02, 0.04, 0.31, 0.61],
                [-0.05, 0.01, 0.30, 0.56],
            ]
        )

    offsets = list(offsets)[:n_colors]
    atmosphere = build_atmosphere_tint(base_h, rng)

    colors = []
    for off in offsets:
        h = (base_h + off + rng.uniform(-0.015, 0.015)) % 1.0

        # 控制饱和度和亮度范围，不让颜色太炸
        s = rng.uniform(0.40, 0.68)
        v = rng.uniform(0.82, 0.97)

        color = hsv_to_rgb255(h, s, v)

        # 向统一氛围色混入一点，提升整体一致性
        unify = rng.uniform(0.18, 0.35)
        color = mix_color(color, atmosphere, unify)

        colors.append(color)

    return colors, atmosphere


def get_xy_grid(width, height):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def normalize_weights(weights, eps=1e-8):
    total = np.sum(weights, axis=0, keepdims=True)
    return weights / (total + eps)


def apply_global_tone(img, rng, atmosphere):
    height, width, _ = img.shape

    # 轻微整体提亮，并向 atmosphere 再统一一点
    unify = rng.uniform(0.08, 0.18)
    img = mix_color(img, atmosphere[None, None, :], unify)

    # 柔和的上下明暗变化
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    v_strength = rng.uniform(0.015, 0.035)
    vignette = 1.0 - v_strength * ((y - 0.5) ** 2) * 4
    img = img * vignette

    # 不再用纯灰，而是用浅雾色做中间柔化
    if rng.random() < 0.9:
        x = np.linspace(0, 1, width, dtype=np.float32)
        center = rng.uniform(0.35, 0.65)
        sigma = rng.uniform(0.14, 0.24)
        strength = rng.uniform(0.06, 0.16)

        mist = mix_color(atmosphere, np.array([245, 245, 245], dtype=np.float32), 0.45)
        center_weight = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        center_weight = (strength * center_weight)[None, :, None]
        img = img * (1 - center_weight) + mist * center_weight

    return np.clip(img, 0, 255)


def generate_linear_gradient(width, height, colors, rng):
    xx, yy = get_xy_grid(width, height)

    c1 = colors[0]
    c2 = colors[1]

    angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    t = xx * direction[0] + yy * direction[1]
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    t = smoothstep(t)[..., None]

    img = (1 - t) * c1 + t * c2

    if rng.random() < 0.45 and len(colors) >= 3:
        c3 = colors[2]
        angle2 = angle + rng.uniform(0.8, 1.5)
        direction2 = np.array([np.cos(angle2), np.sin(angle2)], dtype=np.float32)
        t2 = xx * direction2[0] + yy * direction2[1]
        t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
        t2 = smoothstep(t2)[..., None]
        mix = rng.uniform(0.10, 0.22)
        img = img * (1 - mix * t2) + c3 * (mix * t2)

    return img


def generate_blob_gradient(width, height, colors, rng):
    xx, yy = get_xy_grid(width, height)

    n_blobs = rng.integers(3, min(6, len(colors) + 2))
    weights = []
    blob_colors = []

    for i in range(n_blobs):
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        sx = rng.uniform(0.14, 0.30)
        sy = rng.uniform(0.14, 0.30)

        d = ((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)
        w = np.exp(-d)

        # 更柔和的底权重，避免硬切和脏边
        w = w + rng.uniform(0.03, 0.07)
        weights.append(w)
        blob_colors.append(colors[i % len(colors)])

    weights = np.stack(weights, axis=0)
    weights = weights ** rng.uniform(1.05, 1.35)
    weights = normalize_weights(weights)

    img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(n_blobs):
        img += weights[i][..., None] * blob_colors[i]

    return img


def generate_mesh_gradient(width, height, colors, rng):
    xx, yy = get_xy_grid(width, height)

    n_points = rng.integers(4, 7)
    weights = []
    point_colors = []

    for i in range(n_points):
        px = rng.uniform(0.0, 1.0)
        py = rng.uniform(0.0, 1.0)

        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) + 1e-4

        # 降低幂次波动，减少极端抢色
        power = rng.uniform(1.15, 1.75)
        w = 1.0 / (dist ** power)

        weights.append(w)
        point_colors.append(colors[i % len(colors)])

    weights = np.stack(weights, axis=0)
    weights = weights ** rng.uniform(0.90, 1.10)
    weights = normalize_weights(weights)

    img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(n_points):
        img += weights[i][..., None] * point_colors[i]

    return img


def generate_hybrid_gradient(width, height, colors, rng):
    linear = generate_linear_gradient(width, height, colors, rng)
    blobs = generate_blob_gradient(width, height, colors, rng)
    mesh = generate_mesh_gradient(width, height, colors, rng)

    # blob 占比更高，画面会更柔和
    w1 = rng.uniform(0.08, 0.18)
    w2 = rng.uniform(0.42, 0.58)
    w3 = 1.0 - w1 - w2
    if w3 < 0.20:
        w3 = 0.20
        s = w1 + w2 + w3
        w1, w2, w3 = w1 / s, w2 / s, w3 / s

    img = w1 * linear + w2 * blobs + w3 * mesh
    return img


def generate_gradient_image(width, height, rng):
    mode = rng.choice(["blob", "mesh", "hybrid"], p=[0.40, 0.20, 0.40])
    n_colors = int(rng.integers(3, 5))
    colors, atmosphere = random_palette(rng, n_colors=n_colors)

    if mode == "blob":
        img = generate_blob_gradient(width, height, colors, rng)
    elif mode == "mesh":
        img = generate_mesh_gradient(width, height, colors, rng)
    else:
        img = generate_hybrid_gradient(width, height, colors, rng)

    img = apply_global_tone(img, rng, atmosphere)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mode, colors


def main():
    num_images = 50
    width = 1920
    height = 1080
    output_dir = Path("/mnt/a/hsm/test")
    seed = 42

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(1, num_images + 1):
        img, mode, colors = generate_gradient_image(
            width=width,
            height=height,
            rng=rng,
        )

        save_path = output_dir / f"gradient_{i:04d}.png"
        Image.fromarray(img).save(save_path)

        color_info = [tuple(int(v) for v in color) for color in colors]
        print(f"[{i:04d}] saved: {save_path} | mode={mode} | colors={color_info}")

    print(f"完成，共生成 {num_images} 张图片，目录: {output_dir}")


if __name__ == "__main__":
    main()