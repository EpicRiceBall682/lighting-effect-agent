import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

def random_rgb(rng, sat_min=0.55, sat_max=0.9, val_min=0.8, val_max=1.0):
    h = rng.random()
    s = rng.uniform(sat_min, sat_max)
    v = rng.uniform(val_min, val_max)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r * 255, g * 255, b * 255], dtype=np.float32), h

def random_color_pair(rng):
    left, h1 = random_rgb(rng)

    # 让另一侧颜色与左侧有较明显色相差，避免太接近
    hue_gap = rng.uniform(0.25, 0.60)
    h2 = (h1 + hue_gap) % 1.0
    s2 = rng.uniform(0.55, 0.9)
    v2 = rng.uniform(0.8, 1.0)
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s2, v2)
    right = np.array([r2 * 255, g2 * 255, b2 * 255], dtype=np.float32)

    return left, right

def generate_gradient_image(width, height, left_color, right_color, rng):
    x = np.linspace(0, 1, width, dtype=np.float32)
    t = x[None, :, None]
    # 横向线性渐变
    row = (1 - t) * left_color + t * right_color
    img = np.repeat(row, height, axis=0)

    # 中间轻微灰化（随机强度与范围）
    gray = np.array([150, 150, 150], dtype=np.float32)
    sigma = rng.uniform(0.12, 0.20)
    max_gray_strength = rng.uniform(0.18, 0.35)
    center_weight = np.exp(-((x - 0.5) ** 2) / (2 * (sigma ** 2)))
    center_weight = (max_gray_strength * center_weight)[None, :, None]
    img = img * (1 - center_weight) + gray * center_weight

    # 轻微纵向明暗变化（随机轻度）
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    v_strength = rng.uniform(0.02, 0.06)
    vignette = 1.0 - v_strength * ((y - 0.5) ** 2) * 4
    img = img * vignette

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def main():

    num_images = 50
    width = 1920
    height = 1080
    output_dir = "/mnt/a/hsm/test"
    seed = 42
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for i in range(1, num_images + 1):
        left_color, right_color = random_color_pair(rng)
        img = generate_gradient_image(
            width=width,
            height=height,
            left_color=left_color,
            right_color=right_color,
            rng=rng,
        )

        save_path = output_dir / f"gradient_{i:04d}.png"
        Image.fromarray(img).save(save_path)

        left_int = tuple(int(v) for v in left_color)
        right_int = tuple(int(v) for v in right_color)
        print(f"[{i:04d}] saved: {save_path} | left={left_int} right={right_int}")

    print(f"完成，共生成 {num_images} 张图片，目录: {output_dir}")

if __name__ == "__main__":
    main()