"""Generate thumbnail images for README demo videos."""
from PIL import Image, ImageDraw, ImageFont
import os

THUMBNAILS = {
    "assets/中文-操作-thumb.png": {
        "text": "拖拽操作演示",
        "subtitle": "Drag & Drop Demo",
    },
    "assets/中文-监控-thumb.png": {
        "text": "监控归类演示",
        "subtitle": "Auto Monitor Demo",
    },
    "assets/English-Operate-thumb.png": {
        "text": "Drag & Drop",
        "subtitle": "Watch the demo",
    },
    "assets/English-Monitor-thumb.png": {
        "text": "Auto Monitor",
        "subtitle": "Watch the demo",
    },
}

WIDTH, HEIGHT = 800, 450

def create_thumbnail(path, text, subtitle):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (30, 30, 50, 255))
    draw = ImageDraw.Draw(img)

    # --- Background gradient (dark blue to purple-ish) ---
    for y in range(HEIGHT):
        r = int(30 + (y / HEIGHT) * 20)
        g = int(30 + (y / HEIGHT) * 10)
        b = int(50 + (y / HEIGHT) * 40)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    # --- Grid lines for visual interest ---
    for x in range(0, WIDTH, 40):
        draw.line([(x, 0), (x, HEIGHT)], fill=(255, 255, 255, 6))
    for y in range(0, HEIGHT, 40):
        draw.line([(0, y), (WIDTH, y)], fill=(255, 255, 255, 6))

    # --- Center play button (large circle + triangle) ---
    cx, cy = WIDTH // 2, HEIGHT // 2 - 20
    radius = 50
    # Outer glow
    draw.ellipse([cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6],
                 fill=(255, 255, 255, 20))
    # Outer circle
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill=(255, 255, 255, 30), outline=(255, 255, 255, 180), width=3)
    # Play triangle
    tri_size = 30
    triangle = [
        (cx - tri_size // 2 + 5, cy - tri_size + 5),
        (cx - tri_size // 2 + 5, cy + tri_size - 5),
        (cx + tri_size - 5, cy),
    ]
    draw.polygon(triangle, fill=(255, 255, 255, 220))

    # --- Title text ---
    try:
        font_large = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 36)
        font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    except (IOError, OSError):
        try:
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 20)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Text shadow
    shadow_offset = 2
    bbox = draw.textbbox((0, 0), text, font=font_large)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2 + shadow_offset, cy + radius + 20 + shadow_offset),
              text, fill=(0, 0, 0, 100), font=font_large)
    draw.text(((WIDTH - tw) // 2, cy + radius + 20),
              text, fill=(255, 255, 255, 230), font=font_large)

    # Subtitle
    bbox2 = draw.textbbox((0, 0), subtitle, font=font_small)
    sw = bbox2[2] - bbox2[0]
    draw.text(((WIDTH - sw) // 2 + shadow_offset, cy + radius + 65 + shadow_offset),
              subtitle, fill=(0, 0, 0, 80), font=font_small)
    draw.text(((WIDTH - sw) // 2, cy + radius + 65),
              subtitle, fill=(200, 200, 220, 180), font=font_small)

    # --- Bottom hint ---
    hint = "▶ 点击播放视频"
    bbox3 = draw.textbbox((0, 0), hint, font=font_small)
    hw = bbox3[2] - bbox3[0]
    draw.text((WIDTH - hw - 20, HEIGHT - 35), hint, fill=(255, 255, 255, 60), font=font_small)

    # Convert to RGB for saving as PNG
    img = img.convert("RGB")
    img.save(path, "PNG")
    print(f"  OK Created {path}")

def main():
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(base)
    print("Generating thumbnails...")
    for path, info in THUMBNAILS.items():
        create_thumbnail(path, info["text"], info["subtitle"])
    print("\nDone! All thumbnails created.")

if __name__ == "__main__":
    main()
