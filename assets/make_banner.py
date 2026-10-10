"""Render assets/banner.png (README hero) and assets/qwen3.8-flash-next-banner.png (model logo).

usage: python3 assets/make_banner.py
Numbers on the chips come from benchmarks/*.json so the banner can't drift from the README.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "benchmarks")
FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_axes([100, weight])
        except Exception:
            pass
    return f


def logo_rgba(size):
    """Qwen mark with its pale tile knocked out to transparency."""
    im = Image.open(os.path.join(HERE, "qwen-logo.png")).convert("RGBA").resize((size, size), Image.LANCZOS)
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if min(r, g, b) > 200:                      # tile background
                px[x, y] = (r, g, b, 0)
            elif min(r, g, b) > 120:                    # anti-aliased edge
                px[x, y] = (r, g, b, int(a * (200 - min(r, g, b)) / 80))
    return im


def stats():
    t = json.load(open(os.path.join(BENCH, "throughput_result.json")))["results"]
    tf = next(v for k, v in t.items() if "TensorFold" in k)
    r = json.load(open(os.path.join(BENCH, "reasoning_result_tensorfold.json")))
    return [
        (f"{tf['peak_aggregate_tok_s']:.0f} tok/s", "peak output"),
        (f"{tf['peak_at_users']} users", "concurrent"),
        ("262K", "context"),
        (f"{r['math500']['accuracy'] * 100:.0f}%", "MATH-500"),
    ]


def hero():
    W, H = 1600, 520
    img = Image.new("RGB", (W, H))
    px = img.load()
    # Dark diagonal gradient: deep indigo -> near-black, NVIDIA-green tint on the right.
    for y in range(H):
        for x in range(W):
            t = x / W
            px[x, y] = (int(14 + 6 * t), int(14 + 20 * t * (y / H) + 6), int(34 - 14 * t))
    # DGX Spark photo, faded into the right side.
    photo = Image.open(os.path.join(HERE, "dgx-spark-banner-new.png")).convert("RGB")
    ph = photo.resize((int(photo.width * H / photo.height), H), Image.LANCZOS)
    mask = Image.new("L", ph.size)
    mp = mask.load()
    for x in range(ph.width):
        a = int(255 * min(1, max(0, (x - ph.width * 0.10) / (ph.width * 0.55))) * 0.55)
        for y in range(ph.height):
            mp[x, y] = a
    img.paste(ph, (W - ph.width, 0), mask)
    # Soft glow behind the logo.
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((40, 90, 380, 430), fill=(98, 92, 255, 90))
    img = Image.alpha_composite(img.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(60)))
    img.alpha_composite(logo_rgba(230), (95, 145))

    d = ImageDraw.Draw(img)
    x0 = 370
    d.text((x0, 112), "Qwen3.8-Flash-Next", font=font(FONT, 78, 750), fill=(255, 255, 255))
    d.text((x0, 210), "INT4-AutoRound  ·  TensorFold engine  ·  one NVIDIA DGX Spark (GB10)",
           font=font(FONT, 30, 450), fill=(198, 202, 222))
    # Stat chips: translucent fill on its own layer, then text on top.
    chips = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(chips)
    fb, fs = font(FONT, 34, 700), font(FONT, 20, 450)
    boxes, x = [], x0
    for big, small in stats():
        w = max(d.textlength(big, font=fb), d.textlength(small, font=fs)) + 44
        cd.rounded_rectangle((x, 285, x + w, 385), radius=18, fill=(255, 255, 255, 26),
                             outline=(118, 185, 0, 255), width=2)
        boxes.append((x, big, small))
        x += w + 18
    img.alpha_composite(chips)
    d = ImageDraw.Draw(img)
    for x, big, small in boxes:
        d.text((x + 22, 297), big, font=fb, fill=(255, 255, 255))
        d.text((x + 22, 345), small, font=fs, fill=(170, 220, 90))
    d.text((x0, 425), "16 concurrent users · vision · MTP speculative decoding · metered gateway · Grafana",
           font=font(FONT, 22, 400), fill=(150, 154, 175))
    img.convert("RGB").save(os.path.join(HERE, "banner.png"), optimize=True)
    print("wrote banner.png")


def model_banner():
    W, H = 760, 300
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    img.alpha_composite(logo_rgba(230), (25, 35))
    d = ImageDraw.Draw(img)
    d.text((280, 62), "Qwen 3.8", font=font(FONT, 86, 600), fill=(10, 10, 10))
    d.text((284, 168), "Flash-Next", font=font(BOLD, 56), fill=(10, 10, 10))
    img.convert("RGB").save(os.path.join(HERE, "qwen3.8-flash-next-banner.png"), optimize=True)
    print("wrote qwen3.8-flash-next-banner.png")


hero()
model_banner()
