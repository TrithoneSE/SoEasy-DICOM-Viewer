"""PWA icons from the desktop app's logo (assets/icon.png): any-purpose PNGs, an Apple touch icon without
transparency (iOS would fill it with black) and a maskable icon with the safe-zone padding Android expects.
Run from the repo folder: .venv/Scripts/python.exe web/tools/make_icons.py"""
import os

from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
src = Image.open(os.path.join(here, "..", "..", "assets", "icon.png")).convert("RGBA")
out = os.path.join(here, "..", "public", "icons")
os.makedirs(out, exist_ok=True)
border = src.getpixel((src.width // 2, 12))[:3]          # the logo's dark blue frame


def flat(size, scale=1.0):
    bg = Image.new("RGBA", (size, size), border + (255,))
    inner = int(size * scale)
    ic = src.resize((inner, inner), Image.LANCZOS)
    bg.alpha_composite(ic, ((size - inner) // 2, (size - inner) // 2))
    return bg.convert("RGB")


for s in (192, 512):
    src.resize((s, s), Image.LANCZOS).save(os.path.join(out, f"icon-{s}.png"))
flat(180, 1.04).save(os.path.join(out, "apple-touch-icon.png"))
flat(512, 0.8).save(os.path.join(out, "maskable-512.png"))
print("border", border)
