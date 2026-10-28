"""Paste the close-up face photo into the full-body photos (front, both profiles) at 2x, aligned on landmarks,
colour-matched and feathered, so the reference views carry a sharp head. Then build the normalised sheet.
Landmarks were read off gridded close-ups by hand (pixels in the original images).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from rembg import remove

UP = 2
face = Image.open("s-face.jpeg").convert("RGB")
FF = face.crop((0, 0, 512, 559))   # front close-up
FP = face.crop((512, 0, 1024, 559))  # profile close-up, faces screen-left
# (photo landmark A, B in close-up) -> (same landmarks in body photo); A/B = eye centre & mouth, or nose tip & ear
JOBS = {
    "front": ("s-front.jpeg", None, FF, ((270, 243), (270, 343)), ((515.5, 80), (512, 102))),
    "left": ("s-left.jpeg", None, FP, ((627 - 512, 282), (785 - 512, 250)), ((480, 90), (518, 84))),
    "right": ("s-right-but-2img.jpeg", (0, 0, 512, 559), FP.transpose(Image.FLIP_LEFT_RIGHT),
              ((512 - (627 - 512), 282), (512 - (785 - 512), 250)), None),
}


def head_mask(size, kind):
    """Region of the close-up to keep: head, horns and the hair down to the chin line (not the shoulders)."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if kind == "front":
        d.ellipse((40, -40, 472, 470), fill=255)
    elif kind == "face_only":  # mirrored profile: only the face in front of the ear; her hair covers the ear there
        d.ellipse((285, 185, 430, 415), fill=255)
    else:
        d.ellipse((80, -40, 470, 450), fill=255)
    return m.filter(ImageFilter.GaussianBlur(18))


figs = {}
for key, (path, box, close, (pa, pb), body_lm) in JOBS.items():
    body = Image.open(path).convert("RGB")
    if box:
        body = body.crop(box)
    if body_lm is None:  # right profile: same landmarks as the left one, read off its own photo
        body_lm = ((292, 92), (254, 82))  # nose tip, ear (faces screen-right)
    body = body.resize((body.width * UP, body.height * UP), Image.LANCZOS)
    ba, bb = np.array(body_lm[0]) * UP, np.array(body_lm[1]) * UP
    s = np.linalg.norm(bb - ba) / np.linalg.norm(np.array(pb) - np.array(pa))
    if key == "front":  # eye-mouth distance is short and noisy: blend with the profile scale
        s = 0.5 * s + 0.5 * 0.21 * UP
    cw, ch = round(close.width * s), round(close.height * s)
    c = close.resize((cw, ch), Image.LANCZOS)
    # keep only the close-up's figure (no studio background halo), inside the head region
    fg = remove(close).getchannel("A")
    m = Image.fromarray(np.minimum(np.array(head_mask(close.size, {"front": "front", "left": "profile"}.get(key, "face_only"))),
                                   np.array(fg.filter(ImageFilter.GaussianBlur(2))))).resize((cw, ch))
    off = (round(ba[0] - pa[0] * s), round(ba[1] - pa[1] * s))
    # colour match: per-channel gain so the close-up's skin/hair tones match the body photo's lighting
    region = np.array(body.crop((off[0], off[1], off[0] + cw, off[1] + ch))).astype(np.float32)
    mm = np.array(m).astype(np.float32)[..., None] / 255 > 0.5
    cc = np.array(c).astype(np.float32)
    # skin-like pixels only (red > blue), so the violet hair in the region doesn't tint the face
    sk = mm[..., 0] & (cc[..., 0] > cc[..., 2] + 15) & (region[..., 0] > region[..., 2] + 15)
    gain = np.clip(region[sk].mean(0) / (cc[sk].mean(0) + 1e-3), 0.85, 1.2)
    c = Image.fromarray(np.clip(cc * gain, 0, 255).astype(np.uint8))
    body.paste(c, off, m)
    print(key, "scale %.3f offset %s gain %s" % (s, off, np.round(gain, 2)))
    body.save(f"comp_{key}.png")
    figs[key] = body

figs["back"] = Image.open("s-back.jpeg").convert("RGB")
figs["back"] = figs["back"].resize((figs["back"].width * UP, figs["back"].height * UP), Image.LANCZOS)
order = ["front", "right_photo", "back", "left_photo"]
cut = []
for key in ("front", "left", "back", "right"):  # left-to-right as in the first sheet
    f = remove(figs[key])
    cut.append(f.crop(f.getbbox()))
H = 2000
cut = [f.resize((round(f.width * H / f.height), H), Image.LANCZOS) for f in cut]
gap = 160
sheet = Image.new("RGB", (sum(f.width for f in cut) + gap * 5, H + 2 * gap), (255, 255, 255))
x = gap
for f in cut:
    sheet.paste(f, (x, gap), f)
    x += f.width + gap
sheet.save("s3_sheet_face.png")
print(sheet.size)
