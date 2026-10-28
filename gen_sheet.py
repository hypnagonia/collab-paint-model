"""Multiview shape generation (no texture) from one character sheet with front / side / back figures in a row.

Usage: python gen_sheet.py SHEET.png NAME [views] [--anon]
views: comma list of what the figures are, left to right, e.g. "front,right,back" (default) or "front,left".
With a single side figure facing screen-left ("left"), it is mirrored into a "right" view (what the painter
expects); with both "left" and "right" given, each is kept as is.
--prep-only: stop after writing the input views (to check them before the long generation).
--anon: replace skin (incl. face and tattoos) with a smooth average tone before anything else, so the model
doesn't reproduce a real person's likeness.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from rembg import remove

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHEET, NAME = args[0], args[1]
KEYS = (args[2] if len(args) > 2 else "front,right,back").split(",")
ANON = "--anon" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_" + NAME)
os.makedirs(OUT, exist_ok=True)

cut = remove(Image.open(SHEET).convert("RGB"))
alpha = np.array(cut)[..., 3] > 128
cols = alpha.any(0)
# figures are the runs of non-empty columns; keep the three widest
runs, start = [], None
for x, on in enumerate(list(cols) + [False]):
    if on and start is None:
        start = x
    elif not on and start is not None:
        runs.append((start, x))
        start = None
runs = sorted(sorted(runs, key=lambda r: r[1] - r[0])[-len(KEYS):])
print("figures at columns", runs)
assert len(runs) == len(KEYS), "expected %d figures side by side" % len(KEYS)


def anonymise(fig):
    """Smooth every skin pixel (face, arms, legs, tattoos) to a blurred average skin tone."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from PIL import ImageFilter
    name = "mattmdjaga/segformer_b2_clothes"
    proc = AutoImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    bg = Image.new("RGBA", fig.size, (128, 128, 128, 255))
    bg.alpha_composite(fig)
    with torch.no_grad():
        logits = model(**proc(images=bg.convert("RGB"), return_tensors="pt")).logits
    lab = torch.nn.functional.interpolate(logits, size=fig.size[::-1], mode="bilinear").argmax(1)[0].numpy()
    skin = np.isin(lab, [11, 12, 13, 14, 15]) & (np.array(fig)[..., 3] > 128)
    px = np.array(fig).astype(np.float32)
    tone = np.median(px[skin][:, :3], 0)
    # large-radius blur of the skin only (normalised so background doesn't bleed in), pulled toward the median
    m = Image.fromarray((skin * 255).astype(np.uint8))
    col = Image.fromarray(np.where(skin[..., None], px[..., :3], 0).astype(np.uint8))
    r = max(fig.size) // 25
    bc = np.array(col.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
    bm = np.array(m.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)[..., None] / 255
    smooth = bc / np.maximum(bm, 1e-3)
    smooth = 0.5 * smooth + 0.5 * tone
    px[skin, :3] = smooth[skin]
    return Image.fromarray(px.astype(np.uint8), "RGBA")


def square(img):
    img = img.crop(img.getbbox())
    side = int(max(img.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


images = {}
for key, (x0, x1) in zip(KEYS, runs):
    fig = cut.crop((x0, 0, x1, cut.height))
    if ANON:
        fig = anonymise(fig)
    if key == "left" and "right" not in KEYS:  # only one side view: mirror it into the right view
        fig = fig.transpose(Image.FLIP_LEFT_RIGHT)
        key = "right"
    fig.crop(fig.getbbox()).save(os.path.join(OUT, f"ref_{key}_crop.png"))
    images[key] = square(fig)
    images[key].save(os.path.join(OUT, f"in_{key}.png"))

if "--prep-only" in sys.argv:
    sys.exit(0)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16", device="mps", dtype=torch.float16)
t = time.time()
mesh = pipe(image=images, num_inference_steps=50, octree_resolution=380, num_chunks=20000,
            generator=torch.manual_seed(12345), output_type="trimesh")[0]
print(f"generated in {time.time() - t:.0f}s, faces={len(mesh.faces)}")
mesh = FloaterRemover()(mesh)
mesh = DegenerateFaceRemover()(mesh)
mesh.export(os.path.join(OUT, NAME + "_raw.glb"))
print("DONE")
