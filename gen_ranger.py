"""Multiview shape generation (no texture) for the ranger screenshots on Apple Silicon."""
import os
import sys
import time

import torch
from PIL import Image
from rembg import remove

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

DESKTOP = os.path.expanduser("~/Desktop")
views = {
    "front": "Screenshot 2026-10-28 at 01.03.45.png",
    "right": "Screenshot 2026-10-28 at 01.06.32.png",  # character faces screen-right -> we see its right side
}
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
octree = int(sys.argv[2]) if len(sys.argv) > 2 else 380


def prep(path, key):
    img = remove(Image.open(path).convert("RGB"))  # RGBA with background cut out
    bbox = img.getbbox()
    img = img.crop(bbox)
    # pad to a square canvas so the figure keeps its proportions
    side = int(max(img.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    canvas.save(os.path.join(OUT, f"in_{key}.png"))
    return canvas


images = {k: prep(os.path.join(DESKTOP, v), k) for k, v in views.items()}

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv",
    variant="fp16",
    device="mps",
    dtype=torch.float16,
)

t = time.time()
mesh = pipe(
    image=images,
    num_inference_steps=steps,
    octree_resolution=octree,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type="trimesh",
)[0]
print(f"generated in {time.time() - t:.0f}s, faces={len(mesh.faces)}")

mesh = FloaterRemover()(mesh)
mesh = DegenerateFaceRemover()(mesh)
mesh.export(os.path.join(OUT, "ranger_raw.glb"))
print("DONE")
