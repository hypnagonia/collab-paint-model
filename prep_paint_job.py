"""Do everything the Colab painter does on the CPU here, so the GPU session only paints:
clean the mesh, reduce it to 60k faces, UV-unwrap it (xatlas), and recenter the photos the way the painter does.

Usage: python prep_paint_job.py RAW.glb OUT_DIR front.png [left.png back.png right.png]
"""
import os, sys, time
import numpy as np
import trimesh, xatlas
from PIL import Image
from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer

raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t = time.time()
mesh = trimesh.load(raw, force="mesh")
mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=60000)
vmap, idx, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
m = trimesh.Trimesh(mesh.vertices[vmap], idx, process=False)
m.visual = trimesh.visual.TextureVisuals(uv=uvs)
m.export(os.path.join(out, "mesh.glb"))
print("mesh: %d faces, unwrapped in %.0f s" % (len(idx), time.time() - t))


def recenter(img, border=0.2):
    """Same as Hunyuan3DPaintPipeline.recenter_image: crop to the figure, pad to a square with a 20% border."""
    a = np.array(img.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(a > 0)
    c = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(c.size) / (1 - border))
    sq = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
    return sq


for p in sys.argv[3:]:
    recenter(Image.open(p)).save(os.path.join(out, os.path.basename(p)))
    print("photo", os.path.basename(p))
