"""Low-poly copy of a rigged, animated model for real-time use: decimate every skinned mesh (weights carry over),
keep its UVs and shrink the textures after growing the paint into the gutters, export a small GLB with the armature and its action.

Usage: blender -b NAME_rigged.blend -P lowpoly.py -- TARGET_TRIS TEX_SIZE OUT.glb
"""
import bpy, sys

tris, tex, out = sys.argv[sys.argv.index("--") + 1:][:3]
tris, tex = int(tris), int(tex)
meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
total = sum(len(o.data.polygons) for o in meshes)
highs = {}
for o in meshes:  # untouched full-detail copies to bake from
    h = o.copy(); h.data = o.data.copy(); h.modifiers.clear(); h.parent = None
    h.matrix_world = o.matrix_world
    bpy.context.scene.collection.objects.link(h)
    highs[o.name] = h
for o in meshes:
    bpy.context.view_layer.objects.active = o
    d = o.modifiers.new("lowpoly", "DECIMATE")
    d.ratio = tris / total
    while o.modifiers[0] != d:
        bpy.ops.object.modifier_move_up(modifier=d.name)
    bpy.ops.object.modifier_apply(modifier=d.name)
    print(o.name, len(o.data.polygons), "faces")
import numpy as np


def dilate(im, steps):
    """Grow the baked islands into the empty (black) gutter between them, so shrinking the texture and the GPU's
    mipmaps don't pull black into the island edges (it showed as black lines over the face and body)."""
    w, h = im.size
    px = np.empty(w * h * 4, "f"); im.pixels.foreach_get(px); px = px.reshape(h, w, 4)
    empty = px[..., :3].sum(-1) < 0.004
    for _ in range(steps):
        if not empty.any():
            break
        acc = np.zeros_like(px[..., :3]); cnt = np.zeros((h, w))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(px[..., :3], (dy, dx), (0, 1)); ok = ~np.roll(empty, (dy, dx), (0, 1))
            acc += sh * ok[..., None]; cnt += ok
        grow = empty & (cnt > 0)
        px[grow, :3] = acc[grow] / cnt[grow, None]
        empty &= ~grow
    im.pixels.foreach_set(px.ravel())


for im in bpy.data.images:
    if im.size[0] > tex:
        dilate(im, 48)
        im.scale(tex, tex)
for h in highs.values():
    bpy.data.objects.remove(h)
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
next(o for o in bpy.data.objects if o.type == "ARMATURE").select_set(True)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_image_format="JPEG",
                          export_jpeg_quality=80, export_animations=True, export_force_sampling=True,
                          export_frame_range=True, export_vertex_color="NONE")
print("DONE")
