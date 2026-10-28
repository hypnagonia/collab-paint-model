"""Ink lines for a cel-shaded look: a texture-space mask of the borders between the character's parts (skin, hair,
top, skirt, bracers, boots, horns), so the page can draw black lines where one material meets another.

Parts come from the painted colour and the body position per vertex, then per face (majority). The face-wise part
id is baked into UV space; texels whose neighbour (inside the same UV island) has another id become ink.
Writes OUT.png (white = ink, same orientation as the model's texture).

Usage: blender -b MODEL.blend -P ink_parts.py -- OUT.png SIZE
"""
import bpy, sys
import numpy as np

OUT, SIZE = sys.argv[sys.argv.index("--") + 1:][:2]
SIZE = int(SIZE)
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
n = len(me.vertices)
M = np.array(obj.matrix_world)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
x, y, z = co.T
zmax = z.max()
img = next(nd.image for m in me.materials for nd in m.node_tree.nodes if nd.type == "TEX_IMAGE" and nd.image)
W, H = img.size
px = np.empty(W * H * 4, "f"); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)
uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
vuv = np.zeros((n, 2), "f"); vuv[lv] = uv
c = px[np.clip((vuv[:, 1] * H).astype(int), 0, H - 1), np.clip((vuv[:, 0] * W).astype(int), 0, W - 1), :3]


def seg(p, a, b):
    a, b = np.array(a), np.array(b); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


fore = np.minimum.reduce([seg(co, (s * 0.23, 0.0, 1.14), (s * 0.31, -0.04, 0.82)) for s in (1, -1)])
skin = (c[:, 0] > c[:, 2] + 0.06) & (c[:, 0] > 0.2)
dark = c.max(1) < 0.3
part = np.full(n, 6)                                   # 6 other
part[skin] = 0                                         # skin
violet = (c[:, 0] - c[:, 1] > 0.038) & (c[:, 2] > c[:, 1] + 0.03)  # hair: green well below red and blue; the top is greyer
part[~skin & dark & violet & (z > 1.05) & (fore > 0.07)] = 1   # hair
part[~skin & ~violet & (z > 1.18) & (z < 1.42) & (np.abs(x) < 0.2) & (y < -0.06)] = 2  # top (front of the chest)
part[~skin & (z < 1.06) & (z > 0.32)] = 3              # skirt
part[(part == 6) & (z < 1.06) & (z > 0.32)] = 3
part[~skin & (fore < 0.075)] = 4                       # bracers
part[~skin & (z < 0.34)] = 5                           # boots
part[(z > zmax - 0.16) & (c[:, 2] > c[:, 0] + 0.03) & ~skin] = 7  # horns (painted blue-grey)
# the sheer skirt shows skin through it: inside the skirt band skin and skirt are one region (no lines across the
# cloth); the line stays at the waistband, where the skirt meets the bare belly
part[(part == 0) & (z > 0.05) & (z < 1.0)] = 3
print("parts", np.bincount(part, minlength=8))
F = np.array([list(p.vertices)[:3] for p in me.polygons])
fpart = np.array([np.bincount(part[f], minlength=8).argmax() for f in F])
a = me.color_attributes.new("pid", "FLOAT_COLOR", "CORNER")
cols = np.ones((len(me.loops), 4), "f")
for p in me.polygons:
    v = (fpart[p.index] + 1) / 10.0
    for li in p.loop_indices:
        cols[li, :3] = v
a.data.foreach_set("color", cols.ravel())
mat = bpy.data.materials.new("pid"); nt = mat.node_tree; nt.nodes.clear()
vc = nt.nodes.new("ShaderNodeVertexColor"); vc.layer_name = "pid"
em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(vc.outputs["Color"], em.inputs[0])
o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], o.inputs[0])
tgt = bpy.data.images.new("pid", SIZE, SIZE, float_buffer=True)
tgt.pixels.foreach_set(np.zeros(SIZE * SIZE * 4, "f"))
tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = tgt; nt.nodes.active = tn
me.materials.clear(); me.materials.append(mat)
me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int64))
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 1; sc.cycles.device = "CPU"
sc.render.bake.margin = 0; sc.render.bake.use_clear = False
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.object.bake(type="EMIT")
pid = np.empty(SIZE * SIZE * 4, "f"); tgt.pixels.foreach_get(pid)
pid = np.round(pid.reshape(SIZE, SIZE, 4)[..., 0] * 10).astype(int)  # 0 = empty, 1..8 parts
ink = np.zeros_like(pid, bool)
for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
    nb = np.roll(pid, (dy, dx), (0, 1))
    ink |= (pid > 0) & (nb > 0) & (nb != pid)
for _ in range(1):  # 2-3 texels wide
    ink = ink | np.roll(ink, 1, 0) | np.roll(ink, 1, 1)
ink &= pid > 0
# then grow the ink into empty gap texels next to it, so texture filtering at island borders keeps the line
for _ in range(3):
    g = (np.roll(ink, 1, 0) | np.roll(ink, -1, 0) | np.roll(ink, 1, 1) | np.roll(ink, -1, 1)) & (pid == 0)
    ink |= g
print("ink texels %.2f%%" % (100 * ink.mean()))
out = bpy.data.images.new("ink", SIZE, SIZE)
v = ink.astype("f")
rgba = np.stack([v, v, v, np.ones_like(v)], -1)
out.pixels.foreach_set(rgba.ravel())
out.filepath_raw = OUT; out.file_format = "PNG"; out.save()
print("DONE")
