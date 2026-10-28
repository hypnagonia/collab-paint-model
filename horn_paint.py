"""Paint the horns of a Hunyuan-painted model (it paints them the same dark violet as the hair).

Horns are found by shape: the two columns that rise above the top of the hair, grown along the surface down into
the curls that stick out beside the head. They get a procedural horn surface (blue-slate keratin with growth
ridges along the length and a lighter worn tip), baked into the model's own texture through its UVs.

Usage: blender -b MODEL.blend -P horn_paint.py -- OUT.glb
"""
import bpy, sys
import numpy as np

OUT = sys.argv[sys.argv.index("--") + 1]
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
n = len(me.vertices)
M = np.array(obj.matrix_world)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
x, y, z = co.T
ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
src = np.concatenate([ed[:, 0], ed[:, 1]]); dst = np.concatenate([ed[:, 1], ed[:, 0]])
# UV seams split vertices; join coincident ones so growth crosses seams
key = np.round(co * 2e4).astype(np.int64)
_, grp = np.unique(key, axis=0, return_inverse=True)
grp = grp.ravel()

zt = z.max()
top = z > zt - 0.06
xc = np.median(x[(z > zt - 0.25) & (z < zt - 0.15)])  # head centre line
# seeds: the tips standing clear of the hair crown
seed = top & (np.abs(x - xc) > 0.075)
# allowed: above the crown, or out beside the head where the horns curl
head_half = 0.12
allowed = (z > zt - 0.15) & ((z > zt - 0.065) | (np.abs(x - xc) > head_half))
horn = seed.copy()
for _ in range(400):
    g = np.zeros(n, bool); g[dst[horn[src]]] = True
    g &= allowed
    gg = np.zeros(grp.max() + 1, bool); gg[grp[g | horn]] = True
    g = gg[grp]
    if (g & ~horn).sum() == 0:
        break
    horn |= g
# do not swallow the crown of the hair: keep only the side regions above the crown line
horn &= ~((np.abs(x - xc) < 0.06) & (z < zt - 0.02))
print("horn vertices %d (seeds %d), head centre x=%.3f, top z=%.3f" % (horn.sum(), seed.sum(), xc, zt))
w = horn.astype(float)
for _ in range(2):
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, dst, w[src]); np.add.at(cnt, dst, 1)
    w = np.maximum(w, acc / np.maximum(cnt, 1))
a = me.color_attributes.new("horn_w", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = w[:, None]; a.data.foreach_set("color", c4.ravel())
# distance along the horn from its root: 0 at the base, 1 at the tip (for the worn lighter tip)
along = np.clip((z - (zt - 0.12)) / 0.12, 0, 1)
b = me.color_attributes.new("horn_t", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = along[:, None]; b.data.foreach_set("color", c4.ravel())

# the image the model's material actually shows (older images may still sit unused in the file)
tex = next(nd.image for m in me.materials for nd in m.node_tree.nodes if nd.type == "TEX_IMAGE" and nd.image)
print("painting over", tex.name)
TW, TH = tex.size
mat = bpy.data.materials.new("horn_bake"); nt = mat.node_tree; nt.nodes.clear()
L = nt.links.new
def N(t, **kw):
    nd = nt.nodes.new(t)
    for k, v in kw.items(): setattr(nd, k, v)
    return nd
lin = lambda c: tuple((v / 255) ** 2.2 for v in c) + (1,)
tc = N("ShaderNodeTexCoord")
wv = N("ShaderNodeTexWave", wave_type="RINGS", rings_direction="Z", wave_profile="SAW")
wv.inputs["Scale"].default_value = 32  # coarse rings: fine ones shimmer at a distance; wv.inputs["Distortion"].default_value = 0.8; wv.inputs["Detail"].default_value = 3
L(tc.outputs["Object"], wv.inputs["Vector"])
nz = N("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60; nz.inputs["Detail"].default_value = 6
L(tc.outputs["Object"], nz.inputs["Vector"])
mf = N("ShaderNodeMix", data_type="FLOAT"); mf.inputs["Factor"].default_value = 0.25
L(wv.outputs["Fac"], mf.inputs[2]); L(nz.outputs["Fac"], mf.inputs[3])
rp = N("ShaderNodeValToRGB"); cr = rp.color_ramp
for i, (pos, c) in enumerate([(0.25, (20, 21, 32)), (0.6, (34, 37, 54)), (0.95, (52, 57, 78))]):
    e = cr.elements[i] if i < 2 else cr.elements.new(pos)
    e.position, e.color = pos, lin(c)
L(mf.outputs[0], rp.inputs["Fac"])
tip = N("ShaderNodeVertexColor", layer_name="horn_t")
tipc = N("ShaderNodeMix", data_type="RGBA"); L(tip.outputs["Color"], tipc.inputs["Factor"])
L(rp.outputs["Color"], tipc.inputs[6]); tipc.inputs[7].default_value = lin((92, 94, 108))
tipr = N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY"); tipr.inputs["Factor"].default_value = 1
L(tipc.outputs[2], tipr.inputs[6]); L(rp.outputs["Color"], tipr.inputs[7])
hornc = N("ShaderNodeMix", data_type="RGBA"); hornc.inputs["Factor"].default_value = 0.8
L(rp.outputs["Color"], hornc.inputs[6]); L(tipc.outputs[2], hornc.inputs[7])
uvn = N("ShaderNodeUVMap", uv_map=me.uv_layers.active.name)
tt = N("ShaderNodeTexImage", image=tex); L(uvn.outputs["UV"], tt.inputs["Vector"])
hw = N("ShaderNodeVertexColor", layer_name="horn_w")
mx = N("ShaderNodeMix", data_type="RGBA"); L(hw.outputs["Color"], mx.inputs["Factor"])
L(tt.outputs["Color"], mx.inputs[6]); L(hornc.outputs[2], mx.inputs[7])
em = N("ShaderNodeEmission"); L(mx.outputs[2], em.inputs[0])
out = N("ShaderNodeOutputMaterial"); L(em.outputs[0], out.inputs[0])
new = bpy.data.images.new("painted_horns", TW, TH)  # same size as the face-painted texture
tn = N("ShaderNodeTexImage", image=new); nt.nodes.active = tn
old = list(me.materials); me.materials.clear(); me.materials.append(mat)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 2; sc.cycles.device = "CPU"; sc.render.bake.margin = 8
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
new.generated_color = (0, 0, 0, 1)
bpy.ops.object.bake(type="EMIT")
# keep the gap colours of the source (already filled) instead of black: copy unbaked texels from it
_w, _h = new.size
_a = np.empty(_w * _h * 4, "f"); new.pixels.foreach_get(_a)
_t = np.empty(_w * _h * 4, "f"); tex.pixels.foreach_get(_t)
_a = _a.reshape(-1, 4); _t = _t.reshape(-1, 4)
_m = _a[:, :3].sum(1) < 1e-4
_a[_m] = _t[_m]
new.pixels.foreach_set(_a.ravel())
new.pack()
me.materials.clear()
for m in old:
    me.materials.append(m)
    for nd in m.node_tree.nodes:
        if nd.type == "TEX_IMAGE" and nd.image is tex:
            nd.image = new
for nm in ("horn_w", "horn_t"):
    me.color_attributes.remove(me.color_attributes[nm])
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".glb", ".blend"))
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_image_format="JPEG", export_vertex_color="NONE")
print("DONE")
