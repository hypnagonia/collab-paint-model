"""Put the photo's face (eyes, brows, lips) onto a model painted by Hunyuan3D-2 Paint, whose face comes out soft.

The front photo is placed by scale (photo height = figure height) and by one landmark: the photo's nose tip
(EYE + 0.57 of the eye-to-mouth distance) lands on the mesh's nose tip (the most forward point of the face).
It is projected straight from the front per pixel in the shader, colour-matched to the painted skin, blended in
with a feathered oval mask over the inner face (below the hairline), and baked into the model's own texture.

Usage: blender -b MODEL.blend -P face_overlay.py -- FRONT_CROP.png EYE_X EYE_Y MOUTH_Y OUT.glb
       (EYE/MOUTH in crop pixels, origin top-left)
"""
import bpy, sys
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
FRONT, EX, EY, MY, OUT = a[0], float(a[1]), float(a[2]), float(a[3]), a[4]
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
n = len(me.vertices)
M = np.array(obj.matrix_world)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
nr = np.empty(n * 3, "f"); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3) @ M[:3, :3].T
nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
x, y, z = co.T
ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
zmin, zmax = z.min(), z.max()

ph = bpy.data.images.load(FRONT)
W, H = ph.size
k = H / (zmax - zmin)                       # photo pixels per metre: the crop spans the whole figure
# the painted texture at every vertex (to tell skin from hair)
tex = next(im for im in bpy.data.images if im is not ph and im.size[0] >= 512)
TW, TH = tex.size
tp = np.empty(TW * TH * 4, "f"); tex.pixels.foreach_get(tp); tp = tp.reshape(TH, TW, 4)
uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
vuv = np.zeros((n, 2), "f"); vuv[lv] = uv
painted = tp[np.clip((vuv[:, 1] * TH).astype(int), 0, TH - 1), np.clip((vuv[:, 0] * TW).astype(int), 0, TW - 1), :3]
skin = (painted[:, 0] > painted[:, 2] + 0.06) & (painted[:, 0] > 0.18)
# mesh nose tip: most forward skin-coloured vertex on the front of the head (a hair lock can stick out further)
band = (z > zmax - 0.28) & (z < zmax - 0.10) & skin & (nr[:, 1] < -0.5)
i = np.argmin(np.where(band, y, 9))
nx, nz = x[i], z[i]
pnx, pny = EX, EY + 0.57 * (MY - EY)          # photo nose tip, top-left pixels
# photo pixel (u, v_from_bottom) for a point: u = pnx + (x - nx) k ; v = (H - pny) + (z - nz) k
print("nose tip on mesh x=%.3f z=%.3f; scale %.0f px/m" % (nx, nz, k))

# face oval around the nose: brows to chin, cheek to cheek, front-facing only
eye_mouth = (MY - EY) / k                   # metres between eyes and mouth (~0.04)
fz = nz + 0.2 * eye_mouth
oval = ((x - nx) / (1.45 * eye_mouth)) ** 2 + ((z - fz) / (2.1 * eye_mouth)) ** 2
facing = np.clip((-nr[:, 1] - 0.3) / 0.3, 0, 1)
mask = np.clip((1.0 - oval) / 0.6, 0, 1) ** 1.5 * facing
# keep the photo off the hair locks that fall across the face
hairish = ~skin
mask = mask * np.where(hairish, 0.0, 1.0)
print("face vertices: %d, eye-mouth %.3f m" % ((mask > 0.5).sum(), eye_mouth))

pp = np.empty(W * H * 4, "f"); ph.pixels.foreach_get(pp); pp = pp.reshape(H, W, 4)
U = np.clip((pnx + (x - nx) * k).astype(int), 0, W - 1)
V = np.clip(((H - pny) + (z - nz) * k).astype(int), 0, H - 1)
photo = pp[V, U, :3]
# colour transfer: the photo face takes the painted body's skin statistics (mean and spread per channel), measured
# on the neck and upper chest, so face and body read as one skin. Done in linear values, like the shader.
# above the brows the photo shows her hair falling over the forehead: keep the photo only where it is skin there,
# so those locks stay painted hair instead of a brightened photo of hair
ph_skin = (photo[:, 0] > photo[:, 2] + 0.04) & (photo.mean(1) > 0.16)
forehead = z > nz + 1.0 * eye_mouth
keep = np.where(forehead, ph_skin, True).astype(float)
for _ in range(3):  # soften the cut
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, ed[:, 0], keep[ed[:, 1]]); np.add.at(acc, ed[:, 1], keep[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    keep = 0.5 * keep + 0.5 * acc / np.maximum(cnt, 1)
mask = mask * keep
print("forehead: photo kept on %d of %d vertices" % (((keep > 0.5) & forehead & (mask > 0)).sum(), (forehead & (mask > 0)).sum()))
sel = (mask > 0.5) & skin
body_skin = skin & (z < nz - 0.06) & (z > nz - 0.32) & (np.abs(x - nx) < 0.12) & (nr[:, 1] < -0.2)
lin = lambda c: np.power(np.clip(c, 0, 1), 2.2)
ps, bs = lin(photo[sel]), lin(painted[body_skin])
scale = np.clip(bs.std(0) / (ps.std(0) + 1e-4), 0.8, 1.25)  # keep the eyes and lips contrasty
off = bs.mean(0) - ps.mean(0) * scale
print("skin match: %d face / %d body vertices, scale %s offset %s" % (sel.sum(), body_skin.sum(), np.round(scale, 2), np.round(off, 3)))

a_ = me.color_attributes.new("face_w", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = mask[:, None]; a_.data.foreach_set("color", c4.ravel())

# shader: per-pixel front projection of the photo (no vertex-colour blur), mixed over the painted texture
mat = bpy.data.materials.new("face_bake"); nt = mat.node_tree; nt.nodes.clear()
L = nt.links.new
def N(t, **kw):
    nd = nt.nodes.new(t)
    for kk, vv in kw.items(): setattr(nd, kk, vv)
    return nd
geo = N("ShaderNodeNewGeometry"); sep = N("ShaderNodeSeparateXYZ"); L(geo.outputs["Position"], sep.inputs[0])
def lin(src, scale, off):  # scale * src + off
    m = N("ShaderNodeMath", operation="MULTIPLY_ADD"); L(src, m.inputs[0])
    m.inputs[1].default_value = scale; m.inputs[2].default_value = off
    return m.outputs[0]
u_ = lin(sep.outputs["X"], k / W, (pnx - nx * k) / W)
v_ = lin(sep.outputs["Z"], k / H, ((H - pny) - nz * k) / H)
cmb = N("ShaderNodeCombineXYZ"); L(u_, cmb.inputs[0]); L(v_, cmb.inputs[1])
pt = N("ShaderNodeTexImage", image=ph, extension="CLIP", interpolation="Cubic"); L(cmb.outputs[0], pt.inputs["Vector"])
gn = N("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
L(pt.outputs["Color"], gn.inputs[0]); gn.inputs[1].default_value = tuple(float(v) for v in scale)
gn.inputs[2].default_value = tuple(float(v) for v in off)
uvn = N("ShaderNodeUVMap", uv_map=me.uv_layers.active.name)
tt = N("ShaderNodeTexImage", image=tex); L(uvn.outputs["UV"], tt.inputs["Vector"])
fw = N("ShaderNodeVertexColor", layer_name="face_w")
mx = N("ShaderNodeMix", data_type="RGBA"); L(fw.outputs["Color"], mx.inputs["Factor"])
L(tt.outputs["Color"], mx.inputs[6]); L(gn.outputs[0], mx.inputs[7])
em = N("ShaderNodeEmission"); L(mx.outputs[2], em.inputs[0])
out = N("ShaderNodeOutputMaterial"); L(em.outputs[0], out.inputs[0])
new = bpy.data.images.new("painted_face", TW, TH)
tn = N("ShaderNodeTexImage", image=new); nt.nodes.active = tn
old = list(me.materials); me.materials.clear(); me.materials.append(mat)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 4; sc.cycles.device = "CPU"; sc.render.bake.margin = 8
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.object.bake(type="EMIT")
new.pack()
me.materials.clear()
for m in old:
    me.materials.append(m)
    for nd in m.node_tree.nodes:
        if nd.type == "TEX_IMAGE" and nd.image is tex:
            nd.image = new
me.color_attributes.remove(me.color_attributes["face_w"])
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".glb", ".blend"))
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_image_format="JPEG", export_vertex_color="NONE")
print("DONE")
