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
tex = next(nd.image for m in me.materials for nd in m.node_tree.nodes if nd.type == "TEX_IMAGE" and nd.image)
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
fh = np.clip((z - (nz + 0.8 * eye_mouth)) / (0.6 * eye_mouth), 0, 1)  # 0 below the brows, 1 well above
forehead = fh > 0.5
keep = 1 - fh * (~ph_skin)
for _ in range(3):  # soften the cut
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, ed[:, 0], keep[ed[:, 1]]); np.add.at(acc, ed[:, 1], keep[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    keep = 0.5 * keep + 0.5 * acc / np.maximum(cnt, 1)
mask = mask * keep
# the painter also put skin on the hair above the forehead (up to the horn roots). Wherever the photo shows hair
# above the brows on the front of the head, paint it the hair colour
hair_col = np.median(painted[(painted.max(1) < 0.3) & (painted[:, 2] >= painted[:, 0] - 0.02) & (z > nz)], 0)
upper_front = (z > nz + 0.9 * eye_mouth) & (z < zmax - 0.02) & (np.abs(x - nx) < 0.13) & (nr[:, 1] < 0.1)
in_photo = (U > 0) & (U < W - 1) & (V > 0) & (V < H - 1) & (pp[V, U, 3] > 0.5)
hairw = (upper_front & in_photo & ~ph_skin & skin).astype(float)  # painted as skin, photo says hair
for _ in range(3):
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, ed[:, 0], hairw[ed[:, 1]]); np.add.at(acc, ed[:, 1], hairw[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    hairw = np.maximum(hairw, 0.5 * hairw + 0.5 * acc / np.maximum(cnt, 1))
hairw *= upper_front
mask = mask * (1 - hairw)
print("forehead skin -> hair: %d vertices, hair colour %s" % ((hairw > 0.5).sum(), np.round(hair_col * 255).astype(int)))
print("forehead: photo kept on %d of %d vertices" % (((keep > 0.5) & forehead & (mask > 0)).sum(), (forehead & (mask > 0)).sum()))
sel = (mask > 0.5) & skin
body_skin = skin & (z < nz - 0.06) & (z > nz - 0.32) & (np.abs(x - nx) < 0.12) & (nr[:, 1] < -0.2)
lin = lambda c: np.power(np.clip(c, 0, 1), 2.2)
ps, bs = lin(photo[sel]), lin(painted[body_skin])
scale = np.clip(bs.std(0) / (ps.std(0) + 1e-4), 0.8, 1.25)  # keep the eyes and lips contrasty
off = bs.mean(0) - ps.mean(0) * scale
print("skin match: %d face / %d body vertices, scale %s offset %s" % (sel.sum(), body_skin.sum(), np.round(scale, 2), np.round(off, 3)))

constraint = (mask > 0) * 1.0
constraint = np.where(np.clip((1.0 - oval) / 0.6, 0, 1) * facing > 0.02, keep * (1 - hairw) * np.where(hairish, 0.0, 1.0), 0.0)
for _ in range(2):
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, ed[:, 0], constraint[ed[:, 1]]); np.add.at(acc, ed[:, 1], constraint[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    constraint = 0.5 * constraint + 0.5 * acc / np.maximum(cnt, 1)
a_ = me.color_attributes.new("face_w", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = constraint[:, None]; a_.data.foreach_set("color", c4.ravel())
b_ = me.color_attributes.new("hair_w", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = hairw[:, None]; b_.data.foreach_set("color", c4.ravel())

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
# per-pixel oval: ((x - nx) / ax)^2 + ((z - fz) / az)^2, then (1 - oval) / 0.6 clamped, ^1.5, times facing
ox = lin(sep.outputs["X"], 1 / (1.45 * eye_mouth), -nx / (1.45 * eye_mouth))
oz = lin(sep.outputs["Z"], 1 / (2.1 * eye_mouth), -fz / (2.1 * eye_mouth))
def m2(a_, b_, op):
    nd = N("ShaderNodeMath", operation=op); L(a_, nd.inputs[0])
    if isinstance(b_, float): nd.inputs[1].default_value = b_
    else: L(b_, nd.inputs[1])
    return nd.outputs[0]
oval_s = m2(m2(ox, ox, "MULTIPLY"), m2(oz, oz, "MULTIPLY"), "ADD")
edge = N("ShaderNodeMapRange"); L(oval_s, edge.inputs["Value"])
edge.inputs["From Min"].default_value = 1.0; edge.inputs["From Max"].default_value = 0.4  # clamped 0..1
edge_p = m2(edge.outputs["Result"], 1.5, "POWER")
nrm = N("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], nrm.inputs[0])
fac = N("ShaderNodeMapRange"); L(nrm.outputs["Y"], fac.inputs["Value"])
fac.inputs["From Min"].default_value = -0.3; fac.inputs["From Max"].default_value = -0.6
fsep = N("ShaderNodeSeparateColor"); L(fw.outputs["Color"], fsep.inputs[0])
wmask = m2(m2(edge_p, fac.outputs["Result"], "MULTIPLY"), fsep.outputs[0], "MULTIPLY")
mx = N("ShaderNodeMix", data_type="RGBA"); L(wmask, mx.inputs["Factor"])
L(tt.outputs["Color"], mx.inputs[6]); L(gn.outputs[0], mx.inputs[7])
# soften the photo's cheek and jaw shadows and its slightly redder cast, over a wider oval on the skin, so the face
# sits in the same light and tone as the body
wide = N("ShaderNodeMapRange"); L(oval_s, wide.inputs["Value"])
wide.inputs["From Min"].default_value = 1.8; wide.inputs["From Max"].default_value = 0.6
lift_w = m2(m2(wide.outputs["Result"], fac.outputs["Result"], "MULTIPLY"), fsep.outputs[0], "MULTIPLY")
# protect the features: brows, lashes, pupils, nostrils and the lip line are the darkest values (linear luma below
# ~0.05) and must keep their contrast; only the mid-dark skin shadows get lifted
bw = N("ShaderNodeRGBToBW"); L(mx.outputs[2], bw.inputs[0])
prot = N("ShaderNodeMapRange"); L(bw.outputs[0], prot.inputs["Value"])
prot.inputs["From Min"].default_value = 0.035; prot.inputs["From Max"].default_value = 0.08
lift_w = m2(lift_w, prot.outputs["Result"], "MULTIPLY")
gam = N("ShaderNodeGamma"); L(mx.outputs[2], gam.inputs["Color"]); gam.inputs["Gamma"].default_value = 0.82
hs = N("ShaderNodeHueSaturation"); L(gam.outputs[0], hs.inputs["Color"])
hs.inputs["Saturation"].default_value = 0.95; hs.inputs["Hue"].default_value = 0.503
lifted = N("ShaderNodeMix", data_type="RGBA"); L(lift_w, lifted.inputs["Factor"])
L(mx.outputs[2], lifted.inputs[6]); L(hs.outputs["Color"], lifted.inputs[7])
import os as _os
# brightness of the face region relative to the body (measured: lifted face 0.54 vs chest 0.42 value)
FG = [float(v) for v in _os.environ.get("FACE_GAIN", "0.60,0.53,0.51").split(",")]
fgain = N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY"); fgain.inputs["Factor"].default_value = 1
L(lifted.outputs[2], fgain.inputs[6]); fgain.inputs[7].default_value = tuple(FG) + (1,)
face_reg = m2(m2(wide.outputs["Result"], fac.outputs["Result"], "MULTIPLY"), fsep.outputs[0], "MULTIPLY")
graded = N("ShaderNodeMix", data_type="RGBA"); L(face_reg, graded.inputs["Factor"])
L(lifted.outputs[2], graded.inputs[6]); L(fgain.outputs[2], graded.inputs[7])
hw_ = N("ShaderNodeVertexColor", layer_name="hair_w")
# painted texture darkened to the hair colour (keeps its shading and strands)
tint = N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY"); tint.inputs["Factor"].default_value = 1
L(tt.outputs["Color"], tint.inputs[6])
ratio = np.clip(np.power(hair_col, 2.2) / (np.power(np.median(painted[skin & upper_front], 0), 2.2) + 1e-4), 0, 1) if (skin & upper_front).any() else np.ones(3)
tint.inputs[7].default_value = tuple(float(v) for v in ratio) + (1,)
mx2 = N("ShaderNodeMix", data_type="RGBA"); L(hw_.outputs["Color"], mx2.inputs["Factor"])
L(graded.outputs[2], mx2.inputs[6]); L(tint.outputs[2], mx2.inputs[7])
em = N("ShaderNodeEmission"); L(mx2.outputs[2], em.inputs[0])
out = N("ShaderNodeOutputMaterial"); L(em.outputs[0], out.inputs[0])
new = bpy.data.images.new("painted_face", max(TW, 4096), max(TH, 4096))
tn = N("ShaderNodeTexImage", image=new); nt.nodes.active = tn
old = list(me.materials); me.materials.clear(); me.materials.append(mat)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 8; sc.cycles.device = "CPU"; sc.render.bake.margin = 8; sc.render.bake.use_clear = False  # keep the magenta marker in unbaked texels
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
new.pixels.foreach_set(np.tile(np.array([1, 0, 1, 1], "f"), new.size[0] * new.size[1]))
bpy.ops.object.bake(type="EMIT")
_w, _h = new.size
_px = np.empty(_w * _h * 4, "f"); new.pixels.foreach_get(_px); _px = _px.reshape(_h, _w, 4)
_hole = (_px[..., 0] > 0.99) & (_px[..., 1] < 0.01) & (_px[..., 2] > 0.99)
def push_pull(img, hole):
    if min(img.shape[:2]) <= 2 or not hole.any():
        return img
    w = (~hole).astype("f")
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    cs = (img[:h2 * 2, :w2 * 2] * w[:h2 * 2, :w2 * 2, None]).reshape(h2, 2, w2, 2, 3).sum((1, 3))
    ws = w[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum((1, 3))
    small = cs / np.maximum(ws, 1e-6)[..., None]
    small = push_pull(small, ws < 1e-6)
    up = np.repeat(np.repeat(small, 2, 0), 2, 1)
    up = np.pad(up, ((0, img.shape[0] - up.shape[0]), (0, img.shape[1] - up.shape[1]), (0, 0)), mode="edge")
    out = img.copy(); out[hole] = up[hole]
    return out
_px[..., :3] = push_pull(_px[..., :3], _hole)
new.pixels.foreach_set(_px.ravel())
print("filled %.1f%% of the texture (gaps between UV islands)" % (100 * _hole.mean()))
new.pack()
me.materials.clear()
for m in old:
    me.materials.append(m)
    for nd in m.node_tree.nodes:
        if nd.type == "TEX_IMAGE" and nd.image is tex:
            nd.image = new
for nm_ in ("face_w", "hair_w"):
    me.color_attributes.remove(me.color_attributes[nm_])
for poly in me.polygons:
    poly.use_smooth = True  # the faceted look on the face came from flat shading
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".glb", ".blend"))
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_image_format="JPEG", export_vertex_color="NONE")
print("DONE")
