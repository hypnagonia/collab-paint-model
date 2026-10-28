"""Apply brush marks from the Paint Marks page to a photo-painted mesh.

Each mark says what the brushed spot should be (a part, or an exact colour). Marked vertices get re-coloured:
a part's colour grows in from nearby unmarked vertices of that same part (falling back to the part's median),
horns take a fixed horn colour, colour marks take their colour. The albedo is re-baked with the marked areas
replaced, and the part ids are updated so materials follow.

Usage: blender -b NAME_painted.blend -P apply_marks.py -- MARKS_DIR PART_IDS.npy OUT_DIR
"""
import bpy, glob, json, os, sys
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
MARKS, PIDS, OUT = argv[:3]

obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
NAME = obj.name
n = len(me.vertices)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
part = np.load(PIDS)  # paint_parts ids
IDS = {"skin": 0, "hair": 1, "horn": 2, "top": 3, "skirt": 4, "gloves": 5, "boots": 6}
HORN_SRGB = (0.22, 0.24, 0.30)

# ---------- current colour per vertex, sampled from the baked albedo at the vertex's UV ----------
albedo = next(i for i in bpy.data.images if i.name.endswith("_albedo"))
W, H = albedo.size
px = np.empty(W * H * 4, "f"); albedo.pixels.foreach_get(px); px = px.reshape(H, W, 4)
uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
vuv = np.zeros((n, 2), "f"); vuv[lv] = uv
col = px[np.clip((vuv[:, 1] * H).astype(int), 0, H - 1), np.clip((vuv[:, 0] * W).astype(int), 0, W - 1), :3]

# ---------- marks -> per-vertex target ----------
target = np.full(n, -1)  # part id the vertex should become
fixed = np.full((n, 3), np.nan, "f")  # exact colour (sRGB as stored in the image)
marks = [json.load(open(f)) for f in glob.glob(os.path.join(MARKS, "*.json"))]
marks.sort(key=lambda m: m.get("at", 0))
for m in marks:
    pts = np.array(m.get("points") or [], "f")
    if not len(pts):
        continue
    r = float(m.get("radius") or 0.03)
    d = np.min(np.linalg.norm(co[:, None, :] - pts[None, :, :], axis=2), axis=1) if len(pts) < 200 else \
        np.min(np.stack([np.linalg.norm(co - p, axis=1) for p in pts]), axis=0)
    hit = d < r
    k = m.get("kind")
    if k == "color" and m.get("color"):
        h = m["color"].lstrip("#")
        fixed[hit] = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    elif k in IDS:
        target[hit] = IDS[k]
        if k == "horn":
            fixed[hit] = HORN_SRGB
    print("mark %-6s %4d pts -> %5d vertices" % (k, len(pts), hit.sum()))

marked = (target >= 0) | ~np.isnan(fixed[:, 0])
part = part.copy()
part[target >= 0] = target[target >= 0]

# ---------- grow each part's colour into its marked vertices ----------
ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
src = np.concatenate([ed[:, 0], ed[:, 1]]); dst = np.concatenate([ed[:, 1], ed[:, 0]])
same = part[src] == part[dst]
src, dst = src[same], dst[same]
new = col.copy()
has = ~marked
for _ in range(300):
    ok = has[src] & ~has[dst]
    if not ok.any():
        break
    acc = np.zeros_like(new); cnt = np.zeros(n)
    np.add.at(acc, dst[ok], new[src[ok]]); np.add.at(cnt, dst[ok], 1)
    grow = cnt > 0
    new[grow] = acc[grow] / cnt[grow, None]
    has |= grow
for p in np.unique(part[~has]):
    pool = (part == p) & ~marked
    new[~has & (part == p)] = np.median(col[pool], 0) if pool.any() else col[~has & (part == p)].mean(0)
# relax the regrown patch so it has no streaks
for _ in range(20):
    acc = np.zeros_like(new); cnt = np.zeros(n)
    np.add.at(acc, dst, new[src]); np.add.at(cnt, dst, 1)
    upd = marked & (cnt > 0)
    new[upd] = 0.5 * new[upd] + 0.5 * acc[upd] / cnt[upd, None]
fx = ~np.isnan(fixed[:, 0])
rng = np.random.default_rng(0)
new[fx] = np.clip(fixed[fx] * (1 + 0.06 * rng.standard_normal((fx.sum(), 1))), 0, 1)
print("re-coloured %d vertices" % marked.sum())

# ---------- re-bake: old texture outside the marks, new colours inside ----------
cattr = me.color_attributes.new("mark_col", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = new ** 2.2  # image values are sRGB; vertex colours are linear
cattr.data.foreach_set("color", c4.ravel())
mattr = me.color_attributes.new("mark_mask", "FLOAT_COLOR", "POINT")
m4 = np.ones((n, 4), "f"); m4[:, :3] = marked[:, None].astype("f")
mattr.data.foreach_set("color", m4.ravel())

bake_mat = bpy.data.materials.new("bake_marks")
nt = bake_mat.node_tree; nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = albedo
vc = nt.nodes.new("ShaderNodeVertexColor"); vc.layer_name = "mark_col"
vm = nt.nodes.new("ShaderNodeVertexColor"); vm.layer_name = "mark_mask"
mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
nt.links.new(vm.outputs["Color"], mix.inputs["Factor"])
nt.links.new(tex.outputs["Color"], mix.inputs[6]); nt.links.new(vc.outputs["Color"], mix.inputs[7])
em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(mix.outputs[2], em.inputs["Color"])
out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out.inputs["Surface"])
new_img = bpy.data.images.new(NAME + "_albedo_marked", W, H)
tgt = nt.nodes.new("ShaderNodeTexImage"); tgt.image = new_img; nt.nodes.active = tgt
old_mats = list(me.materials)
me.materials.clear(); me.materials.append(bake_mat)
saved_idx = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("material_index", saved_idx)
me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int64))
s = bpy.context.scene
s.render.engine = "CYCLES"; s.cycles.samples = 1; s.cycles.device = "CPU"; s.render.bake.margin = 8
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.object.bake(type="EMIT")
new_img.filepath_raw = os.path.join(OUT, NAME + "_albedo.png"); new_img.file_format = "PNG"; new_img.save()
# swap the new image in for the old one everywhere and restore the materials
me.materials.clear()
for m in old_mats:
    me.materials.append(m)
me.polygons.foreach_set("material_index", saved_idx)
albedo.user_remap(new_img)
albedo.name = NAME + "_albedo_old"
new_img.name = NAME + "_albedo"
new_img.pack()
for a in ("mark_col", "mark_mask"):
    me.color_attributes.remove(me.color_attributes[a])
np.save(os.path.join(OUT, NAME + "_part_ids.npy"), part)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_painted.blend"))
print("DONE")
