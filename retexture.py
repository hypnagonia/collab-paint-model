"""Replace the photo paint with made-up textures per part, keeping the photo only on the face.

Parts come from the 3D vote labels (part_labels.py: hair / skin / clothing / shoes), refined by position and the
photo colour: horns (grey bits on top of the head), crop top vs shorts, hands wrongly voted as clothing, boot soles.
Each part gets a procedural surface (skin mottling, hair strands, horn ridges, sheer mesh top, linen shorts,
leather boots) in colours measured from the reference photo, baked into a new albedo; per-part materials follow.

Usage: blender -b NAME_painted.blend -P retexture.py -- PART_LABELS.npy OUT_DIR
"""
import bpy, colorsys, os, sys
import numpy as np

PARTS, OUT = sys.argv[sys.argv.index("--") + 1:][:2]
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
NAME = obj.name
n = len(me.vertices)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
nr = np.empty(n * 3, "f"); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
lab = np.load(PARTS)  # 0 none, 1 hair, 2 skin, 3 clothing, 4 shoes
x, y, z, ax = co[:, 0], co[:, 1], co[:, 2], np.abs(co[:, 0])
zmax = z.max()

# photo colour per vertex (from the current albedo at the vertex UV)
albedo = next(i for i in bpy.data.images if i.name.endswith("_albedo"))
W, H = albedo.size
px = np.empty(W * H * 4, "f"); albedo.pixels.foreach_get(px); px = px.reshape(H, W, 4)
uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
vuv = np.zeros((n, 2), "f"); vuv[lv] = uv
col = px[np.clip((vuv[:, 1] * H).astype(int), 0, H - 1), np.clip((vuv[:, 0] * W).astype(int), 0, W - 1), :3]
sat = np.array([colorsys.rgb_to_hsv(*c)[1] for c in col])

ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)

# ---------- parts ----------
P = {"skin": 0, "hair": 1, "horn": 2, "top": 3, "shorts": 4, "boots": 5, "sole": 6}
part = np.full(n, P["skin"])
part[lab == 1] = P["hair"]
part[(lab == 3) | (lab == 0)] = P["top"]
part[lab == 4] = P["boots"]
cloth = part == P["top"]
part[cloth & (z < 0.585 * zmax)] = P["shorts"]
part[cloth & (ax > 0.27) & (z < 1.05)] = P["skin"]  # hands voted as clothing
part[cloth & (ax > 0.165) & (z > 1.0)] = P["skin"]  # shoulder/arm bits voted as the top
part[(part == P["skin"]) & (z > 0.435 * zmax) & (z < 0.565 * zmax) & (ax < 0.21)] = P["shorts"]  # skin strips on the hips
# clean straight-ish borders where the garments really end (top hem, waistband, boot shafts)
band = (ax < 0.155) & (z > 0.597 * zmax) & (z < 0.723 * zmax)
part[band & np.isin(part, [P["skin"], P["shorts"]])] = P["top"]
part[(part == P["top"]) & (z > 0.74 * zmax) & (ax < 0.17)] = P["skin"]
part[(ax < 0.21) & (z > 0.42 * zmax) & (z < 0.569 * zmax) & (part == P["skin"])] = P["shorts"]
part[(part == P["shorts"]) & (z > 0.575 * zmax)] = P["skin"]
part[(z < 0.115 * zmax) & (ax < 0.22) & (part == P["skin"])] = P["boots"]
# sides and back of the head are hair, except her left ear tip that sticks out of it
head = (part == P["skin"]) & (z > 0.83 * zmax)
part[head & ((nr[:, 1] > -0.15) | (y > -0.02)) & ~(x > 0.095)] = P["hair"]
# horns: grey (unsaturated) surface on top of the head, grown from its clearly grey core
top_head = (z > 0.88 * zmax) & (ax > 0.06) & (part != P["skin"])
horn = top_head & (sat < 0.07)
src_e = np.concatenate([ed[:, 0], ed[:, 1]]); dst_e = np.concatenate([ed[:, 1], ed[:, 0]])
for _ in range(3):
    grow = np.zeros(n, bool); grow[dst_e[horn[src_e]]] = True
    horn |= grow & top_head & (sat < 0.16)
part[top_head & ~horn & (part == P["horn"])] = P["hair"]
part[horn] = P["horn"]
part[(part == P["boots"]) & (z < 0.03)] = P["sole"]

# small islands of one part take the part around them
for _ in range(2):
    import collections
    adj = [[] for _ in range(n)]
    for a, b in ed:
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool)
    for i0 in range(n):
        if seen[i0]:
            continue
        comp, stack, border = [], [i0], []
        seen[i0] = True
        while stack:
            i = stack.pop(); comp.append(i)
            for j in adj[i]:
                if part[j] != part[i0]:
                    border.append(part[j])
                elif not seen[j]:
                    seen[j] = True; stack.append(j)
        if len(comp) < (40 if part[i0] == P["horn"] else 250) and border:
            part[comp] = collections.Counter(border).most_common(1)[0][0]
print({k: int((part == v).sum()) for k, v in P.items()})

# face: keep the photo (eyes, brows, lips) on the front of the face, fading out toward the hairline and jaw
fz0, fz1 = 0.80 * zmax, 0.925 * zmax
face = (part == P["skin"]) & (z > fz0) & (z < fz1) & (nr[:, 1] < -0.2)
fw = face * np.clip((0.075 - ax) / 0.025, 0, 1) * np.clip((z - fz0) / 0.03, 0, 1) * np.clip((fz1 - z) / 0.02, 0, 1)
fw *= np.clip((-nr[:, 1] - 0.2) / 0.3, 0, 1)
for _ in range(6):  # soften
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, ed[:, 0], fw[ed[:, 1]]); np.add.at(acc, ed[:, 1], fw[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    fw = np.where(part == P["skin"], 0.5 * fw + 0.5 * acc / np.maximum(cnt, 1), 0)
print("face photo vertices: %d" % (fw > 0.5).sum())

old_uv = me.uv_layers.active.name
bake_uv = me.uv_layers.new(name="bake")
me.uv_layers.active = bake_uv
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=0.5, island_margin=0.002)
bpy.ops.object.mode_set(mode="OBJECT")

# skin on the head that the face photo doesn't cover (temples, sides of the cheeks under the hair) is hair,
# apart from her left ear tip
rest_head = (part == P["skin"]) & (z > 0.83 * zmax) & (fw < 0.25) & ~(x > 0.095)
part[rest_head] = P["hair"]
print("head skin given to hair: %d" % rest_head.sum())

# how far to trust the photo at each vertex: its colour should be close to what that part looks like in the photo
# (skin colour on the hair, hair on the cheek, a stray shorts patch on the hand all fail this and get filled)
MED = {}
for name, k in P.items():
    m = (part == k) & (fw < 0.5)
    MED[name] = np.median(col[m], 0) if m.sum() > 50 else col[part == k].mean(0)
dist = np.linalg.norm((col - np.array([MED[nm] for nm in P])[part]) * 255, axis=1)
trust = np.exp(-(dist / 38.0) ** 2)
src_e = np.concatenate([ed[:, 0], ed[:, 1]]); dst_e = np.concatenate([ed[:, 1], ed[:, 0]])
deg = np.bincount(dst_e, minlength=n).astype(float) + 1e-9


def relax(v, it):
    for _ in range(it):
        acc = np.zeros(n); np.add.at(acc, dst_e, v[src_e])
        v = 0.5 * v + 0.5 * acc / deg
    return v


trust = relax(trust, 4)
print("photo kept on %.0f%% of vertices" % (100 * (trust > 0.5).mean()))
print({k: tuple(int(c * 255) for c in v) for k, v in MED.items()})
attrs = [("facew", fw), ("trust", trust)] + [("m_" + nm, relax((part == k).astype(float), 2)) for nm, k in P.items()]
for nm, vals in attrs:
    a = me.color_attributes.new(nm, "FLOAT_COLOR", "POINT")
    c4 = np.ones((n, 4), "f"); c4[:, :3] = np.asarray(vals, "f")[:, None]
    a.data.foreach_set("color", c4.ravel())

# ---------- procedural surfaces ----------
mat = bpy.data.materials.new("retex_bake")
nt = mat.node_tree; nt.nodes.clear()
N = lambda kind, **kw: (lambda nd: ([setattr(nd, k, v) for k, v in kw.items()], nd)[1])(nt.nodes.new(kind))
L = nt.links.new
tc = N("ShaderNodeTexCoord")
lin = lambda c: tuple((v / 255) ** 2.2 for v in c) + (1,)


def attr(name):
    a = N("ShaderNodeAttribute", attribute_name=name)
    return a.outputs["Fac"]


def ramp(fac, stops):
    r = N("ShaderNodeValToRGB")
    cr = r.color_ramp
    for i, (pos, c) in enumerate(stops):
        e = cr.elements[i] if i < 2 else cr.elements.new(pos)
        e.position, e.color = pos, lin(c)
    L(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def noise(scale, detail=4, rough=0.55, distortion=0.0):
    t = N("ShaderNodeTexNoise")
    t.inputs["Scale"].default_value = scale
    t.inputs["Detail"].default_value = detail
    t.inputs["Roughness"].default_value = rough
    t.inputs["Distortion"].default_value = distortion
    L(tc.outputs["Object"], t.inputs["Vector"])
    return t.outputs["Fac"]


def wave(scale, axis, distortion, detail=3, kind="BANDS", profile="SIN"):
    t = N("ShaderNodeTexWave", wave_type=kind, wave_profile=profile)
    if kind == "BANDS":
        t.bands_direction = axis
    t.inputs["Scale"].default_value = scale
    t.inputs["Distortion"].default_value = distortion
    t.inputs["Detail"].default_value = detail
    L(tc.outputs["Object"], t.inputs["Vector"])
    return t.outputs["Fac"]


def mixf(a, b, f=0.5, op="MIX"):
    m = N("ShaderNodeMix", data_type="FLOAT")
    m.inputs["Factor"].default_value = f
    L(a, m.inputs[2]); L(b, m.inputs[3])
    return m.outputs[0]


def mixc(a, b, fac, blend="MIX"):
    m = N("ShaderNodeMix", data_type="RGBA", blend_type=blend)
    if isinstance(fac, float):
        m.inputs["Factor"].default_value = fac
    else:
        L(fac, m.inputs["Factor"])
    L(a, m.inputs[6]); L(b, m.inputs[7])
    return m.outputs[2]


def tone(name, f):
    return tuple(min(255, int(c * 255 * f)) for c in MED[name])


surf = {}
# fills in each part's own photo colour, with a surface pattern so the filled patches aren't flat
surf["skin"] = ramp(mixf(noise(70, 6, 0.6), noise(4, 2), 0.35), [(0.3, tone("skin", 0.9)), (0.5, tone("skin", 1.0)),
                                                                   (0.72, tone("skin", 1.08))])
streak_co = N("ShaderNodeMapping")
L(tc.outputs["Object"], streak_co.inputs["Vector"])
streak_co.inputs["Scale"].default_value = (90, 90, 4)  # stretched along z: strands run down
strand = N("ShaderNodeTexNoise"); strand.inputs["Detail"].default_value = 8; strand.inputs["Scale"].default_value = 1.2
strand.inputs["Distortion"].default_value = 0.4
L(streak_co.outputs[0], strand.inputs["Vector"])
surf["hair"] = ramp(mixf(strand.outputs["Fac"], noise(5, 3, 0.5, 1.0), 0.35),
                    [(0.3, tone("hair", 0.7)), (0.5, tone("hair", 1.0)), (0.75, tone("hair", 1.3))])
surf["horn"] = ramp(mixf(wave(220, "Z", 2, 2, profile="SAW"), noise(30, 5), 0.4),
                    [(0.2, tone("horn", 0.7)), (0.55, tone("horn", 1.0)), (0.85, tone("horn", 1.3))])
surf["top"] = ramp(mixf(mixf(wave(420, "X", 0.4, 0), wave(420, "Z", 0.4, 0), 0.5), noise(40, 3), 0.5),
                   [(0.35, tone("top", 0.85)), (0.65, tone("top", 1.12))])
surf["shorts"] = ramp(mixf(mixf(wave(600, "X", 1.5, 2), wave(600, "Z", 1.5, 2), 0.5), noise(90, 4), 0.55),
                      [(0.3, tone("shorts", 0.85)), (0.55, tone("shorts", 1.0)), (0.8, tone("shorts", 1.12))])
vor = N("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
vor.inputs["Scale"].default_value = 45
L(tc.outputs["Object"], vor.inputs["Vector"])
surf["boots"] = ramp(mixf(vor.outputs["Distance"], noise(25, 5), 0.5),
                     [(0.05, tone("boots", 0.65)), (0.3, tone("boots", 1.0)), (0.7, tone("boots", 1.15))])
surf["sole"] = ramp(noise(120, 3), [(0.3, tone("sole", 0.85)), (0.7, tone("sole", 1.1))])

out = surf["skin"]
for name in P:
    if name != "skin":
        out = mixc(out, surf[name], attr("m_" + name))  # soft part borders
photo = N("ShaderNodeTexImage", image=albedo)
uvn = N("ShaderNodeUVMap", uv_map=old_uv)
L(uvn.outputs["UV"], photo.inputs["Vector"])
out = mixc(out, photo.outputs["Color"], attr("trust"))
out = mixc(out, photo.outputs["Color"], attr("facew"))
em = N("ShaderNodeEmission"); L(out, em.inputs["Color"])
mo = N("ShaderNodeOutputMaterial"); L(em.outputs[0], mo.inputs["Surface"])
new_img = bpy.data.images.new(NAME + "_albedo_retex", W, H)
tgt = N("ShaderNodeTexImage", image=new_img); nt.nodes.active = tgt

me.materials.clear(); me.materials.append(mat)
me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int64))
s = bpy.context.scene
s.render.engine = "CYCLES"; s.cycles.samples = 1; s.cycles.device = "CPU"; s.render.bake.margin = 8
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.object.bake(type="EMIT")
new_img.filepath_raw = os.path.join(OUT, NAME + "_albedo_retex.png"); new_img.file_format = "PNG"; new_img.save()
new_img.pack()

# ---------- per-part materials sharing the new albedo ----------
ROUGH = {"skin": 0.7, "hair": 0.62, "horn": 0.4, "top": 0.75, "shorts": 0.9, "boots": 0.55, "sole": 0.85}
me.materials.clear()
for name in P:
    m = bpy.data.materials.new(name)
    b = m.node_tree.nodes["Principled BSDF"]
    t = m.node_tree.nodes.new("ShaderNodeTexImage"); t.image = new_img
    m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = ROUGH[name]
    me.materials.append(m)
fl = np.array([np.bincount(part[list(p.vertices)], minlength=len(P)).argmax() for p in me.polygons])
me.polygons.foreach_set("material_index", fl)
me.uv_layers.remove(me.uv_layers[old_uv])
me.uv_layers["bake"].name = "UVMap"
me.uv_layers.active = me.uv_layers["UVMap"]
me.uv_layers["UVMap"].active_render = True
for a, _ in attrs:
    me.color_attributes.remove(me.color_attributes[a])
albedo.user_remap(new_img)
albedo.name = NAME + "_albedo_photo"; new_img.name = NAME + "_albedo"
np.save(os.path.join(OUT, NAME + "_part_ids.npy"), part)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_textured.blend"))
print("DONE")
