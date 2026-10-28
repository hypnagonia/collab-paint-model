"""Part-aware repaint of a photo-painted model: inside each part keep the photo where it looks like that part,
fill the rest with a pattern in that part's own photo colour, with soft borders between parts.

Works per mesh object: parts come from the object's material slots (hand-made part materials) or from a 3D vote
label file (part_labels.py: 1 hair, 2 skin, 3 clothing) for a head mesh, where horns are found by their grey colour
and skin is kept only on the face, neck and ears. Bakes a new albedo on fresh non-overlapping UVs and swaps it in.

Usage: blender -b MODEL.blend -P retex_hybrid.py -- OUT.blend OBJ=materials [OBJ=labels.npy ...]
"""
import bpy, colorsys, os, sys
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
JOBS = [a.split("=", 1) for a in argv[1:]]
scene = bpy.context.scene
scene.render.engine = "CYCLES"; scene.cycles.samples = 1; scene.cycles.device = "CPU"; scene.render.bake.margin = 8


def mesh_arrays(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.array([obj.matrix_world @ v.co for v in me.vertices])
    nr = np.empty(n * 3, "f"); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
    nr = nr @ np.array(obj.matrix_world.to_3x3()).T
    nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
    ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
    return co, nr, ed


def mat_image(m):
    for nd in m.node_tree.nodes:
        if nd.type == "TEX_IMAGE" and nd.image:
            return nd.image


def vertex_photo(obj, img):
    me = obj.data
    W, H = img.size
    px = np.empty(W * H * 4, "f"); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)
    uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
    vuv = np.zeros((len(me.vertices), 2), "f"); vuv[lv] = uv
    return px[np.clip((vuv[:, 1] * H).astype(int), 0, H - 1), np.clip((vuv[:, 0] * W).astype(int), 0, W - 1), :3]


for obj_name, src in JOBS:
    obj = bpy.data.objects[obj_name]
    me = obj.data
    n = len(me.vertices)
    co, nr, ed = mesh_arrays(obj)
    x, y, z = co.T
    src_e = np.concatenate([ed[:, 0], ed[:, 1]]); dst_e = np.concatenate([ed[:, 1], ed[:, 0]])
    deg = np.bincount(dst_e, minlength=n).astype(float) + 1e-9

    def relax(v, it, mask=None):
        for _ in range(it):
            acc = np.zeros(n); np.add.at(acc, dst_e, v[src_e])
            nv = 0.5 * v + 0.5 * acc / deg
            v = nv if mask is None else np.where(mask, nv, v)
        return v

    imgs = [mat_image(m) for m in me.materials]  # one photo albedo per material (the hair may have a tinted copy)
    img = imgs[0]
    col = vertex_photo(obj, img)
    sat = np.array([colorsys.rgb_to_hsv(*c)[1] for c in col])
    facew = np.zeros(n)

    if src == "materials":
        names = [m.name for m in me.materials]
        part = np.zeros(n, int)
        mi = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("material_index", mi)
        F = np.empty(len(me.polygons) * 3, np.int64); me.polygons.foreach_get("vertices", F); F = F.reshape(-1, 3)
        for c in range(3):
            part[F[:, c]] = mi
        for k, im in enumerate(imgs):
            if im is not img and (part == k).any():
                col[part == k] = vertex_photo(obj, im)[part == k]
        if "hair" in names:
            # specks of hair on the skin (purple dots on the arm): small hair islands become skin
            hi, ki = names.index("hair"), names.index("skin")
            hv = part == hi
            adj_ok = hv[src_e] & hv[dst_e]
            comp = np.arange(n)
            for _ in range(200):  # label propagation of the smallest index through hair-hair edges
                new_c = comp.copy()
                np.minimum.at(new_c, dst_e[adj_ok], comp[src_e[adj_ok]])
                if (new_c == comp).all():
                    break
                comp = new_c
            ids, cnt = np.unique(comp[hv], return_counts=True)
            small = np.isin(comp, ids[cnt < 150]) & hv
            part[small] = ki
            flip = (mi == hi) & np.all(part[F] == ki, axis=1)
            mi[flip] = ki
            me.polygons.foreach_set("material_index", mi)
            print("hair specks -> skin: %d vertices in %d islands" % (small.sum(), (cnt < 150).sum()))
        if "gloves" in names and "skirt_on_body" in names:
            # the forearms were fused to the hips: bracer-coloured patches on the hip are hip, not bracer
            gi, ki, ui = names.index("gloves"), names.index("skin"), names.index("skirt_on_body")
            ax_ = np.abs(x)
            t = np.clip((1.18 - z) / (1.18 - 0.95), 0, 1)  # forearm axis: elbow (0.215, 1.18) -> wrist (0.285, 0.95)
            arm_x = 0.215 + t * 0.07
            off_arm = (part == gi) & ((ax_ < arm_x - 0.045) | (z > 1.22))
            part[off_arm] = np.where(z[off_arm] < 1.05, ui, ki)
            # thighs under the sheer free sheet are skin (the sheet over them does the tinting); the wrap at the
            # waist stays cloth
            thigh = (part == ui) & (z < 0.86)
            part[thigh] = ki
            for k in (ui, ki):
                flip = np.isin(mi, [gi, ui]) & np.all(part[F] == k, axis=1)
                mi[flip] = k
            me.polygons.foreach_set("material_index", mi)
            print("bracer patches on the hip: %d, thigh skin under the sheet: %d" % (off_arm.sum(), thigh.sum()))
        if "hair" in names and "top" in names:
            # stray hair faces on the chest (a purple dot on the bra) belong to the top
            # stray hair faces on the chest: dark ones are the bra, light ones are skin
            ti, si, hi = names.index("top"), names.index("skin"), names.index("hair")
            stray = (part == hi) & (z < 1.32) & (np.abs(x) < 0.16)
            med_t0 = np.median(col[part == ti], 0); med_s0 = np.median(col[(part == si) & (z > 1.0) & (z < 1.3)], 0)
            darkish = np.linalg.norm(col - med_t0, axis=1) < np.linalg.norm(col - med_s0, axis=1)
            part[stray & darkish] = ti
            part[stray & ~darkish] = si
            for k in (ti, si):
                flip = (mi == hi) & np.all(part[F] == k, axis=1)
                mi[flip] = k
            me.polygons.foreach_set("material_index", mi)
            print("stray chest hair -> top: %d vertices" % stray.sum())
            # the hand-made top reaches up over the upper chest; there, whatever looks like skin in the photo is skin
            # (the dark straps keep looking like the top and stay)
            ti, si = names.index("top"), names.index("skin")
            med_t = np.median(col[(part == ti) & (z < 1.33)], 0)
            med_s = np.median(col[(part == si) & (z > 1.0) & (z < 1.3)], 0)
            up = (part == ti) & (z > 1.33)
            skinlike = np.linalg.norm(col - med_s, axis=1) < np.linalg.norm(col - med_t, axis=1)
            skinlike = relax(skinlike.astype(float), 3) > 0.5
            part[up & skinlike] = si
            flip = (mi == ti) & np.all(part[F] == si, axis=1)
            mi[flip] = si
            me.polygons.foreach_set("material_index", mi)
            print("upper-chest top -> skin: %d vertices" % (up & skinlike).sum())
    elif src.startswith("body:"):
        # a whole figure from 3D vote labels: clothing splits into the top and the skirt at the bare midriff
        lab = np.load(src[5:])
        names = ["skin", "hair", "top", "skirt"]
        imgs = [img] * 4
        zmin, zmax = z.min(), z.max()
        h = zmax - zmin
        part = np.where(lab == 1, 1, np.where(np.isin(lab, (0, 3)), 3, 0))
        cl = part == 3
        zc = z[cl]
        hist, edges = np.histogram(zc, bins=60, range=(zmin + 0.4 * h, zmin + 0.8 * h))
        gap = edges[np.argmin(hist[10:50]) + 10]  # emptiest height between skirt and top: the midriff
        part[cl & (z > gap)] = 2
        print("top/skirt split at z=%.3f" % gap)
        # the bare midriff between them: cloth there (a strip down the flank) is skin
        top_lo = np.percentile(z[part == 2], 2)
        skirt_hi = np.percentile(z[part == 3], 98)
        mid = (z > skirt_hi - 0.01) & (z < top_lo + 0.01) & np.isin(part, (2, 3))
        part[mid] = 0
        part[(part == 2) & (z < gap + 0.03) & (np.abs(x) > 0.15)] = 0
        part[(part == 3) & (z > gap - 0.03) & (np.abs(x) > 0.15)] = 0
        print("midriff cloth -> skin: %d" % mid.sum())
        # hair only on the head; skin there only on the face front, neck and ears
        head = z > zmin + 0.84 * h
        part[(part == 1) & ~head & (z < zmin + 0.6 * h)] = 0
        face = head & (part == 0) & (nr[:, 1] < -0.3)
        facew = relax(face.astype(float), 6) * (part == 0)
    else:
        lab = np.load(src)
        names = ["skin", "hair", "horn"]
        imgs = [img] * 3
        part = np.where(lab == 2, 0, 1)
        zmin, zmax = z.min(), z.max()
        h = zmax - zmin
        # horns: grey surface high on the head, grown from the clearly grey core
        top = z > zmin + 0.62 * h
        horn = top & (sat < 0.08) & (lab != 2)
        for _ in range(4):
            grow = np.zeros(n, bool); grow[dst_e[horn[src_e]]] = True
            horn |= grow & top & (sat < 0.16) & (lab != 2)
        part[horn] = 2
        # skin only where it can be: the front of the face, the neck and shoulders below the jaw, the ears
        face_front = nr[:, 1] < -0.15
        low = z < zmin + 0.25 * h
        part[(part == 0) & ~face_front & ~low] = 1
        # and no wider than the face: locks hanging beside the cheeks got face skin projected onto them
        xc = np.median(x[(lab == 2) & face_front & (z > zmin + 0.4 * h)])
        # her right side has a lock hanging in front of the cheek: keep the face narrow there; on her left the hair is
        # tucked behind the ear, so the cheek reaches the ear wherever the photo shows skin
        med_skin = np.median(col[(lab == 2) & face_front], 0); med_hair = np.median(col[lab == 1], 0)
        skin_col = np.linalg.norm(col - med_skin, axis=1) < np.linalg.norm(col - med_hair, axis=1)
        dx = x - xc
        part[(part == 0) & (dx < -0.062) & ~low] = 1
        part[(part == 0) & (dx > 0.095) & ~low] = 1
        cheek_l = (dx > 0.03) & (dx < 0.095) & skin_col & (nr[:, 1] < 0.3) & ~low & (z < zmin + 0.62 * h)
        part[cheek_l & (part == 1) & (lab == 2)] = 0
        # the hairline: skin can't reach above the brows at the temples
        brow = zmin + 0.62 * h
        part[(part == 0) & (z > brow) & (np.abs(x - xc) > 0.05)] = 1
        # the face photo is kept as is (eyes, brows, lips differ from the average skin colour)
        facew = ((part == 0) & (nr[:, 1] < -0.3) & (z > zmin + 0.28 * h)).astype(float)
        facew = relax(facew, 6) * (part == 0)
    print(obj_name, {nm: int((part == k).sum()) for k, nm in enumerate(names)})

    MED = {}
    for k, nm in enumerate(names):
        m = (part == k) & (facew < 0.5)
        MED[nm] = np.median(col[m], 0) if m.sum() > 50 else (col[part == k].mean(0) if (part == k).any() else np.full(3, 0.5))
    dist = np.linalg.norm((col - np.array([MED[nm] for nm in names])[part]) * 255, axis=1)
    trust = relax(np.exp(-(dist / 38.0) ** 2), 4)
    if src != "materials" and not src.startswith("body:"):
        head_skin = np.median(col[part == 0], 0)  # the whole face, not just the leftover neck
        head_hair = MED["hair"]
    print(obj_name, "photo kept on %.0f%%" % (100 * (trust > 0.5).mean()),
          {k: tuple(int(c * 255) for c in v) for k, v in MED.items()})

    # fresh UVs for the bake target; the photo is still read through the old ones
    old_uv = me.uv_layers.active.name
    me.uv_layers.active = me.uv_layers.new(name="bake")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=0.5, island_margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")

    attrs = [("facew", facew), ("trust", trust)] + [("m_%d" % k, relax((part == k).astype(float), 2)) for k in range(len(names))]
    for nm, vals in attrs:
        a = me.color_attributes.new(nm, "FLOAT_COLOR", "POINT")
        c4 = np.ones((n, 4), "f"); c4[:, :3] = np.asarray(vals, "f")[:, None]
        a.data.foreach_set("color", c4.ravel())

    mat = bpy.data.materials.new("retex_" + obj_name)
    nt = mat.node_tree; nt.nodes.clear()

    def N(kind, **kw):
        nd = nt.nodes.new(kind)
        for k, v in kw.items():
            setattr(nd, k, v)
        return nd

    L = nt.links.new
    tc = N("ShaderNodeTexCoord")
    lin = lambda c: tuple((v / 255) ** 2.2 for v in c) + (1,)
    attr = lambda nm: N("ShaderNodeAttribute", attribute_name=nm).outputs["Fac"]

    def ramp(fac, stops):
        r = N("ShaderNodeValToRGB")
        for i, (pos, c) in enumerate(stops):
            e = r.color_ramp.elements[i] if i < 2 else r.color_ramp.elements.new(pos)
            e.position, e.color = pos, lin(c)
        L(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def noise(scale, detail=4, vec=None):
        t = N("ShaderNodeTexNoise")
        t.inputs["Scale"].default_value = scale; t.inputs["Detail"].default_value = detail
        L(vec if vec is not None else tc.outputs["Object"], t.inputs["Vector"])
        return t.outputs["Fac"]

    def mixc(a, b, fac):
        m = N("ShaderNodeMix", data_type="RGBA")
        L(fac, m.inputs["Factor"]); L(a, m.inputs[6]); L(b, m.inputs[7])
        return m.outputs[2]

    def tone(nm, f):
        return tuple(min(255, int(c * 255 * f)) for c in MED[nm])

    def surface(nm):
        if nm == "hair":
            mp = N("ShaderNodeMapping"); L(tc.outputs["Object"], mp.inputs["Vector"])
            mp.inputs["Scale"].default_value = (90, 90, 4)  # strands run down
            return ramp(noise(1.2, 8, mp.outputs[0]), [(0.3, tone(nm, 0.7)), (0.5, tone(nm, 1.0)), (0.75, tone(nm, 1.3))])
        if nm == "horn":
            w = N("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z", wave_profile="SAW")
            w.inputs["Scale"].default_value = 220; w.inputs["Distortion"].default_value = 2
            L(tc.outputs["Object"], w.inputs["Vector"])
            return ramp(w.outputs["Fac"], [(0.2, tone(nm, 0.7)), (0.6, tone(nm, 1.0)), (0.9, tone(nm, 1.3))])
        if nm in ("boots", "gloves"):
            v = N("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE"); v.inputs["Scale"].default_value = 45
            L(tc.outputs["Object"], v.inputs["Vector"])
            return ramp(v.outputs["Distance"], [(0.05, tone(nm, 0.7)), (0.3, tone(nm, 1.0)), (0.7, tone(nm, 1.12))])
        return ramp(noise(60, 6), [(0.3, tone(nm, 0.9)), (0.5, tone(nm, 1.0)), (0.72, tone(nm, 1.08))])

    out = surface(names[0])
    for k in range(1, len(names)):
        if (part == k).any():
            out = mixc(out, surface(names[k]), attr("m_%d" % k))
    uvn = N("ShaderNodeUVMap", uv_map=old_uv)
    photo = None
    for k, im in enumerate(imgs):
        t = N("ShaderNodeTexImage", image=im); L(uvn.outputs["UV"], t.inputs["Vector"])
        photo = t.outputs["Color"] if photo is None else (photo if im is img else mixc(photo, t.outputs["Color"], attr("m_%d" % k)))
    if src == "materials" and "skin" in names and "head_skin" in globals():
        gain = head_skin / MED["skin"]  # the body photo is redder and darker than the face photo: match the face
        g = N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        g.inputs["Factor"].default_value = 1.0
        L(photo, g.inputs[6]); g.inputs[7].default_value = tuple(float(v) ** 2.2 for v in gain) + (1,)
        photo = mixc(photo, g.outputs[2], attr("m_%d" % names.index("skin")))
        print("body skin gain", np.round(gain, 2))
    if src == "materials" and "hair" in names and "head_hair" in globals() and (part == names.index("hair")).any():
        gain = head_hair / MED["hair"]
        g = N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        g.inputs["Factor"].default_value = 1.0
        L(photo, g.inputs[6]); g.inputs[7].default_value = tuple(float(v) ** 2.2 for v in gain) + (1,)
        photo = mixc(photo, g.outputs[2], attr("m_%d" % names.index("hair")))
        MED["hair"] = head_hair
        print("body hair gain", np.round(gain, 2))
    out = mixc(out, photo, attr("trust"))
    out = mixc(out, photo, attr("facew"))
    em = N("ShaderNodeEmission"); L(out, em.inputs["Color"])
    mo = N("ShaderNodeOutputMaterial"); L(em.outputs[0], mo.inputs["Surface"])
    W, H = img.size
    new_img = bpy.data.images.new(obj_name + "_albedo_hybrid", W, H)
    tgt = N("ShaderNodeTexImage", image=new_img); nt.nodes.active = tgt

    old_mats = list(me.materials)
    saved = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("material_index", saved)
    me.materials.clear(); me.materials.append(mat)
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int64))
    bpy.ops.object.bake(type="EMIT")
    new_img.filepath_raw = os.path.join(os.path.dirname(OUT), obj_name + "_albedo_hybrid.png")
    new_img.file_format = "PNG"; new_img.save(); new_img.pack()
    me.materials.clear()
    for m in old_mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", saved)
    for m in old_mats:  # point this object's materials at the new albedo
        for nd in m.node_tree.nodes:
            if nd.type == "TEX_IMAGE" and nd.image in imgs:
                nd.image = new_img
    me.uv_layers.remove(me.uv_layers[old_uv])
    me.uv_layers["bake"].name = "UVMap"
    me.uv_layers["UVMap"].active_render = True
    for a, _ in attrs:
        me.color_attributes.remove(me.color_attributes[a])
    np.save(os.path.join(os.path.dirname(OUT), obj_name + "_hybrid_parts.npy"), part)

for m in bpy.data.materials:  # the hair read as plastic: matte it down
    if m.name.startswith(("hair", "succubus2_head")) and m.node_tree and "Principled BSDF" in m.node_tree.nodes:
        r = m.node_tree.nodes["Principled BSDF"].inputs["Roughness"]
        r.default_value = max(r.default_value, 0.62)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("DONE")
