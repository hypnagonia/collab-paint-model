"""Paint the generated mesh by projecting the front/side references onto it, then bake to one UV texture.

Front reference projects along -Y, side reference along -X and +X (the figure is close to symmetric).
Per-vertex ray casts mark which views actually see each vertex so occluded parts don't pick up colour
from whatever is in front of them. Back-facing areas have no reference and get cloak cloth / boot leather.
"""
import bpy, bmesh, math, os, sys, time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
BLEND, FRONT, SIDE, OUT = argv[:4]
TEX = int(argv[4]) if len(argv) > 4 else 4096
BACK = argv[5] if len(argv) > 5 and argv[5] != '-' else None  # optional back view ('-' = none)
PARTS = argv[6] if len(argv) > 6 and argv[6] != '-' else None  # optional per-vertex part labels (part_labels.py)
LEFT = argv[7] if len(argv) > 7 else None  # optional real left view; without it the side view is mirrored

bpy.ops.wm.open_mainfile(filepath=BLEND)
obj = next(o for o in bpy.data.objects if o.type == "MESH")
NAME = obj.name
me = obj.data
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

n = len(me.vertices)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
nr = np.empty(n * 3, "f"); me.vertex_normals.foreach_get("vector", nr); nr = nr.reshape(-1, 3)
(xmin, ymin, zmin), (xmax, ymax, zmax) = co.min(0), co.max(0)
print("bounds", xmin, xmax, ymin, ymax, zmin, zmax, "aspect front", (zmax - zmin) / (xmax - xmin),
      "side", (zmax - zmin) / (ymax - ymin))

# ---------- visibility per view ----------
t = time.time()
bm = bmesh.new(); bm.from_mesh(me)
bvh = BVHTree.FromBMesh(bm)
dirs = [Vector((0, -1, 0)), Vector((-1, 0, 0)), Vector((1, 0, 0)), Vector((0, 1, 0))]  # front, right, left, back
vis = np.zeros((n, 4), "f")
for i in range(n):
    p, nn = Vector(co[i]), Vector(nr[i])
    for k, d in enumerate(dirs):
        if nn.dot(d) <= 0.05:
            continue
        hit = bvh.ray_cast(p + nn * 0.002 + d * 0.002, d)
        if hit[0] is None:
            vis[i, k] = 1.0
bm.free()
print("visibility in %.0fs" % (time.time() - t), vis.mean(0))
attr = me.color_attributes.new("vis", "FLOAT_COLOR", "POINT")
me.color_attributes["vis"].data.foreach_set("color", vis.ravel())  # re-fetch: edit mode invalidated the old handle

# ---------- UVs ----------
t = time.time()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0015)
bpy.ops.object.mode_set(mode="OBJECT")
print("uv in %.0fs" % (time.time() - t))

# ---------- projection shader ----------
mat = bpy.data.materials.new("projection")
obj.data.materials.clear()
obj.data.materials.append(mat)
nt = mat.node_tree
nt.nodes.clear()
L = nt.links.new


def node(kind, **props):
    nd = nt.nodes.new(kind)
    for k, v in props.items():
        setattr(nd, k, v)
    return nd


def math_(op, a, b=None):
    nd = node("ShaderNodeMath", operation=op)
    for sock, val in zip(nd.inputs, (a, b)):
        if val is None:
            continue
        if isinstance(val, (int, float)):
            sock.default_value = val
        else:
            L(val, sock)
    return nd.outputs[0]


def vmath(op, a, b=None, scale=None):
    nd = node("ShaderNodeVectorMath", operation=op)
    for sock, val in zip(nd.inputs, (a, b)):
        if val is not None:
            if isinstance(val, (tuple, list)):
                sock.default_value = val
            else:
                L(val, sock)
    if scale is not None:
        L(scale, nd.inputs["Scale"]) if not isinstance(scale, float) else setattr(nd.inputs["Scale"], "default_value", scale)
    return nd.outputs[0]


def maprange(val, a, b):
    nd = node("ShaderNodeMapRange", clamp=False)
    L(val, nd.inputs["Value"])
    nd.inputs["From Min"].default_value = a
    nd.inputs["From Max"].default_value = b
    return nd.outputs["Result"]


def projected(path, u, v):
    img = bpy.data.images.load(path)
    tex = node("ShaderNodeTexImage", image=img, extension="CLIP", interpolation="Cubic")
    uv = node("ShaderNodeCombineXYZ")
    L(u, uv.inputs[0]); L(v, uv.inputs[1])
    L(uv.outputs[0], tex.inputs["Vector"])
    return tex.outputs["Color"], tex.outputs["Alpha"]


def load_alpha(path):
    im = bpy.data.images.load(path, check_existing=True)
    w, h = im.size
    px = np.empty(w * h * 4, "f")
    im.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[..., 3] > 0.5, w, h  # row 0 is the bottom


def silhouette_cells(a, zc, cell):
    """Cell centres covered by the mesh's orthographic silhouette on the (a, z) plane."""
    a0, z0 = a.min(), zc.min()
    g = np.zeros((int((zc.max() - z0) / cell) + 5, int((a.max() - a0) / cell) + 5), bool)
    g[((zc - z0) / cell + 2).astype(int), ((a - a0) / cell + 2).astype(int)] = True
    for _ in range(2):  # close the gaps between splatted vertices
        g = g | np.roll(g, 1, 0) | np.roll(g, -1, 0) | np.roll(g, 1, 1) | np.roll(g, -1, 1)
    g = g & np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1)
    gz, ga = np.nonzero(g)
    return a0 + (ga - 2 + 0.5) * cell, z0 + (gz - 2 + 0.5) * cell


def fit_view(a, zc, path, tag, band=None, init=None):
    """Find the scale/offset that best overlays the reference silhouette on the mesh silhouette (IoU).
    With `band` only the part above that height is matched (used for the head).
    Returns the mesh-space (a, z) ranges that the image spans."""
    A, w, h = load_alpha(path)
    rows = np.concatenate([[0], np.cumsum(A.sum(1))])  # fg pixels in rows [0, v)
    ac, zm = (a.max() + a.min()) / 2, (zc.max() + zc.min()) / 2
    p0 = h / (zc.max() - zc.min())

    def cells(cell):
        ca, cz = silhouette_cells(a, zc, cell)
        return (ca, cz) if band is None else (ca[cz > band], cz[cz > band])

    def score(ca, cz, cell, s, du, dv):
        p = p0 * s
        u = ((ca - ac) * p + w / 2 + du).astype(int)
        v = ((cz - zm) * p + h / 2 + dv).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        k = (cell * p) ** 2
        inter = A[v[ok], u[ok]].sum() * k
        vcut = 0 if band is None else int(np.clip((band - zm) * p + h / 2 + dv, 0, h))
        fg = rows[-1] - rows[vcut]
        return inter / (len(ca) * k + fg - inter)

    best = (-1, 1.0, 0, 0)
    if init is None:
        grid = (np.linspace(0.85, 1.15, 13), range(-30, 31, 3), range(-30, 31, 3))
    else:
        grid = (np.linspace(init[0] - 0.1, init[0] + 0.1, 11), range(init[1] - 24, init[1] + 25, 3),
                range(init[2] - 24, init[2] + 25, 3))
    ca, cz = cells(0.008)
    for s in grid[0]:
        for du in grid[1]:
            for dv in grid[2]:
                best = max(best, (score(ca, cz, 0.008, s, du, dv), s, du, dv))
    _, s0, du0, dv0 = best
    ca, cz = cells(0.004)
    for s in np.arange(s0 - 0.025, s0 + 0.026, 0.005):
        for du in range(du0 - 3, du0 + 4):
            for dv in range(dv0 - 3, dv0 + 4):
                best = max(best, (score(ca, cz, 0.004, s, du, dv), s, du, dv))
    iou, s, du, dv = best
    p = p0 * s
    a_lo = ac - (w / 2 + du) / p
    z_lo = zm - (h / 2 + dv) / p
    print("fit %s: IoU %.3f scale %.3f shift %d,%d px" % (tag, iou, s, du, dv))
    return a_lo, a_lo + w / p, z_lo, z_lo + h / p, (s, du, dv)


geo = node("ShaderNodeNewGeometry")
pos = node("ShaderNodeSeparateXYZ"); L(geo.outputs["Position"], pos.inputs[0])
nrm = node("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], nrm.inputs[0])
x, y, z = pos.outputs
nx, ny, nz = nrm.outputs
vz = maprange(z, zmin, zmax)

BUST = os.environ.get("PAINT_BUST") == "1"  # a head/bust, not a full figure: skip the body-specific rules
# optional vertical warp of the front photo: PAINT_ZWARP_FRONT="meshz:photoz,..." pins facial features
# (eye line, nostrils, lip line) read off a close-up render to where the mesh has them
ZW = sorted(tuple(float(v) for v in pair.split(":")) for pair in os.environ.get("PAINT_ZWARP_FRONT", "").split(",") if pair)


def zwarp(zarr):
    return np.interp(zarr, [a for a, _ in ZW], [b for _, b in ZW]) if ZW else zarr


HEAD = 99.0 if BUST else 1.5  # above this the head-only fit takes over (hood + face need tighter registration than the cloak)
head_mix = node("ShaderNodeMapRange")
L(z, head_mix.inputs["Value"])
head_mix.inputs["From Min"].default_value = HEAD - 0.08
head_mix.inputs["From Max"].default_value = HEAD


def nose_pin(fit, path, a_mesh):
    """Shift a profile fit so the photo's nose tip (front-most pixel around nose height) lands on the
    mesh's (front-most vertex there). Silhouette fits on a bust are dominated by hair and drift."""
    A, w, h = load_alpha(path)  # row 0 = bottom
    a_lo, a_hi, z_lo, z_hi = fit[:4]
    zc = co[:, 2]
    zr = zc.max() - zc.min()
    band = (zc > zc.min() + 0.42 * zr) & (zc < zc.min() + 0.56 * zr)
    i = np.argmax(np.where(band, a_mesh, -np.inf))
    a_n, z_n = a_mesh[i], zc[i]
    rows = np.arange(int(0.42 * h), int(0.56 * h))
    cols = [np.nonzero(A[r])[0].max() if A[r].any() else -1 for r in rows]
    k = int(np.argmax(cols))
    u_n, v_n = cols[k] + 0.5, rows[k] + 0.5
    pa, pz = (a_hi - a_lo) / w, (z_hi - z_lo) / h
    da = a_n - (a_lo + u_n * pa)
    dz = z_n - (z_lo + v_n * pz)
    print("nose pin %s: shift %.3f, %.3f m" % (os.path.basename(path), da, dz))
    return (a_lo + da, a_hi + da, z_lo + dz, z_hi + dz) + tuple(fit[4:])


def view(path, a_mesh, a_sock, sign, tag):
    body = fit_view(a_mesh, co[:, 2], path, tag)
    if BUST and tag == "side":
        body = nose_pin(body, path, a_mesh)
    head = fit_view(a_mesh, co[:, 2], path, tag + " head", band=HEAD - 0.1, init=body[4])
    outs = []
    zs = z
    if ZW and tag == "front":
        # piecewise-linear z -> photo z, built from map-range nodes chained by "greater than" masks
        zs = None
        pts = ZW
        acc = None
        for (m0, p0), (m1, p1) in zip(pts[:-1], pts[1:]):
            seg = node("ShaderNodeMapRange", clamp=False)
            L(z, seg.inputs["Value"])
            seg.inputs["From Min"].default_value, seg.inputs["From Max"].default_value = m0, m1
            seg.inputs["To Min"].default_value, seg.inputs["To Max"].default_value = p0, p1
            inside = math_("MULTIPLY", math_("GREATER_THAN", z, m0 - (1e3 if (m0, p0) == pts[0] else 0)),
                           math_("LESS_THAN", z, m1 + (1e3 if (m1, p1) == pts[-1] else 0)))
            term = math_("MULTIPLY", seg.outputs["Result"], inside)
            acc = term if acc is None else math_("ADD", acc, term)
        zs = acc
    for f in (body, head):
        outs.append(projected(path, maprange(a_sock, sign * f[0], sign * f[1]), maprange(zs, f[2], f[3])))
    c = node("ShaderNodeMix", data_type="RGBA")
    L(head_mix.outputs[0], c.inputs["Factor"]); L(outs[0][0], c.inputs[6]); L(outs[1][0], c.inputs[7])
    al = node("ShaderNodeMix", data_type="FLOAT")
    L(head_mix.outputs[0], al.inputs["Factor"]); L(outs[0][1], al.inputs[2]); L(outs[1][1], al.inputs[3])
    return c.outputs[2], al.outputs[0], (body, head)


front_c, front_a, front_fits = view(FRONT, co[:, 0], x, 1.0, "front")
side_c, side_a, side_fits = view(SIDE, -co[:, 1], y, -1.0, "side")  # side ref faces screen-right = -Y
if LEFT:
    left_c, left_a, left_fits = view(LEFT, co[:, 1], y, 1.0, "left")  # seen from +X, faces screen-left = -Y
if BACK:
    back_photo_c, back_photo_a, back_fits = view(BACK, -co[:, 0], x, -1.0, "back")  # from behind, screen-right = -X


# ---------- cross-view validation ----------
# Where both the front and a side photo see a vertex, their colours should agree. If they don't (face skin
# vs hood cloth at the hood rim), the view hitting the surface at the more grazing angle is the wrong one.
def sample(path, fits, a, zz=None):
    im = bpy.data.images.load(path, check_existing=True)
    w, h = im.size
    px = np.empty(w * h * 4, "f")
    im.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    m = np.clip((co[:, 2] - (HEAD - 0.08)) / 0.08, 0, 1)[:, None]
    out = []
    for f in fits:
        u = np.clip(((a - f[0]) / (f[1] - f[0]) * w).astype(int), 0, w - 1)
        v = np.clip((((co[:, 2] if zz is None else zz) - f[2]) / (f[3] - f[2]) * h).astype(int), 0, h - 1)
        out.append(px[v, u])
    return out[0] * (1 - m) + out[1] * m


cf = sample(FRONT, front_fits, co[:, 0], zwarp(co[:, 2]))
cs = sample(SIDE, side_fits, -co[:, 1])
both = (cf[:, 3] > 0.5) & (cs[:, 3] > 0.5)
diff = np.linalg.norm(cf[:, :3] - cs[:, :3], axis=1)
fface = np.maximum(-nr[:, 1], 0)
sface = np.abs(nr[:, 0])
if LEFT:
    # each side photo is compared on its own side
    cl = sample(LEFT, left_fits, co[:, 1])
    on_left = nr[:, 0] > 0
    cs_side = np.where(on_left[:, None], cl, cs)
    both = (cf[:, 3] > 0.5) & (cs_side[:, 3] > 0.5)
    diff = np.linalg.norm(cf[:, :3] - cs_side[:, :3], axis=1)
seen_side = np.maximum(vis[:, 1], vis[:, 2]) > 0
clash = both & (vis[:, 0] > 0) & seen_side & (diff > 0.15)
drop_front = clash & (fface < sface)
drop_side = clash & ~drop_front
vis[drop_front, 0] = 0
vis[drop_side, 1:3] = 0
print("cross-view clashes: %d (front dropped %d, side dropped %d)" % (clash.sum(), drop_front.sum(), drop_side.sum()))
# ---------- semantic vote ----------
# Each reference has a hair / skin / clothing / shoes map (segment_refs.py). Every view that sees a vertex
# votes for its class, weighted by how squarely it sees it; views that disagree with the winner don't paint
# it. That keeps a face profile off the hair and face skin off a hood rim.
def labels(path, fits, a, zz=None):
    seg = path.replace("_crop.png", "_seg.npy")
    if not os.path.exists(seg):
        return None
    lab = np.load(seg)
    h, w = lab.shape
    m = np.clip((co[:, 2] - (HEAD - 0.08)) / 0.08, 0, 1)
    out = []
    for f in fits:
        u = np.clip(((a - f[0]) / (f[1] - f[0]) * w).astype(int), 0, w - 1)
        v = np.clip((((co[:, 2] if zz is None else zz) - f[2]) / (f[3] - f[2]) * h).astype(int), 0, h - 1)
        out.append(lab[h - 1 - v, u])
    return np.where(m > 0.5, out[1], out[0])


side_lab = labels(SIDE, side_fits, -co[:, 1])
# the left side only has the mirrored right-side photo: hair parts and asymmetries make it the least
# trustworthy view, so its vote counts for little and it loses every disagreement
views = [(labels(FRONT, front_fits, co[:, 0], zwarp(co[:, 2])), [0], fface, 1.0),
         (side_lab, [1], np.maximum(-nr[:, 0], 0), 1.0),
         (labels(LEFT, left_fits, co[:, 1]) if LEFT else side_lab, [2], np.maximum(nr[:, 0], 0), 1.0 if LEFT else 0.25)]
if BACK:
    views.append((labels(BACK, back_fits, -co[:, 0]), [3], np.maximum(nr[:, 1], 0), 1.0))
if all(v[0] is not None for v in views):
    # the mirrored photo is only valid where the figure is symmetric: compare the front labels at x and -x
    fl, fl_m = views[0][0], labels(FRONT, front_fits, -co[:, 0], zwarp(co[:, 2]))
    if LEFT:
        # side figures on a sheet are small and register worst on the head (face profile lands on hair):
        # above the chin let front, back and the in-part fill do the work
        fv = (fl == 2) & (co[:, 2] > zmax - 0.22) & (fface > 0.6)
        if fv.sum() > 50:
            chin = np.percentile(co[fv, 2], 5)
            # only the face itself (front-facing skin the front photo sees); hair keeps all four views
            face_zone = (co[:, 2] > chin - 0.02) & (fl == 2) & (nr[:, 1] < -0.2)
            vis[face_zone, 1:3] = 0
            print("side views off on the face: %d vertices" % face_zone.sum())
    asym = (fl > 0) & (fl_m > 0) & (fl != fl_m)
    if not LEFT:  # these guards are only for the mirrored stand-in
        vis[asym, 2] = 0
        # on the head the mirrored side may only paint what the front photo agrees with at that spot, even
        # where the front photo sees it edge-on (else a face profile lands on hair that hides the left ear)
        head = co[:, 2] > HEAD - 0.1
        vis[head & (fl > 0) & (side_lab > 0) & (side_lab != fl), 2] = 0
    print("asymmetric for the mirrored side: %d vertices" % asym.sum())
    score = np.zeros((len(co), 5))
    for lab, cols, face, trust in views:
        seen = vis[:, cols].max(1) * face * trust
        np.add.at(score, (np.arange(len(co)), lab), seen * (lab > 0))
    if PARTS:
        # 3D part labels (voted over 30 renders) decide wherever no reference sees the vertex squarely;
        # a reference looking straight at it still overrules them
        part = np.load(PARTS)
        # head prior: behind the ears there is only hair (the first paint can leave skin there, and the
        # part labels then inherit it). Head axis from the face skin the front photo sees squarely.
        front_lab = views[0][0]
        face_v = (front_lab == 2) & (co[:, 2] > zmax - 0.22) & (fface > 0.6)
        if face_v.sum() > 50:
            hx, hy = np.median(co[face_v, 0]), np.median(co[face_v, 1])
            chin = np.percentile(co[face_v, 2], 5)
            behind = (co[:, 1] > hy + 0.07) & (co[:, 2] > chin - 0.05)
            outside = np.hypot(co[:, 0] - hx, co[:, 1] - hy - 0.07) > 0.075  # not the neck column
            hair_prior = behind & outside & (part == 2)
            part = part.copy()
            part[hair_prior] = 1
            print("head prior: %d skin vertices behind the ears -> hair" % hair_prior.sum())
        score[np.arange(len(co)), part] += 0.6 * (part > 0)
        print("using part labels:", np.bincount(part, minlength=5).tolist())
    winner = score.argmax(1)
    decided = score.max(1) > 0
    dropped = 0
    for lab, cols, face, trust in views:
        bad = decided & (lab != winner) & (vis[:, cols].max(1) > 0)
        dropped += bad.sum()
        vis[np.ix_(bad, cols)] = 0
    # per-class palette from the references (hair / skin / clothing / shoes), linear colour; every vertex gets
    # the colour of its class as the fill for anything no photo paints reliably
    sums, counts = np.zeros((5, 3)), np.zeros(5)
    for path in [FRONT, SIDE] + ([BACK] if BACK else []):
        im = bpy.data.images.load(path, check_existing=True)
        w, h = im.size
        px = np.empty(w * h * 4, "f")
        im.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)[::-1]  # row 0 = top, like the label map
        lab = np.load(path.replace("_crop.png", "_seg.npy"))
        for c in range(1, 5):
            m = (lab == c) & (px[..., 3] > 0.5)
            sums[c] += px[m, :3].sum(0)
            counts[c] += m.sum()
    palette = (sums / np.maximum(counts, 1)[:, None]) ** 2.2  # sRGB-ish -> linear
    palette[0] = palette[3]
    cls = np.where(decided, winner, np.where(fl > 0, fl, 3))
    cheek = (cls == 1) & (fl == 2) & (fface > 0.5) & (co[:, 2] > zmax - 0.3)
    cls[cheek] = 2
    print("face skin wrongly in hair -> skin: %d" % cheek.sum())
    # skin is a closed volume (legs, arms, torso); a sheer skirt or corset panel is a thin sheet. Anything
    # labelled skin that is under ~1.2 cm thick is fabric (head with its thin ears and the hands excluded).
    tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    cand = np.nonzero((cls == 2) & (co[:, 2] > 0.3) & (co[:, 2] < 1.35) & (np.abs(co[:, 0]) < 0.24))[0] \
        if not BUST else np.zeros(0, int)
    thin = np.zeros(len(co), bool)
    for i in cand:
        n_ = Vector(nr[i])
        hit = tree.ray_cast(Vector(co[i]) - n_ * 0.001, -n_, 0.012)
        thin[i] = hit[0] is not None
    cls[thin] = 3
    print("thin sheets relabelled skin -> clothing: %d" % thin.sum())
    # likewise above the waist: a "skin" sheet under ~3 cm thick that isn't an ear is a lock of hair
    fv = (fl == 2) & (co[:, 2] > zmax - 0.22) & (fface > 0.6)
    if fv.sum() > 50 and not BUST:
        hx, hy, chin = np.median(co[fv, 0]), np.median(co[fv, 1]), np.percentile(co[fv, 2], 5)
        ear = (np.hypot(co[:, 0] - hx, co[:, 1] - hy) < 0.12) & (co[:, 2] > chin)
        cand = np.nonzero((cls == 2) & (co[:, 2] > 1.1) & ~ear)[0]
        lock = np.zeros(len(co), bool)
        for i in cand:
            n_ = Vector(nr[i])
            lock[i] = tree.ray_cast(Vector(co[i]) - n_ * 0.001, -n_, 0.03)[0] is not None
        cls[lock] = 1
        print("thin locks relabelled skin -> hair: %d" % lock.sum())
    print("class palette (linear):", palette[1:].round(3).tolist())
    if os.environ.get("PAINT_DEBUG"):
        r = (co[:, 0] > 0.04) & (co[:, 2] > 1.55) & (nr[:, 0] > 0.4)
        print("DEBUG left head", r.sum(), "front lab", np.bincount(views[0][0][r], minlength=5),
              "side lab", np.bincount(side_lab[r], minlength=5), "winner", np.bincount(winner[r], minlength=5),
              "vis f/r/l/b", vis[r].mean(0).round(3), "fface", fface[r].mean().round(2), "asym", asym[r].mean().round(2))
    print("semantic vote: %d view-vertex pairs dropped; classes %s" % (
        dropped, np.bincount(winner[decided], minlength=5).tolist()))

# a photo hitting the surface almost edge-on smears whatever is next to it (face onto the hood rim): drop it
vis[:, 0] *= np.clip((fface - 0.15) / 0.2, 0, 1)
vis[:, 1:3] *= np.clip((sface - 0.15) / 0.2, 0, 1)[:, None]
if BACK:
    vis[:, 3] *= np.clip((np.maximum(nr[:, 1], 0) - 0.15) / 0.2, 0, 1)
if BUST and "cls" in globals():
    # the face is the point of a bust: whatever the front photo sees as skin stays skin and is painted from
    # the front photo alone; cheeks and temples the profile sees as skin come from the profile
    zone_f = (fl == 2) & (fface > 0.2)
    fz = co[zone_f, 2]
    lo, hi = (np.percentile(fz, 2), np.percentile(fz, 98)) if zone_f.sum() > 50 else (0, 0)
    zone_s = (side_lab == 2) & (np.abs(nr[:, 0]) > 0.35) & (co[:, 2] > lo) & (co[:, 2] < hi) & ~zone_f \
        & (nr[:, 1] < 0.3)
    cls[zone_f | zone_s] = 2
    vis[zone_f, 0] = 1.0
    vis[zone_f, 1:] = 0.0
    vis[zone_s, 0] = 0.0
    vis[zone_s, 3] = 0.0
    vis[zone_s & (nr[:, 0] < 0), 1] = 1.0
    vis[zone_s & (nr[:, 0] > 0), 2] = 1.0
    print("face lock: %d from the front photo, %d from the profile" % (zone_f.sum(), zone_s.sum()))
me.color_attributes["vis"].data.foreach_set("color", vis.ravel())  # re-fetch: edit mode invalidated the old handle

# ---------- fill: grow each part's own colours into what no photo paints ----------
if "cls" in globals():
    lin = lambda c: np.clip(c[:, :3], 0, 1) ** 2.2
    cols = [(lin(cf), vis[:, 0] * fface ** 3 * cf[:, 3])]
    if LEFT:
        cols += [(lin(cs), vis[:, 1] * np.maximum(-nr[:, 0], 0) ** 6 * cs[:, 3]),
                 (lin(cl), vis[:, 2] * np.maximum(nr[:, 0], 0) ** 6 * cl[:, 3])]
    else:
        cols.append((lin(cs), (vis[:, 1] * np.maximum(-nr[:, 0], 0) ** 6
                               + vis[:, 2] * np.maximum(nr[:, 0], 0) ** 6) * cs[:, 3]))
    if BACK:
        cb = sample(BACK, back_fits, -co[:, 0])
        cols.append((lin(cb), vis[:, 3] * np.maximum(nr[:, 1], 0) ** 3 * cb[:, 3]))
    wsum = sum(w for _, w in cols)
    val = sum(c * w[:, None] for c, w in cols) / np.maximum(wsum, 1e-6)[:, None]
    has = wsum > 0.05

    # spot cleanup: inside hair and skin, a vertex whose photo colour is far from that part's typical colour
    # is a stray (skin or an ear on the hair, a strap or a strand on a shoulder). Drop its projection and let the
    # part's own colour grow back in. The face keeps its details (eyes, brows and lips are "outliers" too).
    def to_lab(c):
        c = np.clip(c, 1e-6, 1)
        xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
        xyz /= np.array([0.9505, 1.0, 1.089])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

    face = (co[:, 2] > HEAD) & (nr[:, 1] < -0.35) & (np.abs(co[:, 0]) < 0.07)
    lab_c = to_lab(val)
    spots = np.zeros(len(co), bool)
    for c, floor in (() if BUST else ((2, 9.0),)):  # skin (hair is handled by hue below: its violet highlights are not spots)
        m = has & (cls == c) & ~face
        if m.sum() < 100:
            continue
        med = np.median(lab_c[m], 0)
        d = np.linalg.norm(lab_c - med, axis=1)
        cut = max(floor, np.median(d[m]) + 2.2 * np.median(np.abs(d[m] - np.median(d[m]))))
        spots |= m & (d > cut)
        print("cleanup %s: median Lab %s, cut %.1f, spots %d" % (["", "hair", "skin"][c], med.round(1), cut,
                                                                  (m & (d > cut)).sum()))
    skin_m0 = has & (cls == 2) & ~face
    if skin_m0.sum() > 100:
        sm = np.median(lab_c[skin_m0], 0)
        hue_d = np.abs(np.degrees(np.arctan2(lab_c[:, 2], lab_c[:, 1]) - np.arctan2(sm[2], sm[1])))
        hair_spot = has & (cls == 1) & ~face & (hue_d < 25) & (np.hypot(lab_c[:, 1], lab_c[:, 2]) > 8) \
            & (lab_c[:, 0] > 12)
        spots |= hair_spot
        print("cleanup hair: %d skin-toned vertices" % hair_spot.sum())
    # clothing, shoes and accessories never show skin (sheer fabric included): skin-toned vertices there are
    # strays too
    skin_m = has & (cls == 2) & ~face
    if skin_m.sum() > 100:
        skin_med = np.median(lab_c[skin_m], 0)
        # wide radius for light tones (legs seen through the sheer skirt read as a muddy skin tone), but only
        # for colours at least as light as shadowed skin, so dark fabric and the gold trim stay
        # judged by hue rather than distance: gold trim is about as far from skin as a muddy leg-through-skirt
        # tone, but its hue sits ~35 degrees further toward yellow
        hue = np.degrees(np.arctan2(lab_c[:, 2], lab_c[:, 1]))
        chroma = np.hypot(lab_c[:, 1], lab_c[:, 2])
        skin_hue = np.degrees(np.arctan2(skin_med[2], skin_med[1]))
        skinlike = (np.abs(hue - skin_hue) < 14) & (chroma > 7) & (lab_c[:, 0] > 26)
        on_cloth = has & np.isin(cls, (3, 4)) & skinlike
        spots |= on_cloth
        print("cleanup clothing: %d skin-toned vertices" % on_cloth.sum())
    # spots come in clusters: grow the mask one ring so their soft edges go too
    e2 = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", e2)
    e2 = e2.reshape(-1, 2)
    grow = spots.copy()
    grow[e2[spots[e2[:, 0]], 1]] = True
    grow[e2[spots[e2[:, 1]], 0]] = True
    spots = grow & np.isin(cls, (1, 2, 3, 4)) & ~face
    has &= ~spots
    vis[spots] = 0
    me.color_attributes["vis"].data.foreach_set("color", vis.ravel())
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    ed = ed[cls[ed[:, 0]] == cls[ed[:, 1]]]  # colour only flows inside a part
    src = np.concatenate([ed[:, 0], ed[:, 1]])
    dst = np.concatenate([ed[:, 1], ed[:, 0]])
    for it in range(400):
        ok = has[src] & ~has[dst]
        if not ok.any():
            break
        acc = np.zeros_like(val)
        cnt = np.zeros(len(co))
        np.add.at(acc, dst[ok], val[src[ok]])
        np.add.at(cnt, dst[ok], 1)
        new = cnt > 0
        val[new] = acc[new] / cnt[new, None]
        has |= new
    # regrown patches pick up whatever sat on their rim: relax them toward their in-part neighbours
    regrown = ~(wsum > 0.05) | spots
    for _ in range(30):
        acc = np.zeros_like(val)
        cnt = np.zeros(len(co))
        np.add.at(acc, dst, val[src])
        np.add.at(cnt, dst, 1)
        upd = regrown & (cnt > 0) & has
        val[upd] = 0.5 * val[upd] + 0.5 * acc[upd] / cnt[upd, None]
    val[~has] = palette[cls[~has]]
    print("fill: %d painted by photos, %d grown inside their part, %d from the part palette" % (
        (wsum > 0.05).sum(), (has & ~(wsum > 0.05)).sum(), (~has).sum()))
    fill = np.ones((len(co), 4), "f")
    fill[:, :3] = val
    me.color_attributes.new("fill", "FLOAT_COLOR", "POINT").data.foreach_set("color", fill.ravel())

vc = node("ShaderNodeVertexColor", layer_name="vis")
vsep = node("ShaderNodeSeparateColor"); L(vc.outputs["Color"], vsep.inputs[0])
v_front, v_right, v_left = vsep.outputs
v_back = vc.outputs["Alpha"]


def facing(comp, sign, power=3.0):
    f = math_("MAXIMUM", math_("MULTIPLY", comp, sign), 0.0)
    return math_("POWER", f, power)


w_front = math_("MULTIPLY", math_("MULTIPLY", facing(ny, -1.0), v_front), front_a)
if LEFT:
    w_side = math_("MULTIPLY", math_("MULTIPLY", facing(nx, -1.0, 6.0), v_right), side_a)
    w_left = math_("MULTIPLY", math_("MULTIPLY", facing(nx, 1.0, 6.0), v_left), left_a)
else:
    w_side = math_("MULTIPLY", math_("ADD", math_("MULTIPLY", facing(nx, -1.0, 6.0), v_right),
                                     math_("MULTIPLY", facing(nx, 1.0, 6.0), v_left)), side_a)
if BACK:
    w_bphoto = math_("MULTIPLY", math_("MULTIPLY", facing(ny, 1.0), v_back), back_photo_a)
    w_back = math_("MULTIPLY", 0.02, 1.0)  # only a faint fallback for what no photo sees
else:
    w_back = math_("MULTIPLY", math_("ADD", facing(ny, 1.0), 0.02), v_back)  # always a little fallback

# back / unseen colour: cloak cloth above the boots, leather below, with some mottling
ramp = node("ShaderNodeValToRGB")
L(z, ramp.inputs["Fac"])
ramp.color_ramp.elements[0].position = 0.11
ramp.color_ramp.elements[0].color = (0.035, 0.024, 0.017, 1)
ramp.color_ramp.elements[1].position = 0.16
ramp.color_ramp.elements[1].color = (0.024, 0.030, 0.018, 1)
noise = node("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 6.0
L(geo.outputs["Position"], noise.inputs["Vector"])
mott = math_("ADD", math_("MULTIPLY", noise.outputs["Fac"], 0.8), 0.6)
back_c = vmath("SCALE", ramp.outputs["Color"], scale=mott)
if "fill" in me.color_attributes:
    # semantic fill: the vertex's class colour (hair / skin / clothing / shoes) with a little mottling
    fill_node = node("ShaderNodeVertexColor", layer_name="fill")
    back_c = vmath("SCALE", fill_node.outputs["Color"], scale=math_("ADD", math_("MULTIPLY", noise.outputs["Fac"], 0.5), 0.75))

num = vmath("ADD", vmath("ADD", vmath("SCALE", front_c, scale=w_front), vmath("SCALE", side_c, scale=w_side)),
            vmath("SCALE", back_c, scale=w_back))
den = math_("ADD", math_("ADD", w_front, w_side), math_("ADD", w_back, 1e-4))
if BACK:
    num = vmath("ADD", num, vmath("SCALE", back_photo_c, scale=w_bphoto))
    den = math_("ADD", den, w_bphoto)
if LEFT:
    num = vmath("ADD", num, vmath("SCALE", left_c, scale=w_left))
    den = math_("ADD", den, w_left)
col = vmath("SCALE", num, scale=math_("DIVIDE", 1.0, den))

emit = node("ShaderNodeEmission")
L(col, emit.inputs["Color"])
out = node("ShaderNodeOutputMaterial")
L(emit.outputs[0], out.inputs["Surface"])

bake_img = bpy.data.images.new(NAME + "_albedo", TEX, TEX)
target = node("ShaderNodeTexImage", image=bake_img)
nt.nodes.active = target

# ---------- bake ----------
t = time.time()
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.device = "CPU"
scene.render.bake.margin = 8
bpy.ops.object.bake(type="EMIT")
print("bake in %.0fs" % (time.time() - t))
bake_img.filepath_raw = os.path.join(OUT, NAME + "_albedo.png")
bake_img.file_format = "PNG"
bake_img.save()

# ---------- final material ----------
fin = bpy.data.materials.new(NAME)
b = fin.node_tree.nodes["Principled BSDF"]
b.inputs["Roughness"].default_value = 0.72
tx = fin.node_tree.nodes.new("ShaderNodeTexImage")
tx.image = bake_img
fin.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
obj.data.materials.clear()
obj.data.materials.append(fin)
me.color_attributes.remove(me.color_attributes["vis"])
if "fill" in me.color_attributes:
    me.color_attributes.remove(me.color_attributes["fill"])

# previews
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "TEXTURE"
scene.render.resolution_x, scene.render.resolution_y = 640, 1000
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = 2.15
scene.collection.objects.link(cam)
scene.camera = cam
for tag, ang in (("front", 0), ("three_q", 40), ("side", -90), ("back", 180)):
    a = math.radians(ang)
    cam.location = (5 * math.sin(a), -5 * math.cos(a), 0.97)
    cam.rotation_euler = (math.pi / 2, 0, a)
    scene.render.filepath = os.path.join(OUT, "paint_%s.png" % tag)
    bpy.ops.render.render(write_still=True)
bpy.data.objects.remove(cam)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_painted.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + "_painted.glb"), export_format="GLB",
                          export_image_format="JPEG", export_vertex_color="NONE")
print("DONE")
