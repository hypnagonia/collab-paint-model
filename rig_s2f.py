"""Humanoid armature + skin weights for s2f (single mesh painted by Hunyuan3D-2 Paint, A-pose, long hair, wrap skirt).

Character faces -Y, +X is her left (.L), feet at z=0. Joints were read off 10 cm grid renders (front + side).
The mesh is weighted with Blender's bone heat (distance fallback); the long hair (from the 3D part labels)
only follows the upper body.

Parts come from the painted colours: hair = dark violet above the shoulders, bracers = dark around the forearms,
the free skirt sheet = surface hanging away from both legs.

Usage: blender -b painted_face.blend -P rig_s2f.py -- - OUT_DIR
"""
import bpy, os, sys
import numpy as np
from mathutils import Vector

PARTS, OUT = sys.argv[sys.argv.index("--") + 1:][:2]  # PARTS unused (kept for the call shape)
body = next(o for o in bpy.data.objects if o.type == "MESH")
# the painter's UV unwrap split the mesh into islands with duplicated vertices; bone heat needs one connected
# surface. Welding is safe: UVs live on face corners, so the seams stay in the texture coordinates
import bmesh
_bm = bmesh.new(); _bm.from_mesh(body.data)
_n0 = len(_bm.verts)
bmesh.ops.remove_doubles(_bm, verts=_bm.verts[:], dist=1e-5)
_bm.to_mesh(body.data); _bm.free()
print("welded %d duplicated vertices" % (_n0 - len(body.data.vertices)))
scene = bpy.context.scene

# name: (head, tail, parent, deform, limb radius limit or "torso")
J = {
    "root": ((0, 0, 0), (0, 0.25, 0), None, False, None),
    "hips": ((0, -0.02, 0.90), (0, -0.02, 1.04), "root", True, "torso"),
    "spine": ((0, -0.02, 1.04), (0, -0.03, 1.20), "hips", True, "torso"),
    "chest": ((0, -0.03, 1.20), (0, -0.04, 1.40), "spine", True, "torso"),
    "neck": ((0, -0.04, 1.40), (0, -0.05, 1.52), "chest", True, 0.10),
    "head": ((0, -0.05, 1.52), (0, -0.06, 1.78), "neck", True, 0.24),
}
for s, sx in (("L", 1), ("R", -1)):
    J.update({
        f"shoulder.{s}": ((0.03 * sx, -0.04, 1.37), (0.15 * sx, -0.03, 1.36), "chest", True, 0.08),
        f"upper_arm.{s}": ((0.15 * sx, -0.03, 1.36), (0.23 * sx, 0.0, 1.14), f"shoulder.{s}", True, 0.075),
        f"forearm.{s}": ((0.23 * sx, 0.0, 1.14), (0.30 * sx, -0.02, 0.96), f"upper_arm.{s}", True, 0.07),
        f"hand.{s}": ((0.30 * sx, -0.02, 0.96), (0.31 * sx, -0.04, 0.82), f"forearm.{s}", True, 0.07),
        f"thigh.{s}": ((0.085 * sx, -0.02, 0.88), (0.13 * sx, -0.06, 0.52), "hips", True, 0.12),
        f"shin.{s}": ((0.13 * sx, -0.06, 0.52), (0.20 * sx, -0.03, 0.11), f"thigh.{s}", True, 0.10),
        f"foot.{s}": ((0.20 * sx, -0.03, 0.11), (0.21 * sx, -0.18, 0.03), f"shin.{s}", True, 0.09),
    })

arm_data = bpy.data.armatures.new("succubus2_rig")
rig = bpy.data.objects.new("succubus2_rig", arm_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones
for name, (h, t, parent, deform, _) in J.items():
    b = eb.new(name)
    b.head, b.tail = h, t
    b.use_deform = deform
for name, (h, t, parent, deform, _) in J.items():
    if parent:
        eb[name].parent = eb[parent]
        eb[name].use_connect = (Vector(eb[parent].tail) - Vector(h)).length < 1e-4 and deform
for b in eb:  # roll so local X is the character's left-right axis on every bone
    b.align_roll(Vector((0, 0, 1)) if abs(b.vector.normalized().z) < 0.7 else Vector((0, -1, 0)))
bpy.ops.object.mode_set(mode="OBJECT")

deform = [(n, np.array(h), np.array(t), lim) for n, (h, t, _, d, lim) in J.items() if d]
names = [d[0] for d in deform]


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def dist_weights(co, allowed=None):
    D = np.stack([seg_dist(co, a, b) for _, a, b, _ in deform], 1)
    W = 1.0 / (D + 0.015) ** 5
    for k, (n, _, _, lim) in enumerate(deform):
        if allowed is not None and n not in allowed:
            W[:, k] = 0
        elif lim != "torso" and allowed is None:
            W[:, k] *= np.clip((lim - D[:, k]) / (0.3 * lim), 0, 1)
    return W


def smooth(mesh, W, it=4):
    ed = np.empty(len(mesh.data.edges) * 2, np.int64)
    mesh.data.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    deg = np.bincount(ed.ravel(), minlength=len(W)).astype(np.float64)[:, None] + 1e-9
    for _ in range(it):
        acc = np.zeros_like(W)
        np.add.at(acc, ed[:, 0], W[ed[:, 1]])
        np.add.at(acc, ed[:, 1], W[ed[:, 0]])
        W = 0.5 * W + 0.5 * acc / deg
    return W / (W.sum(1, keepdims=True) + 1e-12)


def write_groups(mesh, W):
    for g in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(g)
    for k, n in enumerate(names):
        vg = mesh.vertex_groups.new(name=n)
        w = W[:, k]
        for i in np.nonzero(w > 0.002)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    for m in [m for m in mesh.modifiers if m.type == "ARMATURE"]:
        mesh.modifiers.remove(m)
    mesh.modifiers.new("rig", "ARMATURE").object = rig
    mesh.parent = rig


# ---------- body: bone heat, distance fallback ----------
co = np.array([body.matrix_world @ v.co for v in body.data.vertices], dtype=np.float64)
W_dist = dist_weights(co)
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
W = np.zeros((len(co), len(names)))
gi = {g.index: names.index(g.name) for g in body.vertex_groups if g.name in names}
for v in body.data.vertices:
    for g in v.groups:
        if g.group in gi:
            W[v.index, gi[g.group]] = g.weight
miss = W.sum(1) < 1e-3
print("heat-unsolved body vertices: %d of %d" % (miss.sum(), len(co)))
W[miss] = W_dist[miss]

# the long hair hangs down the back beside the arms: it follows only the upper body
upper = [names.index(n) for n in ("chest", "neck", "head", "spine")]
# painted colour per vertex
_img = next(i for i in bpy.data.images if i.size[0] >= 512)
_TW, _TH = _img.size
_tp = np.empty(_TW * _TH * 4, "f"); _img.pixels.foreach_get(_tp); _tp = _tp.reshape(_TH, _TW, 4)
_uv = np.empty(len(body.data.loops) * 2, "f"); body.data.uv_layers.active.data.foreach_get("uv", _uv); _uv = _uv.reshape(-1, 2)
_lv = np.empty(len(body.data.loops), np.int64); body.data.loops.foreach_get("vertex_index", _lv)
_vuv = np.zeros((len(co), 2), "f"); _vuv[_lv] = _uv
col = _tp[np.clip((_vuv[:, 1] * _TH).astype(int), 0, _TH - 1), np.clip((_vuv[:, 0] * _TW).astype(int), 0, _TW - 1), :3]
dark_violet = (col.max(1) < 0.30) & (col[:, 2] >= col[:, 0] - 0.02)
hair = dark_violet & (co[:, 2] > 1.12) & ~((np.abs(co[:, 0]) < 0.17) & (co[:, 1] < -0.06) & (co[:, 2] < 1.40))
print("hair vertices", hair.sum())
W[np.ix_(hair, [k for k in range(len(names)) if k not in upper])] = 0
W[hair] += dist_weights(co[hair], {"chest", "neck", "head", "spine"}) * 1e-6
# arms own nothing near the hip/skirt below the wrist line apart from the hand itself
arm_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("upper_arm", "forearm", "hand", "shoulder")]
leg_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("thigh", "shin", "foot")]
# arms own nothing on the hips/shorts beside the hands
W[np.ix_((np.abs(co[:, 0]) < 0.2) & (co[:, 2] < 1.0), arm_k)] = 0  # the hips beside the fists
# legs don't own anything of the torso above the hip joints
W[np.ix_(co[:, 2] > 0.92, leg_k)] = 0
W[W.sum(1) < 1e-6] = W_dist[W.sum(1) < 1e-6] + 1e-9
W /= W.sum(1, keepdims=True)
W = smooth(body, W, 10)  # wide blends at the joints: soft, round bends
# the free skirt sheet hangs away from both legs: it moves with the hips only (leg weight shears it)
leg_d = np.minimum.reduce([seg_dist(co, np.array(J[f"{b}.{s}"][0]), np.array(J[f"{b}.{s}"][1]))
                           for b in ("thigh", "shin") for s in "LR"])
# soft: fully on the hips far from the legs, blending into the leg weights near them (a hard edge tore the
# sheet away from the thigh it is fused to); bare skin is leg, never sheet
skin_col = (col[:, 0] > col[:, 2] + 0.08) & (col[:, 0] > 0.25)
# the dark gloved hands hang beside the hips too: anything near a forearm/hand bone is arm, never sheet
arm_d = np.minimum.reduce([seg_dist(co, np.array(J[f"{b}.{s}"][0]), np.array(J[f"{b}.{s}"][1]))
                           for b in ("forearm", "hand") for s in "LR"])
f = np.clip((leg_d - 0.08) / 0.12, 0, 1) ** 1.5 * ((co[:, 2] < 0.95) & (co[:, 2] > 0.15) & ~skin_col)
f *= np.clip((arm_d - 0.07) / 0.04, 0, 1)
hips_k = names.index("hips")
W = W * (1 - f[:, None]); W[:, hips_k] += f
W = smooth(body, W, 8)
sheet = f > 0.5
# bracers: dark shells around the forearms take the weights of the arm surface under them
from mathutils.kdtree import KDTree
arm_all = [k for k, n in enumerate(names) if n.split(".")[0] in ("upper_arm", "forearm", "hand")]
for side, sgn in (("L", 1), ("R", -1)):
    fa = J[f"forearm.{side}"]
    near = seg_dist(co, np.array(fa[0]), np.array(fa[1])) < 0.075
    shell = near & (col.max(1) < 0.25) & (np.sign(co[:, 0]) == sgn)
    src = np.nonzero(near & ~shell & (W[:, arm_all].sum(1) > 0.6))[0]
    if len(src):
        kd = KDTree(len(src))
        for i in src:
            kd.insert(co[i], int(i))
        kd.balance()
        for i in np.nonzero(shell)[0]:
            W[i] = W[kd.find(co[i])[1]]
print("skirt sheet vertices", sheet.sum())
write_groups(body, W)

for b in rig.pose.bones:
    b.rotation_mode = "QUATERNION"
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "succubus2_rigged.blend"))
print("DONE")
