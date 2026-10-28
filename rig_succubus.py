"""Humanoid armature + skin weights for the merged succubus2 (body object + high-res head object).

Character faces -Y, +X is her left (.L), feet at z=0. Joints were read off 10 cm grid renders (front + side).
The body is weighted with Blender's bone heat (distance fallback); her hair curls only follow the upper body;
the head object (head, neck and the hair falling behind) is weighted by distance to the spine bones only.

Usage: blender -b merged.blend -P rig_succubus.py -- OUT_DIR
"""
import bpy, os, sys
import numpy as np
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1:][0]
body = bpy.data.objects["succubus2"]
head = bpy.data.objects["head"]
scene = bpy.context.scene

# name: (head, tail, parent, deform, limb radius limit or "torso")
J = {
    "root": ((0, 0, 0), (0, 0.25, 0), None, False, None),
    "hips": ((0, -0.02, 0.90), (0, -0.02, 1.05), "root", True, "torso"),
    "spine": ((0, -0.02, 1.05), (0, -0.03, 1.22), "hips", True, "torso"),
    "chest": ((0, -0.03, 1.22), (0, -0.04, 1.44), "spine", True, "torso"),
    "neck": ((0, -0.04, 1.44), (0, -0.06, 1.56), "chest", True, 0.10),
    "head": ((0, -0.06, 1.56), (0, -0.07, 1.80), "neck", True, 0.24),
}
for s, sx in (("L", 1), ("R", -1)):
    hip_x, knee_x, ankle_x = (0.085, 0.15, 0.215) if s == "L" or os.environ.get("RIG_SYM") else (0.085, 0.115, 0.185)
    J.update({
        f"shoulder.{s}": ((0.03 * sx, -0.04, 1.42), (0.14 * sx, -0.03, 1.41), "chest", True, 0.08),
        f"upper_arm.{s}": ((0.14 * sx, -0.03, 1.41), (0.215 * sx, 0.0, 1.18), f"shoulder.{s}", True, 0.075),
        f"forearm.{s}": ((0.215 * sx, 0.0, 1.18), (0.285 * sx, -0.02, 0.95), f"upper_arm.{s}", True, 0.07),
        f"hand.{s}": ((0.285 * sx, -0.02, 0.95), (0.305 * sx, -0.04, 0.81), f"forearm.{s}", True, 0.07),
        f"thigh.{s}": ((hip_x * sx, -0.02, 0.88), (knee_x * sx, -0.07, 0.49), "hips", True, 0.12),
        f"shin.{s}": ((knee_x * sx, -0.07, 0.49), (ankle_x * sx, -0.04, 0.10), f"thigh.{s}", True, 0.10),
        f"foot.{s}": ((ankle_x * sx, -0.04, 0.10), ((ankle_x + 0.01) * sx, -0.19, 0.03), f"shin.{s}", True, 0.09),
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

# vertices by material (hair curls on the shoulder, the free skirt sheet)
mat_of = np.zeros(len(co), int)
mnames = [m.name for m in body.data.materials]
for p in body.data.polygons:
    for v in p.vertices:
        mat_of[v] = p.material_index
is_mat = lambda n: mat_of == mnames.index(n) if n in mnames else np.zeros(len(co), bool)
upper = [names.index(n) for n in ("chest", "neck", "head", "spine")]
hair = is_mat("hair")
W[np.ix_(hair, [k for k in range(len(names)) if k not in upper])] = 0
W[hair] += dist_weights(co[hair], {"chest", "neck", "head", "spine"}) * 1e-6
# arms own nothing near the hip/skirt below the wrist line apart from the hand itself
arm_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("upper_arm", "forearm", "hand", "shoulder")]
leg_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("thigh", "shin", "foot")]
skirt = is_mat("skirt") | is_mat("skirt_on_body")
W[np.ix_(skirt, arm_k)] = 0
# legs don't own anything of the torso above the hip joints
W[np.ix_(co[:, 2] > 0.98, leg_k)] = 0
# the bracers are shells around the forearms: only that side's arm bones may move them (heat weighting lets
# the torso pull on them, which tears them while walking)
gloves = is_mat("gloves")
for side, sgn in (("L", 1), ("R", -1)):
    arm = [names.index(f"{b}.{side}") for b in ("upper_arm", "forearm", "hand")]
    fa = J[f"forearm.{side}"]
    hd = J[f"hand.{side}"]
    g = gloves & (np.sign(co[:, 0]) == sgn)
    W[np.ix_(g, [k for k in range(len(names)) if k not in arm])] = 0
    empty = g & (W.sum(1) < 1e-6)
    W[np.ix_(empty, arm)] = dist_weights(co[empty], {f"{b}.{side}" for b in ("upper_arm", "forearm", "hand")})[:, arm]
W[W.sum(1) < 1e-6] = W_dist[W.sum(1) < 1e-6] + 1e-9
W /= W.sum(1, keepdims=True)
W = smooth(body, W, 10)  # wider blend across elbows/knees/hips: softer, rounder bends
# the free sheet of the skirt hangs from the waist: it may swing with the thighs but never follows the shins/feet,
# and keeps most of its weight on the hips so big leg moves (kicks, dancing) don't stretch it across the floor
fs = is_mat("skirt")
low = [k for k, n in enumerate(names) if n.split(".")[0] in ("shin", "foot")]
W[np.ix_(fs, low)] = 0
W[fs] /= W[fs].sum(1, keepdims=True) + 1e-12
W[fs] = 0  # any leg weight shears the long sheet (the hem stretched in every animation): it moves with the hips
W[fs, names.index("hips")] = 1
# the bracers are a separate shell over the forearm: give every bracer vertex exactly the weights of the arm
# surface right next to it, so shell and arm move as one (own weights let it lag and stick out mid-stride)
from mathutils.kdtree import KDTree
arm_all = [k for k, n in enumerate(names) if n.split(".")[0] in ("upper_arm", "forearm", "hand")]
for sgn in (1, -1):
    g = gloves & (np.sign(co[:, 0]) == sgn)
    src = np.nonzero(~gloves & (np.sign(co[:, 0]) == sgn) & (W[:, arm_all].sum(1) > 0.6))[0]
    kd = KDTree(len(src))
    for i in src:
        kd.insert(co[i], int(i))
    kd.balance()
    for i in np.nonzero(g)[0]:
        _, j, d = kd.find(co[i])
        if d < 0.06:
            W[i] = W[j]
write_groups(body, W)

# ---------- head object: spine chain only ----------
hco = np.array([head.matrix_world @ v.co for v in head.data.vertices], dtype=np.float64)
HW = dist_weights(hco, {"chest", "neck", "head"})
HW = smooth(head, HW / HW.sum(1, keepdims=True), 8)
write_groups(head, HW)

for b in rig.pose.bones:
    b.rotation_mode = "QUATERNION"
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "succubus2_rigged.blend"))
print("DONE")
