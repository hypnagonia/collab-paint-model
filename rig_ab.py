"""Humanoid armature + skin weights for "ab" (single mesh from a front + back sheet, A-pose, ponytail).

Character faces -Y, +X is her left (.L), feet at z=0. Joints were read off 10 cm grid renders (front + side).
The mesh is weighted with Blender's bone heat (distance fallback); the long hair (from the 3D part labels)
only follows the upper body.

Usage: blender -b s3_painted.blend -P rig_ab.py -- PART_LABELS.npy OUT_DIR
"""
import bpy, os, sys
import numpy as np
from mathutils import Vector

PARTS, OUT = sys.argv[sys.argv.index("--") + 1:][:2]
body = bpy.data.objects["ab"]
scene = bpy.context.scene

# name: (head, tail, parent, deform, limb radius limit or "torso")
J = {
    "root": ((0, 0, 0), (0, 0.25, 0), None, False, None),
    "hips": ((0, 0.02, 0.78), (0, 0.01, 0.93), "root", True, "torso"),
    "spine": ((0, 0.01, 0.93), (0, 0.0, 1.08), "hips", True, "torso"),
    "chest": ((0, 0.0, 1.08), (0, -0.02, 1.30), "spine", True, "torso"),
    "neck": ((0, -0.02, 1.30), (0, -0.03, 1.40), "chest", True, 0.08),
    "head": ((0, -0.03, 1.40), (0, -0.03, 1.62), "neck", True, 0.22),
}
for s, sx in (("L", 1), ("R", -1)):
    J.update({
        f"shoulder.{s}": ((0.03 * sx, -0.01, 1.30), (0.15 * sx, 0.02, 1.33), "chest", True, 0.07),
        f"upper_arm.{s}": ((0.15 * sx, 0.02, 1.33), (0.25 * sx, 0.01, 1.06), f"shoulder.{s}", True, 0.065),
        f"forearm.{s}": ((0.25 * sx, 0.01, 1.06), (0.32 * sx, -0.03, 0.80), f"upper_arm.{s}", True, 0.06),
        f"hand.{s}": ((0.32 * sx, -0.03, 0.80), (0.33 * sx, -0.04, 0.69), f"forearm.{s}", True, 0.07),
        f"thigh.{s}": ((0.09 * sx, 0.02, 0.76), (0.10 * sx, 0.0, 0.36), "hips", True, 0.12),
        f"shin.{s}": ((0.10 * sx, 0.0, 0.36), (0.105 * sx, 0.04, 0.08), f"thigh.{s}", True, 0.10),
        f"foot.{s}": ((0.105 * sx, 0.04, 0.08), (0.11 * sx, -0.10, 0.02), f"shin.{s}", True, 0.09),
    })

arm_data = bpy.data.armatures.new("ab_rig")
rig = bpy.data.objects.new("ab_rig", arm_data)
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
hair = (np.load(PARTS) == 1) & (co[:, 2] > 1.1)  # the ponytail
W[np.ix_(hair, [k for k in range(len(names)) if k not in upper])] = 0
W[hair] += dist_weights(co[hair], {"chest", "neck", "head", "spine"}) * 1e-6
# arms own nothing near the hip/skirt below the wrist line apart from the hand itself
arm_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("upper_arm", "forearm", "hand", "shoulder")]
leg_k = [k for k, n in enumerate(names) if n.split(".")[0] in ("thigh", "shin", "foot")]
# arms own nothing on the hips/shorts beside the hands
W[np.ix_((np.abs(co[:, 0]) < 0.2) & (co[:, 2] < 1.0), arm_k)] = 0
# legs don't own anything of the torso above the hip joints
W[np.ix_(co[:, 2] > 0.86, leg_k)] = 0
W[W.sum(1) < 1e-6] = W_dist[W.sum(1) < 1e-6] + 1e-9
W /= W.sum(1, keepdims=True)
W = smooth(body, W, 4)
write_groups(body, W)

for b in rig.pose.bones:
    b.rotation_mode = "QUATERNION"
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "ab_rigged.blend"))
print("DONE")
