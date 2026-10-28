"""Humanoid armature + distance-based skin weights + procedural in-place walk cycle for the painted ranger.

Character faces -Y, +X is the character's left (.L). Joint positions were read off the measured front/side renders.
"""
import bpy, math, os, sys
import numpy as np
from mathutils import Vector, Quaternion, Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
BLEND, OUT = argv[:2]
CYCLE = 32  # frames per walk cycle

bpy.ops.wm.open_mainfile(filepath=BLEND)
mesh = bpy.data.objects["ranger"]
scene = bpy.context.scene

# ---------- armature ----------
# name: (head, tail, parent, deform, limb radius limit for skinning or None = always a candidate)
J = {}
for s, sx in (("L", 1), ("R", -1)):
    J.update({
        f"shoulder.{s}": ((0.04 * sx, -0.10, 1.47), (0.22 * sx, -0.08, 1.44), "chest", True, 0.10),
        f"upper_arm.{s}": ((0.24 * sx, -0.07, 1.43), (0.35 * sx, -0.05, 1.15), f"shoulder.{s}", True, 0.10),
        f"forearm.{s}": ((0.35 * sx, -0.05, 1.15), (0.33 * sx, -0.13, 0.92), f"upper_arm.{s}", True, 0.085),
        f"hand.{s}": ((0.33 * sx, -0.13, 0.92), (0.31 * sx, -0.16, 0.79), f"forearm.{s}", True, 0.07),
        f"thigh.{s}": ((0.10 * sx, -0.07, 0.93), (0.16 * sx, -0.10, 0.52), "hips", True, 0.13),
        f"shin.{s}": ((0.16 * sx, -0.10, 0.52), (0.19 * sx, -0.06, 0.11), f"thigh.{s}", True, 0.12),
        f"foot.{s}": ((0.19 * sx, -0.06, 0.11), (0.21 * sx, -0.23, 0.03), f"shin.{s}", True, 0.11),
        f"foot_ik.{s}": ((0.19 * sx, -0.06, 0.11), (0.21 * sx, -0.23, 0.03), "root", False, None),  # same rest as foot
        f"knee_pole.{s}": ((0.16 * sx, -0.70, 0.52), (0.16 * sx, -0.80, 0.52), "root", False, None),
    })
J.update({
    "root": ((0, 0, 0), (0, 0.25, 0), None, False, None),
    "hips": ((0, -0.08, 0.95), (0, -0.08, 1.10), "root", True, "torso"),
    "spine": ((0, -0.08, 1.10), (0, -0.09, 1.30), "hips", True, "torso"),
    "chest": ((0, -0.09, 1.30), (0, -0.11, 1.52), "spine", True, "torso"),
    "neck": ((0, -0.11, 1.52), (0, -0.13, 1.62), "chest", True, 0.13),
    "head": ((0, -0.13, 1.62), (0, -0.13, 1.85), "neck", True, 0.24),
})

arm_data = bpy.data.armatures.new("ranger_rig")
rig = bpy.data.objects.new("ranger_rig", arm_data)
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

# ---------- skin weights (segment distance, limb bones only near their own geometry) ----------
co = np.array([v.co for v in mesh.data.vertices], dtype=np.float64)
deform = [(n, np.array(h), np.array(t), lim) for n, (h, t, _, d, lim) in J.items() if d]


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


# distance weights: only a fallback for vertices bone heat can't solve
D = np.stack([seg_dist(co, a, b) for _, a, b, _ in deform], 1)
W_dist = 1.0 / (D + 0.015) ** 5
for k, (n, _, _, lim) in enumerate(deform):
    if lim != "torso":
        W_dist[:, k] *= np.clip((lim - D[:, k]) / (0.3 * lim), 0, 1)

# Blender's bone heat weighting (respects the surface, so arms don't drag the coat beside them)
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
names = [d[0] for d in deform]
W = np.zeros((len(co), len(names)))
gi = {g.index: names.index(g.name) for g in mesh.vertex_groups if g.name in names}
for v in mesh.data.vertices:
    for g in v.groups:
        if g.group in gi:
            W[v.index, gi[g.group]] = g.weight
miss = W.sum(1) < 1e-3
print("heat-unsolved vertices: %d of %d" % (miss.sum(), len(co)))
W[miss] = W_dist[miss]

# overrides: legs don't own the cloak hanging behind/beside them, arms don't own the cloak outside them
ax, z = np.abs(co[:, 0]), co[:, 2]
# boots splay out wider than the knees, but only the toes reach that far; the cloak corners beside them don't
leg_x = np.where((z < 0.3) & (co[:, 1] < -0.04), 0.42, np.where(z < 0.3, 0.30, 0.32))
leg_zone = np.clip((0.14 - co[:, 1]) / 0.04, 0, 1) * np.clip((leg_x - ax) / 0.05, 0, 1)
# the cloak hangs behind and outside the arms: arms own nothing behind their back surface or past their outer side
arm_zone = np.clip((0.43 - ax) / 0.03, 0, 1) * np.clip((0.03 - co[:, 1]) / 0.04, 0, 1)
# below the elbow the coat skirt sits right next to the fist: forearm/hand keep only what's outside it
# (the coat's side wall sits at |x| 0.25-0.29 and a little behind the fist, so the fist owns only what's
# clearly outside that wall or in front of it)
outside = np.clip((ax - 0.27) / 0.03, 0, 1)
in_front = np.clip((ax - 0.19) / 0.02, 0, 1) * np.clip((-0.07 - co[:, 1]) / 0.03, 0, 1)
low_arm_zone = arm_zone * np.where(z < 1.1, np.maximum(outside, in_front), 1.0)
# the pouch on the left hip sits right in front of the fist: it belongs to the hips
pouch = (co[:, 0] > 0.08) & (co[:, 0] < 0.27) & (z > 0.92) & (z < 1.16) & (co[:, 1] < -0.04)
low_arm_zone[pouch] = 0
for k, n in enumerate(names):
    part = n.split(".")[0]
    if part in ("thigh", "shin", "foot"):
        W[:, k] *= leg_zone
    elif part == "upper_arm":
        W[:, k] *= arm_zone
    elif part == "forearm":
        W[:, k] *= low_arm_zone * np.clip((0.1 - D[:, k]) / 0.03, 0, 1)
    elif part == "hand":
        # the fist's mesh is fused to the coat, so own exactly a fist-sized ellipsoid and nothing else
        sx = 1 if n.endswith(".L") else -1
        e = np.sqrt(((co[:, 0] - 0.33 * sx) / 0.058) ** 2 + ((co[:, 1] + 0.15) / 0.065) ** 2
                    + ((co[:, 2] - 0.855) / 0.085) ** 2)
        W[:, k] = np.maximum(W[:, k], 0.6) * np.clip((1.15 - e) / 0.15, 0, 1) * arm_zone
lost = W.sum(1) < 1e-3  # e.g. cloak hem that only had leg weight: hand it to the torso
torso = [k for k, d in enumerate(deform) if d[3] == "torso"]
W[np.ix_(lost, torso)] = W_dist[np.ix_(lost, torso)] + 1e-6
W /= W.sum(1, keepdims=True)

# Laplacian smoothing of the weights along mesh edges
ed = np.empty(len(mesh.data.edges) * 2, np.int64)
mesh.data.edges.foreach_get("vertices", ed)
ed = ed.reshape(-1, 2)
deg = np.bincount(ed.ravel(), minlength=len(co)).astype(np.float64)[:, None] + 1e-9
for _ in range(4):
    acc = np.zeros_like(W)
    np.add.at(acc, ed[:, 0], W[ed[:, 1]])
    np.add.at(acc, ed[:, 1], W[ed[:, 0]])
    W = 0.5 * W + 0.5 * acc / deg
W /= W.sum(1, keepdims=True) + 1e-12

# The generated mesh fuses forearm/fist to the coat and the boots to the cloak hem, so any weight split
# stretches a membrane between them in motion. Cut the surface there instead: faces owned by the lower arm
# or lower leg are detached from the faces owned by the body (the seams sit on inner sides, out of view).
import bmesh
part_of = [n.split(".")[0] for n in names]
arm_k = [k for k, p in enumerate(part_of) if p in ("upper_arm", "forearm", "hand")]
low_k = [k for k, p in enumerate(part_of) if p in ("forearm", "hand")]
leg_k = [k for k, p in enumerate(part_of) if p in ("thigh", "shin", "foot")]
shin_k = [k for k, p in enumerate(part_of) if p in ("shin", "foot")]
ARM_CUT, LEG_CUT = 1.12, 0.5
cat_v = np.where(W[:, arm_k].sum(1) > 0.5, 1, np.where(W[:, leg_k].sum(1) > 0.5, 2, 0))
bm = bmesh.new()
bm.from_mesh(mesh.data)
orig = bm.verts.layers.int.new("orig")
fcat = bm.faces.layers.int.new("part")
for v in bm.verts:
    v[orig] = v.index
for f in bm.faces:
    f[fcat] = int(np.bincount([cat_v[v.index] for v in f.verts], minlength=3).argmax())
# majority-vote smoothing of the face labels so no tiny islands get cut loose
bm.faces.ensure_lookup_table()
nbr = [[e.link_faces[0].index if e.link_faces[0] is not f else e.link_faces[1].index
        for e in f.edges if len(e.link_faces) == 2] for f in bm.faces]
lab = np.array([f[fcat] for f in bm.faces])
for _ in range(6):
    new = lab.copy()
    for i, ns in enumerate(nbr):
        votes = np.bincount(lab[ns + [i]], minlength=3)
        if votes.max() * 2 > len(ns) + 1:
            new[i] = votes.argmax()
    lab = new
# any small island of one label (a sole edge marked as body, a cloak shred marked as leg) takes the label
# of what surrounds it
seen = np.zeros(len(lab), bool)
for i0 in range(len(lab)):
    if seen[i0]:
        continue
    comp, stack, border = [], [i0], []
    seen[i0] = True
    while stack:
        i = stack.pop()
        comp.append(i)
        for j in nbr[i]:
            if lab[j] != lab[i0]:
                border.append(lab[j])
            elif not seen[j]:
                seen[j] = True
                stack.append(j)
    if len(comp) < 400 and border:
        lab[comp] = np.bincount(border, minlength=3).argmax()
for f in bm.faces:
    f[fcat] = int(lab[f.index])


def should_cut(e):
    if len(e.link_faces) != 2:
        return False
    a, b = e.link_faces[0][fcat], e.link_faces[1][fcat]
    if a == b:
        return False
    zc = (e.verts[0].co.z + e.verts[1].co.z) / 2
    return zc < (ARM_CUT if 1 in (a, b) else LEG_CUT)


cut = [e for e in bm.edges if should_cut(e)]
bmesh.ops.split_edges(bm, edges=cut)
# whatever still comes loose as a small island goes back to moving with the body
bm.faces.ensure_lookup_table()
seen = set()
for f in bm.faces:
    if f.index in seen or f[fcat] == 0:
        continue
    comp, stack = [], [f]
    seen.add(f.index)
    while stack:
        g = stack.pop()
        comp.append(g)
        for e in g.edges:
            for h in e.link_faces:
                if h.index not in seen and h[fcat] == g[fcat]:
                    seen.add(h.index)
                    stack.append(h)
    if len(comp) < 400:
        for g in comp:
            g[fcat] = 0
print("cut %d edges between limbs and clothing" % len(cut))
src = np.array([v[orig] for v in bm.verts])
cats = [{f[fcat] for f in v.link_faces} for v in bm.verts]
vz = np.array([v.co.z for v in bm.verts])
bm.to_mesh(mesh.data)
bm.free()
only = lambda c: np.array([s == {c} for s in cats])
W = W[src]
other = lambda keep: [k for k in range(len(names)) if k not in keep]
W[np.ix_(only(1) & (vz < ARM_CUT), other(arm_k))] = 0  # detached arm pieces move only with the arm
W[np.ix_(only(2) & (vz < LEG_CUT), other(leg_k))] = 0  # detached boots/shins only with the leg
W[np.ix_(only(0) & (vz < ARM_CUT), low_k)] = 0  # the coat doesn't follow the forearm/fist
W[np.ix_(only(0) & (vz < LEG_CUT), shin_k)] = 0  # and the cloak hem doesn't follow the boots
empty = W.sum(1) < 1e-6
for i in np.nonzero(empty)[0]:  # anything left without weight takes the nearest bone by distance
    W[i] = W_dist[src[i]]
W /= W.sum(1, keepdims=True) + 1e-12
for g in list(mesh.vertex_groups):
    mesh.vertex_groups.remove(g)
for k, n in enumerate(names):
    vg = mesh.vertex_groups.new(name=n)
    w = W[:, k]
    for i in np.nonzero(w > 0.002)[0]:
        vg.add([int(i)], float(w[i]), "REPLACE")
if not any(m.type == "ARMATURE" for m in mesh.modifiers):
    mesh.modifiers.new("rig", "ARMATURE").object = rig

# ---------- IK legs ----------
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
pb = rig.pose.bones
for b in pb:
    b.rotation_mode = "QUATERNION"
for s in ("L", "R"):
    ik = pb[f"shin.{s}"].constraints.new("IK")
    ik.target, ik.subtarget = rig, f"foot_ik.{s}"
    ik.pole_target, ik.pole_subtarget = rig, f"knee_pole.{s}"
    ik.chain_count = 2
    cr = pb[f"foot.{s}"].constraints.new("COPY_ROTATION")
    cr.target, cr.subtarget = rig, f"foot_ik.{s}"
    cr.mix_mode = "REPLACE"


# pick the pole angle that keeps the knee pointing forward
def knee_y(s):
    bpy.context.view_layer.update()
    return (rig.matrix_world @ pb[f"shin.{s}"].head).y, (rig.matrix_world @ pb[f"shin.{s}"].head).x


for s, sx in (("L", 1), ("R", -1)):
    pb[f"foot_ik.{s}"].location = (0, 0, 0)
    best = None
    for ang in range(-180, 180, 15):
        pb[f"shin.{s}"].constraints["IK"].pole_angle = math.radians(ang)
        pb[f"foot_ik.{s}"].location = rig.data.bones[f"foot_ik.{s}"].matrix_local.to_3x3().inverted() @ Vector((0, 0, 0.25))
        y, x = knee_y(s)
        score = -y - abs(x - 0.16 * sx)  # forward and not splayed
        best = max(best or (score, ang), (score, ang))
    pb[f"shin.{s}"].constraints["IK"].pole_angle = math.radians(best[1])
    pb[f"foot_ik.{s}"].location = (0, 0, 0)
    print("pole", s, best[1])


# ---------- walk cycle ----------
def local_axis(bone, world_axis):
    return (rig.data.bones[bone].matrix_local.to_3x3().inverted() @ Vector(world_axis)).normalized()


def rot(bone, *pairs):
    q = Quaternion()
    for axis, deg in pairs:
        q = Quaternion(local_axis(bone, axis), math.radians(deg)) @ q
    pb[bone].rotation_quaternion = q
    pb[bone].keyframe_insert("rotation_quaternion")


def loc(bone, world_delta):
    pb[bone].location = rig.data.bones[bone].matrix_local.to_3x3().inverted() @ Vector(world_delta)
    pb[bone].keyframe_insert("location")


def smooth(x):
    return x * x * (3 - 2 * x)


STRIDE_HALF = 0.25  # foot travel either side of the hip during stance
LIFT = 0.12
ANKLE = 0.11


def foot(phase):
    """Foot relative to rest for a phase in [0,1): 0 = heel strike, 0.6 = toe off.
    Returns (forward offset, ankle height, pitch deg; + = heel up)."""
    if phase < 0.6:
        s = phase / 0.6
        fwd = STRIDE_HALF - 2 * STRIDE_HALF * s
        if s < 0.15:
            pitch = -15 * (1 - smooth(s / 0.15))
        elif s < 0.65:
            pitch = 0
        else:
            pitch = 38 * smooth((s - 0.65) / 0.35)
        lift = 0
    else:
        w = (phase - 0.6) / 0.4
        fwd = -STRIDE_HALF + 2 * STRIDE_HALF * smooth(w)
        pitch = 38 * (1 - smooth(min(1, w / 0.45))) if w < 0.45 else -15 * smooth((w - 0.45) / 0.55)
        lift = LIFT * math.sin(math.pi * w ** 0.85)
    # rolling over the toe (heel up) or the heel (toes up) raises the ankle
    roll = 0.15 * math.sin(math.radians(pitch)) if pitch > 0 else 0.06 * math.sin(math.radians(-pitch))
    return fwd, ANKLE + lift + roll, pitch


rest = {s: Vector(J[f"foot_ik.{s}"][0]) for s in ("L", "R")}
scene.frame_start, scene.frame_end = 1, CYCLE
scene.render.fps = 30  # one cycle = 32/30 s, matches fps_suggested for the 16-frame sprites
for f in range(CYCLE + 1):
    t = f / CYCLE
    scene.frame_set(f + 1)
    for s, sx, off in (("L", 1, 0.0), ("R", -1, 0.5)):
        fwd, z, pitch = foot((t + off) % 1)
        target = Vector((0.11 * sx, -0.06 - fwd, z))
        loc(f"foot_ik.{s}", target - rest[s])
        rot(f"foot_ik.{s}", ((1, 0, 0), pitch))  # +X rotation tips the toes down
        # arms swing opposite to the leg on the same side; elbow bends more on the forward swing
        swing = math.cos(2 * math.pi * (t + off + 0.5))
        rot(f"upper_arm.{s}", ((1, 0, 0), -20 * swing), ((0, 1, 0), -3 * sx))  # -X rotation swings forward
        rot(f"forearm.{s}", ((1, 0, 0), -(12 + 14 * max(0.0, swing))))
        rot(f"hand.{s}", ((1, 0, 0), -4 * swing))
    bob = math.cos(4 * math.pi * (t - 0.3))  # high at mid-stance, low at heel strikes
    sway = math.cos(2 * math.pi * (t - 0.3))  # over the stance foot (left first)
    loc("hips", (0.018 * sway, 0, -0.035 + 0.022 * bob))
    yaw = math.cos(2 * math.pi * t)  # left hip forward at left heel strike
    rot("hips", ((0, 0, 1), -6 * yaw), ((0, 1, 0), -3 * sway))  # swing-side hip drops
    rot("spine", ((0, 0, 1), 5 * yaw), ((1, 0, 0), 3), ((0, 1, 0), 1.5 * sway))
    rot("chest", ((0, 0, 1), 6 * yaw), ((1, 0, 0), 1 + 1.5 * bob), ((0, 1, 0), 1.5 * sway))
    rot("neck", ((0, 0, 1), -3 * yaw))
    rot("head", ((0, 0, 1), -2 * yaw), ((1, 0, 0), -1.5 * bob))

act = rig.animation_data.action
act.name = "walk"
for fc in getattr(act, "fcurves", []):
    for k in fc.keyframe_points:
        k.interpolation = "LINEAR"
scene.frame_end = CYCLE
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "ranger_rigged.blend"))
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
rig.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "ranger_walk.glb"), export_format="GLB", use_selection=True,
                          export_image_format="JPEG", export_animations=True, export_force_sampling=True,
                          export_frame_range=True, export_vertex_color="NONE")
print("DONE")
