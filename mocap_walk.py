"""Replace the procedural walk with a real motion-capture walk (CMU mocap, cgspeed BVH conversion).

One gait cycle is cut from the clip (left heel strike to the next), made in-place, resampled to CYCLE frames
and retargeted onto the ranger rig as *deltas from the mocap actor's average pose*, so the ranger keeps his
own stance (wide boots, arms clear of the coat) while moving like the actor. Hips are then lowered per frame
so the lowest foot touches the ground.

Usage: blender -b ranger_rigged.blend -P mocap_walk.py -- CLIP.bvh OUT_DIR
"""
import bpy, math, os, sys
import numpy as np
from mathutils import Vector, Matrix, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:]
BVH, OUT = argv[0], argv[1]
CYCLE = 32

scene = bpy.context.scene
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
NAME = rig.name.replace("_rig", "")
pb = rig.pose.bones

# ---------- read the mocap ----------
CLIP_FPS = 1 / float(next(l for l in open(BVH) if l.startswith("Frame Time")).split()[-1])  # CMU 120, Bandai Namco 30
bpy.ops.import_anim.bvh(filepath=BVH, use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)
mc = bpy.context.active_object
mact = mc.animation_data.action
f0, f1 = (int(x) for x in mact.frame_range)
JOINTS = ["Hips", "LowerBack", "Spine1", "Neck", "Neck1", "Head",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase",
          "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]
# Bandai Namco Research Motion dataset names, mapped onto the CMU ones used below:
# (bone, "head" | "tail") per CMU joint
BN = {"Hips": ("Hips", "head"), "LowerBack": ("Spine", "head"), "Spine1": ("Chest", "head"), "Neck": ("Neck", "head"),
      "Neck1": ("Head", "head"), "Head": ("Head", "head"), "HeadTop": ("Head", "tail")}
for side, s in (("Left", "L"), ("Right", "R")):
    BN.update({f"{side}Shoulder": (f"Shoulder_{s}", "head"), f"{side}Arm": (f"UpperArm_{s}", "head"),
               f"{side}ForeArm": (f"LowerArm_{s}", "head"), f"{side}Hand": (f"Hand_{s}", "head"),
               f"{side}FingerBase": (f"Hand_{s}", "tail"), f"{side}UpLeg": (f"UpperLeg_{s}", "head"),
               f"{side}Leg": (f"LowerLeg_{s}", "head"), f"{side}Foot": (f"Foot_{s}", "head"),
               f"{side}ToeBase": (f"Toes_{s}", "head"), f"{side}ToeTip": (f"Toes_{s}", "tail")})
CMU = {j: (j, "head") for j in JOINTS}
CMU.update({"HeadTop": ("Head", "tail"), "LeftToeTip": ("LeftToeBase", "tail"), "RightToeTip": ("RightToeBase", "tail")})
NAMES = BN if "UpperLeg_L" in mc.pose.bones else CMU
P = {j: [] for j in JOINTS + ["HeadTop", "LeftToeTip", "RightToeTip"]}
for f in range(f0, f1 + 1):
    scene.frame_set(f)
    for j in P:
        bone, end = NAMES[j]
        P[j].append(mc.matrix_world @ getattr(mc.pose.bones[bone], end))
P = {k: np.array([tuple(v) for v in vs]) for k, vs in P.items()}
if NAMES is BN:
    # the Bandai Namco end sites have zero length, so the head / hand / toe tips are continued along the bone before
    for tip, a, b in (("HeadTop", "Neck", "Head"), ("LeftFingerBase", "LeftForeArm", "LeftHand"),
                      ("RightFingerBase", "RightForeArm", "RightHand"), ("LeftToeTip", "LeftFoot", "LeftToeBase"),
                      ("RightToeTip", "RightFoot", "RightToeBase")):
        d = P[b] - P[a]
        P[tip] = P[b] + 0.5 * d
bpy.data.objects.remove(mc)
bpy.data.actions.remove(mact)  # else the export carries the raw clip as a second, static animation

# face the walk along -Y (the ranger's forward) and put the ground at z=0
travel = P["Hips"][-1] - P["Hips"][0]
travel[2] = 0
yaw = math.atan2(travel[0], -travel[1])  # angle that takes the travel direction onto -Y
R = np.array(Matrix.Rotation(yaw, 3, "Z"))
for k in P:
    P[k] = P[k] @ R.T
ground = min(P["LeftToeTip"][:, 2].min(), P["RightToeTip"][:, 2].min(), P["LeftFoot"][:, 2].min())
for k in P:
    P[k][:, 2] -= ground
leg_len = np.linalg.norm(P["LeftUpLeg"][0] - P["LeftLeg"][0]) + np.linalg.norm(P["LeftLeg"][0] - P["LeftFoot"][0])

# ---------- one clean cycle: between two left heel strikes in the middle of the clip ----------
fwd = -(P["LeftFoot"][:, 1] - P["Hips"][:, 1])  # how far the left ankle is ahead of the hips
n = len(fwd)
peaks = [0, n - 1] + [i for i in range(3, n - 3) if fwd[i] == fwd[i - 3:i + 4].max() and fwd[i] > 0.25 * leg_len]
peaks = sorted(peaks[2:]) if len(peaks) > 3 else [0, n - 1]
peaks = [p for k, p in enumerate(peaks) if k == 0 or p - peaks[k - 1] > 0.25 * CLIP_FPS]
mid = min(range(len(peaks) - 1), key=lambda k: abs((peaks[k] + peaks[k + 1]) / 2 - n / 2))
a, b = peaks[mid], peaks[mid + 1]
WHOLE = os.environ.get("CLIP_WHOLE")  # not a gait: loop (a window of) the whole clip
if WHOLE:
    trim = int(0.15 * CLIP_FPS)
    span = min(n - 2 * trim, int(float(os.environ.get("CLIP_MAX", 12)) * CLIP_FPS))
    a = (n - span) // 2
    b = a + span - 1
    CYCLE = max(8, round((b - a) / CLIP_FPS * 30))  # keep the clip's own speed at 30 fps
print("clip frames %d, cycle %d..%d (%.2f s), leg %.2f" % (n, a, b, (b - a) / CLIP_FPS, leg_len))


def resample(arr):
    t = np.linspace(a, b, CYCLE + 1)  # both heel strikes, so the loop can be closed below
    i0 = np.floor(t).astype(int)
    w = (t - i0)[:, None]
    return arr[i0] * (1 - w) + arr[np.minimum(i0 + 1, n - 1)] * w


C = {k: resample(v) for k, v in P.items()}
# in-place: remove the steady forward travel (keep the small speed variations)
drift = np.linspace(0, 1, CYCLE + 1)[:, None] * (P["Hips"][b] - P["Hips"][a])
drift[:, 2] = 0
base = P["Hips"][a].copy()
base[2] = 0
ramp = np.linspace(0, 1, CYCLE + 1)[:, None]
if WHOLE:
    # a dance or a gesture wanders: take out the slow travel of the hips (1 s average), keep the sway
    hx = C["Hips"].copy()
    k = max(3, CYCLE // 12)
    pad = np.concatenate([hx[:1].repeat(k, 0), hx, hx[-1:].repeat(k, 0)])
    slow = np.stack([pad[i:i + 2 * k + 1].mean(0) for i in range(len(hx))])
    slow[:, 2] = 0
    slow -= slow[0]
    for key in C:
        C[key] = C[key] - slow
for k in C:
    C[k] = C[k] - drift - base
    # two heel strikes of a real walk never match exactly: spread the mismatch over the cycle so the last
    # frame flows into the first instead of jumping (the head showed it most)
    C[k] = (C[k] - ramp * (C[k][-1] - C[k][0]))[:-1]

# ---------- retarget ----------
SEGMENTS = {  # our bone: (mocap from, mocap to)
    "spine": ("LowerBack", "Spine1"), "chest": ("Spine1", "Neck"), "neck": ("Neck", "Neck1"),
    "head": ("Neck1", "HeadTop"),
}
for s, side in (("L", "Left"), ("R", "Right")):
    SEGMENTS.update({
        f"shoulder.{s}": (f"{side}Shoulder", f"{side}Arm"), f"upper_arm.{s}": (f"{side}Arm", f"{side}ForeArm"),
        f"forearm.{s}": (f"{side}ForeArm", f"{side}Hand"), f"hand.{s}": (f"{side}Hand", f"{side}FingerBase"),
        f"thigh.{s}": (f"{side}UpLeg", f"{side}Leg"), f"shin.{s}": (f"{side}Leg", f"{side}Foot"),
        f"foot.{s}": (f"{side}Foot", f"{side}ToeTip"),
    })
ORDER = ["spine", "chest", "neck", "head"] + [f"{p}.{s}" for s in "LR" for p in
                                              ("shoulder", "upper_arm", "forearm", "hand", "thigh", "shin", "foot")]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


seg_dir = {b: unit(C[t] - C[f]) for b, (f, t) in SEGMENTS.items()}
seg_mean = {b: Vector(unit(d.mean(0))) for b, d in seg_dir.items()}


def hip_basis(i):
    lat = C["LeftUpLeg"][i] - C["RightUpLeg"][i]
    up = C["LowerBack"][i] - C["Hips"][i]
    up = up / np.linalg.norm(up)
    lat = lat - up * (lat @ up)
    lat = lat / np.linalg.norm(lat)
    return Matrix((lat, np.cross(up, lat), up)).transposed()  # columns: x=left, y=back, z=up


hip_mean = Matrix(np.mean([np.array(hip_basis(i)) for i in range(CYCLE)], axis=0)).to_quaternion().to_matrix()

# drop the procedural animation and the leg IK: the mocap drives every bone directly
for b in pb:
    for c in list(b.constraints):
        b.constraints.remove(c)
    b.location = (0, 0, 0)
    b.rotation_quaternion = (1, 0, 0, 0)
rig.animation_data_clear()
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
hip_rest_z = rest["hips"].translation.z
mc_hip_mean_z = C["Hips"][:, 2].mean()
scale = (rest["thigh.L"].translation.z - rest["foot.L"].translation.z) / leg_len  # character hip-to-ankle over actor's
ANKLE_H = rest["foot.L"].translation.z


# upper-body sway is exaggerated on a different body: keep only part of it
DAMP = {"spine": 0.5, "chest": 0.4, "neck": 0.4, "head": 0.35, "shoulder": 0.3, "upper_arm": 0.85}
KNEE_REST_FLEX = {s: min(Vector(seg_dir[f"thigh.{s}"][i]).angle(Vector(seg_dir[f"shin.{s}"][i])) for i in range(CYCLE))
                  for s in "LR"}
KNEE_REST_FLEX_OURS = {s: rest[f"thigh.{s}"].col[1].xyz.angle(rest[f"shin.{s}"].col[1].xyz) for s in "LR"}


def aim(name, target_dir):
    """Rotate bone (in armature space, about its head) so its Y axis points along target_dir."""
    bone = pb[name]
    t = Vector(target_dir)
    if t.length < 1e-6 or t.length != t.length:  # degenerate direction (zero-length mocap segment): leave as is
        return
    m = bone.matrix.copy()
    rot = m.to_quaternion().normalized()  # rotation only: rebuilding from it keeps scale at 1 (it crept up to 20x)
    cur = (rot @ Vector((0, 1, 0))).normalized()
    q = cur.rotation_difference(t.normalized()).normalized()
    bone.matrix = Matrix.Translation(m.translation) @ (q @ rot).to_matrix().to_4x4()
    bone.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def lowest_foot():
    zs = []
    for s in "LR":
        ankle = rig.matrix_world @ pb[f"foot.{s}"].head
        toe = rig.matrix_world @ pb[f"foot.{s}"].tail
        zs += [ankle.z - (ANKLE_H - 0.01), toe.z - 0.025]
    return min(zs)


scene.frame_start, scene.frame_end = 1, CYCLE
scene.render.fps = 30
plant = []
for i in range(CYCLE + 1):
    k = i % CYCLE
    scene.frame_set(i + 1)
    for b in pb:
        b.location = (0, 0, 0)
        b.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    # hips: actor's orientation change, height change scaled to the ranger, sideways sway kept
    dq = Quaternion().slerp((hip_basis(k) @ hip_mean.inverted()).to_quaternion(), 0.7)
    hip = pb["hips"]
    off = Vector(((C["Hips"][k, 0]) * scale, (C["Hips"][k, 1]) * scale, (C["Hips"][k, 2] - mc_hip_mean_z) * scale))
    m = rest["hips"].copy()
    hip.matrix = Matrix.Translation(m.translation + off) @ dq.to_matrix().to_4x4() @ Matrix.Translation(-m.translation) @ m
    bpy.context.view_layer.update()
    for name in ORDER:
        rest_dir = rest[name].col[1].xyz.normalized()
        if name.startswith("shin."):
            # the knee is a hinge: bend the shin off the thigh's current line by the actor's knee angle, always forward
            # (aiming it on its own lets a straightened stance leg overshoot into a backward knee)
            s = name[-1]
            thigh = pb[f"thigh.{s}"].matrix.col[1].xyz.normalized()
            flex = Vector(seg_dir[f"thigh.{s}"][k]).angle(Vector(seg_dir[name][k]))
            flex = max(flex - KNEE_REST_FLEX[s], 0.0) + KNEE_REST_FLEX_OURS[s]
            lateral = Vector((1, 0, 0))
            a = Quaternion(lateral, flex) @ thigh
            b = Quaternion(lateral, -flex) @ thigh
            aim(name, a if a.y > b.y else b)  # +y is backward: the foot swings back, the knee stays in front
            continue
        delta = seg_mean[name].rotation_difference(Vector(seg_dir[name][k]))
        delta = Quaternion().slerp(delta, DAMP.get(name.split(".")[0], 1.0))
        aim(name, delta @ rest_dir)
    # plant: lowest point of either boot on the ground
    plant.append(-lowest_foot())
    hip.matrix = Matrix.Translation((0, 0, plant[-1])) @ hip.matrix
    bpy.context.view_layer.update()
    for name in ["hips"] + ORDER:
        pb[name].keyframe_insert("rotation_quaternion")
    hip.keyframe_insert("location")

# the planted height jumps when support passes from one boot to the other; smooth it around the loop so the head
# doesn't jerk at each step (a few mm of sole sinking into the ground is the price)
raw = np.array(plant[:CYCLE])
kern = np.array([1, 4, 6, 4, 1], float) / 16
smooth_h = sum(w * np.roll(raw, s) for w, s in zip(kern, range(-2, 3)))
smooth_h = sum(w * np.roll(smooth_h, s) for w, s in zip(kern, range(-2, 3)))
for i in range(CYCLE + 1):
    scene.frame_set(i + 1)
    bpy.context.view_layer.update()
    hip.matrix = Matrix.Translation((0, 0, smooth_h[i % CYCLE] - raw[i % CYCLE])) @ hip.matrix
    bpy.context.view_layer.update()
    hip.keyframe_insert("location")

act = rig.animation_data.action
act.name = os.environ.get("ACTION_NAME", "walk_mocap")
bpy.ops.object.mode_set(mode="OBJECT")
scene.frame_end = CYCLE + 1  # frame CYCLE+1 repeats frame 1: the exported clip ends where it starts
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_rigged.blend"))
bpy.ops.object.select_all(action="DESELECT")
for o in bpy.data.objects:
    if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers):
        o.select_set(True)
rig.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + "_walk.glb"), export_format="GLB", use_selection=True,
                          export_image_format="JPEG", export_animations=True, export_force_sampling=True,
                          export_frame_range=True, export_vertex_color="NONE")
print("DONE")
