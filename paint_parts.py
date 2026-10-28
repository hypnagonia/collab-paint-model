"""Flat per-part colouring from hand-set rules (read off clay renders with a metric grid), no photo projection.

Usage: blender -b NAME_hy3d.blend -P paint_parts.py -- OUT_DIR [PART_LABELS.npy]
The figure faces -Y, feet at z=0, about 1.85 m tall. Rules are for succubus2; adjust the numbers per model.
"""
import bpy, os, sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
PARTS = argv[1] if len(argv) > 1 else None

obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
NAME = obj.name
n = len(me.vertices)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
nr = np.empty(n * 3, "f"); me.vertex_normals.foreach_get("vector", nr); nr = nr.reshape(-1, 3)
x, y, z = co.T
HX = 0.03  # head centre line
ax = np.abs(x)

# thin sheets (the wrap skirt): a ray going into the surface comes out the other side within 1.5 cm
tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
thin = np.zeros(n, bool)
thin_lock = np.zeros(n, bool)
for i in np.nonzero(z < 1.08)[0]:
    nv = Vector(nr[i])
    thin[i] = tree.ray_cast(Vector(co[i]) - nv * 0.001, -nv, 0.015)[0] is not None
for i in np.nonzero((z > 1.1) & (z < 1.56))[0]:
    nv = Vector(nr[i])
    thin_lock[i] = tree.ray_cast(Vector(co[i]) - nv * 0.001, -nv, 0.03)[0] is not None

# part ids
SKIN, HAIR, HORN, TOP, SKIRT, GLOVES, BOOTS, LIPS, SKIRT_ON_BODY = range(9)
part = np.full(n, SKIN)
hx = np.abs(x - HX)

lab = np.load(PARTS) if PARTS else np.zeros(n, int)  # 1 = hair in the 30-view vote


def capsule(a, b, r):
    a, b = np.array(a), np.array(b)
    t = np.clip(((co - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(co - (a + t[:, None] * (b - a)), axis=1) < r


# arm centre lines straight from the geometry: below the armpit the arm is the cluster outside the torso
def arm_line(sx):
    pts = []
    for zc in np.arange(0.84, 1.26, 0.03):
        m = (np.abs(z - zc) < 0.015) & (x * sx > 0.185) & ~thin
        m &= y > -0.09  # hair locks hang in front of the upper arm
        if m.sum() > 20:
            pts.append((np.median(x[m]), np.median(y[m]), zc))
    pts.append((0.2 * sx, -0.015, 1.36))
    pts.append((0.175 * sx, -0.015, 1.45))  # shoulder joint
    return np.array(pts)


def near_line(pts, r):
    d = np.full(n, np.inf)
    for a_, b_ in zip(pts[:-1], pts[1:]):
        t = np.clip(((co - a_) @ (b_ - a_)) / ((b_ - a_) @ (b_ - a_)), 0, 1)
        d = np.minimum(d, np.linalg.norm(co - (a_ + t[:, None] * (b_ - a_)), axis=1))
    return d < r


upper_arm = np.zeros(n, bool)
fore_arm = np.zeros(n, bool)
arm_tight = np.zeros(n, bool)  # just the limb surface, for telling arms from hair curling next to them
for sx in (-1, 1):
    line = arm_line(sx)
    print("arm", sx, line.round(3).tolist())
    on = near_line(line, 0.06)
    arm_tight |= near_line(line, 0.048 if sx < 0 else 0.06)  # curls hang right next to the right arm only
    upper_arm |= on & (z >= 1.12)
    fore_arm |= on & (z < 1.12)
arm = upper_arm | fore_arm

# head: above the chin everything is hair except the face; the face box comes off the close-up render
part[z > 1.55] = HAIR
face = (z > 1.53) & (z < 1.75) & (x > -0.065) & (x < 0.115) & (y < -0.09) & (nr[:, 1] < -0.2)
part[face] = SKIN
part[face & (lab == 1) & (z > 1.62)] = HAIR  # the lock that falls across the forehead
lips = face & (z > 1.562) & (z < 1.585) & (hx < 0.025) & (y < -0.15)
part[lips] = LIPS
# neck and upper chest: skin unless the vote saw hair there (locks over the collarbones)
mid = (z > 1.12) & (z <= 1.55)
part[mid & (lab == 1) & ~arm] = HAIR
# behind the back and outside the arms the long hair hangs down to ~1.15 m
part[(z > 1.2) & (z <= 1.55) & (y > 0.04) & (nr[:, 1] > 0.2) & (ax < 0.22) & ~arm] = HAIR
part[(z > 1.12) & (z <= 1.55) & (ax > 0.2) & ~arm_tight & ~fore_arm] = HAIR
part[(z > 1.35) & (z <= 1.55) & (y > 0.0) & (nr[:, 1] > 0.3) & (ax < 0.15)] = HAIR  # hair covers the upper back
# hair locks are thin (< 3 cm); arms, neck and shoulders are not
part[thin_lock & (part == SKIN)] = HAIR
# horns: above 1.74 m, off to the sides of the skull, plus the tips above the scalp
part[((z > 1.74) & (hx > 0.078)) | (z > 1.80)] = HORN

# top: around the chest only (not the arms), cleavage V stays skin, plus the straps
top = (z > 1.29) & (z < 1.43) & (ax < 0.18) & ~upper_arm & (part != HAIR) & ~thin_lock
# the back band runs under the hair
top |= (z > 1.28) & (z < 1.34) & (ax < 0.17) & (y > -0.02) & ~upper_arm & ~thin_lock
top &= ~((z > 1.36) & (np.abs(x - 0.005) < 0.018 + (z - 1.36) * 0.5) & (y < -0.08))
part[top] = TOP
part[(z > 1.42) & (z < 1.52) & (np.abs(ax - 0.085) < 0.012) & (part == SKIN)] = TOP

# skirt: the thin wrap anywhere below the waist, and the hip band it is tied over
part[thin & (z < 1.08)] = SKIRT
# where the wrap lies on the legs it is no thinner than the leg: take the 30-view vote's clothing there
part[(lab == 3) & (z < 1.06) & (z > 0.33) & ~fore_arm] = SKIRT
part[(z > 0.86) & (z < 1.05) & (ax < 0.2) & ~thin & ~arm] = SKIRT

# fingerless gloves: the forearm down to the knuckles; fingers stay skin
part[fore_arm & (z > 0.83) & ~thin] = GLOVES

# boots
part[(z < 0.33) & ~thin] = BOOTS

# only the free-hanging sheet is see-through; where the skirt lies on the body (hips, legs) the surface is
# the body itself, so it stays opaque and gets the look of sheer grey over skin
part[(part == SKIRT) & ~thin] = SKIRT_ON_BODY
names = ["skin", "hair", "horn", "top", "skirt", "gloves", "boots", "lips", "skirt_on_body"]
np.save(os.path.join(OUT, NAME + "_part_ids.npy"), part)
print({nm: int((part == k).sum()) for k, nm in enumerate(names)})

# surfaces: sRGB colour from the reference, roughness, alpha
SURF = {
    "skin": ((0.60, 0.38, 0.27), 0.55, 1.0),
    "hair": ((0.24, 0.14, 0.30), 0.38, 1.0),
    "horn": ((0.22, 0.24, 0.30), 0.22, 1.0),
    "top": ((0.15, 0.15, 0.17), 0.8, 1.0),
    "skirt": ((0.36, 0.36, 0.38), 0.6, 0.5),
    "gloves": ((0.13, 0.13, 0.15), 0.5, 1.0),
    "boots": ((0.10, 0.10, 0.11), 0.45, 1.0),
    "lips": ((0.48, 0.24, 0.20), 0.4, 1.0),
    "skirt_on_body": ((0.42, 0.33, 0.29), 0.6, 1.0),
}
lin = lambda c: tuple(v ** 2.2 for v in c)
me.materials.clear()
for nm in names:
    col, rough, alpha = SURF[nm]
    m = bpy.data.materials.new(nm)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*lin(col), 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Alpha"].default_value = alpha
    m.diffuse_color = (*lin(col), 1)
    if alpha < 1:
        m.use_backface_culling = False
    me.materials.append(m)
fi = np.array([np.bincount(part[list(p.vertices)], minlength=len(names)).argmax() for p in me.polygons])
me.polygons.foreach_set("material_index", fi)
me.update()

# previews
s = bpy.context.scene
s.render.engine = "BLENDER_WORKBENCH"
s.display.shading.light = "STUDIO"
s.display.shading.color_type = "MATERIAL"
s.render.film_transparent = True
s.render.resolution_x, s.render.resolution_y = 600, 1000
import math
cam = bpy.data.objects.new("pc", bpy.data.cameras.new("pc")); cam.data.type = "ORTHO"; cam.data.ortho_scale = 2.0
s.collection.objects.link(cam); s.camera = cam
for tag, ang in (("front", 0), ("three_q", 40), ("right", -90), ("back", 180)):
    a = math.radians(ang); cam.location = (5 * math.sin(a), -5 * math.cos(a), 0.95); cam.rotation_euler = (math.pi / 2, 0, a)
    s.render.filepath = os.path.join(OUT, "flat_%s.png" % tag); bpy.ops.render.render(write_still=True)
bpy.data.objects.remove(cam)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_flat.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + "_flat.glb"), export_format="GLB")
print("DONE")
