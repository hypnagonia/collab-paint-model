"""Mirror the upper body (torso above the waist and both arms) from one side to the other, so a hair cape on one
shoulder disappears and both shoulders and arms match. The skirt below stays as generated (it is asymmetric on
purpose: a wrap with a slit and a hanging panel).

Faces of the upper region on the side being replaced are deleted, the kept side's faces are duplicated and mirrored
in X (materials and UVs come along, so the paint is mirrored too), and the seam at the waist / mid-line is welded.

Usage: blender -b IN.blend -P symmetrize_upper.py -- BODY_OBJ KEEP_SIDE(+1|-1) WAIST_Z OUT.blend
(WAIST_Z 0: mirror the whole body except the free skirt sheet)
"""
import bpy, bmesh, sys
from mathutils import Matrix

args = sys.argv[sys.argv.index("--") + 1:]
obj_name, keep, waist, out = args[0], float(args[1]), float(args[2]), args[3]
obj = bpy.data.objects[obj_name]
me = obj.data
bm = bmesh.new(); bm.from_mesh(me)


FREE = {i for i, m in enumerate(me.materials) if m.name == "skirt"}


def upper(f):
    if waist <= 0:  # whole body except the free hanging sheet of the skirt, which keeps its asymmetric cut
        return f.material_index not in FREE
    c = f.calc_center_median()
    return c.z > waist or (abs(c.x) > 0.19 and c.z > 0.7)  # the torso above the waist, and the arms down to the hands


drop = [f for f in bm.faces if upper(f) and f.calc_center_median().x * keep < 0]
# below the waist the replaced side keeps its own (asymmetric) skirt, but skin/bracer bits there that have no mirror
# partner on the kept side are leftovers of the old fist-to-hip bridge: they go too
from mathutils.kdtree import KDTree
kept_v = [v for v in bm.verts if v.co.x * keep > 0]
kd = KDTree(len(kept_v))
for v in kept_v:
    kd.insert(v.co, v.index)
kd.balance()
names = [m.name for m in me.materials]
body_mats = {names.index(n) for n in ("skin", "gloves") if n in names}
bridge = []
for f in bm.faces:
    c = f.calc_center_median()
    if waist > 0 and c.x * keep < 0 and not upper(f) and c.z > 0.7 and f.material_index in body_mats:
        if kd.find((-c.x, c.y, c.z))[2] > 0.015:
            bridge.append(f)
drop += bridge
print("bridge leftovers without a mirror partner: %d faces" % len(bridge))
bmesh.ops.delete(bm, geom=drop, context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
src = [f for f in bm.faces if upper(f) and f.calc_center_median().x * keep >= 0]
dup = bmesh.ops.duplicate(bm, geom=src)
new_v = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]
new_f = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMFace)]
bmesh.ops.transform(bm, matrix=Matrix.Scale(-1, 4, (1, 0, 0)), verts=new_v)
bmesh.ops.reverse_faces(bm, faces=new_f)
# weld the seam: open edges near the mid-line and at the waist find their partners
seam = list({v for e in bm.edges if e.is_boundary for v in e.verts})
before = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=seam, dist=0.012)
print("welded %d seam vertices" % (before - len(bm.verts)))
# close what is still open along the seam (small loops only)
loops = [e for e in bm.edges if e.is_boundary and (waist <= 0 and abs(sum(v.co.x for v in e.verts)) < 0.06
                                                      or waist > 0 and min(v.co.z for v in e.verts) > waist - 0.06)]
filled = bmesh.ops.holes_fill(bm, edges=loops, sides=60)["faces"]
filled = bmesh.ops.triangulate(bm, faces=filled)["faces"]
skin = [m.name for m in me.materials].index("skin")
for f in filled:
    f.material_index = skin
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
# pieces cut loose by the swap (the old fist-to-hip bridge on the replaced side) go
bm.faces.ensure_lookup_table()
seen, small = set(), []
for f0 in bm.faces:
    if f0.index in seen:
        continue
    comp, stack = [], [f0]
    seen.add(f0.index)
    while stack:
        f = stack.pop(); comp.append(f)
        for e in f.edges:
            for g in e.link_faces:
                if g.index not in seen:
                    seen.add(g.index); stack.append(g)
    if len(comp) < 3000:
        small += comp
bmesh.ops.delete(bm, geom=small, context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
print("dropped %d faces in loose pieces" % len(small))
print("removed %d faces, mirrored %d, filled %d" % (len(drop), len(new_f), len(filled)))
bm.to_mesh(me); bm.free()
bpy.ops.wm.save_as_mainfile(filepath=out)
