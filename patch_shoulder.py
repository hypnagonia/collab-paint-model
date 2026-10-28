"""Remove the hair strand from one shoulder and rebuild only the surface it covered from the other shoulder.

All hair faces on the strand's side are deleted. Faces of the opposite side whose mirror image lands where hair
was (within REACH) are copied mirrored in X, local faces that the copies now overlap are dropped, and the patch
border is welded. Everything away from the strand (the breasts, the rest of the body) stays as generated.

Usage: blender -b IN.blend -P patch_shoulder.py -- BODY_OBJ STRAND_SIDE(+1|-1) REACH OUT.blend
"""
import bpy, bmesh, sys
from mathutils import Matrix
from mathutils.kdtree import KDTree

obj_name, side, reach, out = sys.argv[sys.argv.index("--") + 1:][:4]
side, reach = float(side), float(reach)
obj = bpy.data.objects[obj_name]
me = obj.data
names = [m.name for m in me.materials]
hair, skin = names.index("hair"), names.index("skin")
free = names.index("skirt") if "skirt" in names else -1
bm = bmesh.new(); bm.from_mesh(me)

strand = [f for f in bm.faces if f.material_index == hair and f.calc_center_median().x * side > 0]
where = [f.calc_center_median().copy() for f in strand]
kd = KDTree(len(where))
for i, c in enumerate(where):
    kd.insert(c, i)
kd.balance()
bmesh.ops.delete(bm, geom=strand, context="FACES")

# copies: faces on the other side (not hair, not the free skirt sheet) whose mirror lands on the strand's place
src = [f for f in bm.faces if f.calc_center_median().x * side < 0 and f.material_index not in (hair, free)]
src = [f for f in src if kd.find((-f.calc_center_median().x, f.calc_center_median().y, f.calc_center_median().z))[2] < reach]
dup = bmesh.ops.duplicate(bm, geom=src)
nv = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]
nf = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMFace)]
bmesh.ops.transform(bm, matrix=Matrix.Scale(-1, 4, (1, 0, 0)), verts=nv)
bmesh.ops.reverse_faces(bm, faces=nf)
# local faces the patch now covers go (no double surface)
kd2 = KDTree(len(nf))
for i, f in enumerate(nf):
    kd2.insert(f.calc_center_median(), i)
kd2.balance()
nfs = set(nf)
covered = [f for f in bm.faces if f not in nfs and f.calc_center_median().x * side > 0 and f.material_index != free
           and kd2.find(f.calc_center_median())[2] < 0.008]
bmesh.ops.delete(bm, geom=covered, context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
border = list({v for e in bm.edges if e.is_boundary for v in e.verts if v.co.x * side > 0})
n0 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=border, dist=0.01)
small = [e for e in bm.edges if e.is_boundary and all(v.co.x * side > 0 and v.co.z > 1.0 for v in e.verts)]
filled = bmesh.ops.holes_fill(bm, edges=small, sides=40)["faces"]
filled = bmesh.ops.triangulate(bm, faces=filled)["faces"]
for f in filled:
    f.material_index = skin
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
print("strand %d faces out, patch %d faces in, %d covered dropped, welded %d, filled %d"
      % (len(strand), len(nf), len(covered), n0 - len(bm.verts), len(filled)))
bm.to_mesh(me); bm.free()
bpy.ops.wm.save_as_mainfile(filepath=out)
