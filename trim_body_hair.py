"""Clean the merge seam of a body + separately generated head.

Body: its own hair is a cape that ends in a flat cut where the head was swapped. Hair faces standing off the body
(> GAP from the nearest non-hair surface) are deleted; hair faces lying on the body (shoulders, back) are the only
surface there, so they stay and become skin.
Head: the bust's own shoulders/neck below HEAD_ZCUT are dropped (the body has real ones), hair is kept.

Usage: blender -b merged.blend -P trim_body_hair.py -- BODY_OBJ GAP OUT.blend [HEAD_OBJ HEAD_LABELS.npy HEAD_ZCUT]
"""
import bpy, bmesh, sys
import numpy as np
from mathutils.kdtree import KDTree

args = sys.argv[sys.argv.index("--") + 1:]
obj_name, gap, out = args[0], float(args[1]), args[2]
obj = bpy.data.objects[obj_name]
me = obj.data
names = [m.name for m in me.materials]
hair, skin = names.index("hair"), names.index("skin")
bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table()
hair_v = set(v.index for f in bm.faces if f.material_index == hair for v in f.verts)
body_v = [v for v in bm.verts if v.index not in hair_v]
kd = KDTree(len(body_v))
for v in body_v:
    kd.insert(v.co, v.index)
kd.balance()
off = {i: kd.find(bm.verts[i].co)[2] for i in hair_v}
import os
side = float(os.environ.get("TRIM_SIDE", 0))  # only the strand on one side (sign of x); 0 = both
drop = [f for f in bm.faces if f.material_index == hair and max(off[v.index] for v in f.verts) > gap
        and (side == 0 or f.calc_center_median().x * side > 0)
        and f.calc_center_median().z < float(os.environ.get("TRIM_ZMAX", 99))]  # only below this height
bmesh.ops.delete(bm, geom=drop, context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
# loose leftovers of the cape (fins hanging off the shoulders) go; the main body is one big piece
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
# close the holes the cape leaves on the chest, shoulders and back (above the waist: the skirt hems stay open)
upper = [e for e in bm.edges if e.is_boundary and min((obj.matrix_world @ v.co).z for v in e.verts) > 1.1]
filled = bmesh.ops.holes_fill(bm, edges=upper, sides=2000)["faces"]
filled = bmesh.ops.triangulate(bm, faces=filled)["faces"]
for f in filled:
    f.material_index = skin
print("removed %d loose pieces' faces, filled %d hole faces" % (len(small), len(filled)))
kept = 0
for f in bm.faces:
    if f.material_index == hair and (side == 0 or f.calc_center_median().x * side > 0) \
            and f.calc_center_median().z < float(os.environ.get("TRIM_ZMAX", 99)) and not os.environ.get("TRIM_KEEP_HAIR"):
        f.material_index = skin
        kept += 1
bm.to_mesh(me); bm.free()
print("body hair: removed %d faces standing off the body, %d lying on it became skin" % (len(drop), kept))

if len(args) > 3:
    head = bpy.data.objects[args[3]]
    lab = np.load(args[4])
    hz = float(args[5])
    bm = bmesh.new(); bm.from_mesh(head.data)
    layer = bm.verts.layers.int.new("lab")  # labels ride along with the vertices through the delete
    for v in bm.verts:
        v[layer] = int(lab[v.index])
    mw = head.matrix_world
    drop = [f for f in bm.faces if max((mw @ v.co).z for v in f.verts) < hz and all(v[layer] != 1 for v in f.verts)]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    new_lab = np.array([v[layer] for v in bm.verts])
    bm.verts.layers.int.remove(layer)
    bm.to_mesh(head.data); bm.free()
    np.save(args[4].replace(".npy", "_trimmed.npy"), new_lab)
    print("head: removed %d bust faces below %s" % (len(drop), hz))
bpy.ops.wm.save_as_mainfile(filepath=out)
