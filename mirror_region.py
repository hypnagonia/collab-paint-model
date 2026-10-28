"""Replace one box-shaped region of a body with the mirror image of the same region on the other side
(e.g. a shoulder and upper arm that a hair strand left with holes), and weld the patch in.

Faces whose centre lies in the region on the damaged side (side sign, |x| in [XMIN, XMAX], z in [ZMIN, ZMAX];
the free skirt sheet excluded) are deleted; the matching faces of the other side are copied mirrored in X.

Usage: blender -b IN.blend -P mirror_region.py -- BODY_OBJ SIDE XMIN XMAX ZMIN ZMAX [more boxes...] OUT.blend
"""
import bpy, bmesh, sys
import os
from mathutils import Matrix

a = sys.argv[sys.argv.index("--") + 1:]
obj_name, side = a[0], float(a[1])
# one or more boxes: XMIN XMAX ZMIN ZMAX [XMIN XMAX ZMIN ZMAX ...] OUT
vals = [float(v) for v in a[2:-1]]
boxes = [vals[i:i + 4] for i in range(0, len(vals), 4)]
out = a[-1]
xmin = min(b[0] for b in boxes); xmax = max(b[1] for b in boxes)
zmin = min(b[2] for b in boxes); zmax = max(b[3] for b in boxes)
X0 = float(os.environ.get("MIRROR_X0", 0))  # the body's own mid-plane (generated bodies sit a little off x=0)
# a generated body is rarely symmetric about one plane: MIRROR_WARP="XC,R0,R1" puts the mid-line at XC near the centre
# (the cleavage), easing to X0 between R0 and R1 from it (the shoulders and arms)
W = [float(v) for v in os.environ.get("MIRROR_WARP", "").split(",") if v]
XC, R0, R1 = W if W else (X0, 1.0, 2.0)


def plane(x):
    """Mid-line position for a point at x on the kept side."""
    t = min(max((abs(x - XC) - R0) / (R1 - R0), 0.0), 1.0)
    return XC + (X0 - XC) * t


def mirror_x(x):
    return 2 * plane(x) - x
obj = bpy.data.objects[obj_name]
me = obj.data
names = [m.name for m in me.materials]
free = names.index("skirt") if "skirt" in names else -1
skin = names.index("skin")
bm = bmesh.new(); bm.from_mesh(me)


def in_boxes(x, z):
    return any(b[0] <= abs(x - X0) <= b[1] and b[2] <= z <= b[3] for b in boxes)


# MIRROR_KEEP="XMAX,YMAX,ZMIN,ZMAX": a front region (|x-X0| < XMAX, y < YMAX, ZMIN < z < ZMAX) that stays as generated
# on both sides (the cleavage), so the mirror only replaces what is around it
K = [float(v) for v in os.environ.get("MIRROR_KEEP", "").split(",") if v]


HAIR = {i for i, m in enumerate(me.materials) if m.name == "hair"}


def kept(c, mat=None):
    return bool(K) and mat not in HAIR and abs(c.x - X0) < K[0] and c.y < K[1] and K[2] < c.z < K[3]


def inside(f, sgn):
    """Damaged side: beyond the centre line (XC) on the side being replaced, within a box."""
    c = f.calc_center_median()
    return (c.x - XC) * sgn > 0 and f.material_index != free and in_boxes(c.x, c.z) and not kept(c, f.material_index)


def source(f):
    """Kept side whose mirror image lands in the replaced boxes."""
    c = f.calc_center_median()
    if (c.x - XC) * side > 0 or f.material_index == free:
        return False
    m = c.copy(); m.x = mirror_x(c.x)
    return in_boxes(m.x, c.z) and not kept(m)


# cut the body exactly along the centre line first, so no face straddles it: the mirrored half then meets the kept
# half edge to edge instead of overlapping (that overlap crumpled the cleavage)
near = [] if K else [f for f in bm.faces if abs(f.calc_center_median().x - XC) < 0.05 and zmin - 0.02 <= f.calc_center_median().z <= zmax + 0.02]
geom = list({g for f in near for g in list(f.verts) + list(f.edges)}) + near
cut = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(XC, 0, 0), plane_no=(1, 0, 0))
on_plane = [g for g in cut["geom_cut"] if isinstance(g, bmesh.types.BMVert)]
for v in on_plane:
    v.co.x = XC
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
print("bisected the centre: %d vertices on the plane" % len(on_plane))
drop = [f for f in bm.faces if inside(f, side)]
bmesh.ops.delete(bm, geom=drop, context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
src = [f for f in bm.faces if source(f)]
dup = bmesh.ops.duplicate(bm, geom=src)
nv = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]
nf = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMFace)]
for v in nv:
    v.co.x = mirror_x(v.co.x)
bmesh.ops.reverse_faces(bm, faces=nf)
border = list({v for e in bm.edges if e.is_boundary for v in e.verts
               if ((v.co.x - X0) * side > 0 or abs(v.co.x - X0) < 0.02) and xmin - 0.03 <= abs(v.co.x - X0) <= xmax + 0.03
               and zmin - 0.03 <= v.co.z <= zmax + 0.03})
n0 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=border, dist=0.012)
gaps = [e for e in bm.edges if e.is_boundary and all(((v.co.x - X0) * side > 0 or abs(v.co.x - X0) < 0.02)
                                                      and zmin - 0.04 <= v.co.z <= zmax + 0.04
                                                      and xmin - 0.04 <= abs(v.co.x - X0) <= xmax + 0.04 for v in e.verts)]
filled = bmesh.ops.holes_fill(bm, edges=gaps, sides=40)["faces"]
filled = bmesh.ops.triangulate(bm, faces=filled)["faces"]
for f in filled:
    f.material_index = skin
# the centre seam (cleavage, belly, spine): close the slits and smooth out the ridge the two halves leave
band = [] if K else [v for v in bm.verts if abs(v.co.x - XC) < 0.003 and zmin - 0.02 <= v.co.z <= zmax + 0.02]
n1 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=band, dist=0.002)
slit = [e for e in bm.edges if e.is_boundary and all(abs(v.co.x - XC) < 0.03 and zmin - 0.03 <= v.co.z <= zmax + 0.03
                                                      for v in e.verts)]
closed = bmesh.ops.holes_fill(bm, edges=slit, sides=400)["faces"]
closed = bmesh.ops.triangulate(bm, faces=closed)["faces"]
for f in closed:
    f.material_index = skin
band = [] if K else [v for v in bm.verts if abs(v.co.x - XC) < 0.006 and zmin - 0.02 <= v.co.z <= zmax + 0.02 and v.link_faces]
for _ in range(1):
    bmesh.ops.smooth_vert(bm, verts=band, factor=0.3, use_axis_x=False, use_axis_y=True, use_axis_z=False)
print("centre seam: merged %d, closed %d faces, smoothed %d verts" % (n1 - len(bm.verts), len(closed), len(band)))
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
print("region: %d faces out, %d mirrored in, welded %d, filled %d" % (len(drop), len(nf), n0 - len(bm.verts), len(filled)))
bm.to_mesh(me); bm.free()
bpy.ops.wm.save_as_mainfile(filepath=out)
