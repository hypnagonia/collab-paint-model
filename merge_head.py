"""Place a separately generated high-res head (bust) onto the body.

Step "place": blender -b HEAD_hy3d.blend -P merge_head.py -- place BODY_hy3d.blend OUT_DIR
    scales/moves the bust so its nose and chin land on the body's, cuts it below the chin, saves
    OUT_DIR/HEAD_placed.blend (ready for paint_mesh.py) + a landmark check render.
Step "join":  blender -b BODY_final.blend -P merge_head.py -- join HEAD_painted.blend OUT_DIR
    removes the body above the chin and appends the painted head; exports OUT_DIR/merged.glb
"""
import bpy, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_profile import face_landmarks

argv = sys.argv[sys.argv.index("--") + 1:]
STEP, OTHER, OUT = argv[:3]
OVERLAP = 0.05  # how far below the chin the new head reaches (hidden inside the old neck)


def mesh_obj():
    return next(o for o in bpy.context.scene.objects if o.type == "MESH")


def verts(o):
    return np.array([o.matrix_world @ v.co for v in o.data.vertices])


if STEP == "place":
    head = mesh_obj()
    hco = verts(head)
    # the bust comes out at its own scale; search the face in its upper part
    hz0, hz1 = hco[:, 2].min(), hco[:, 2].max()
    if len(argv) > 3:
        # landmarks read off gridded clay renders: "head_chin_x,y,z;body_chin_x,y,z;scale"
        hc_s, bc_s, s_s = argv[3].split(";")
        hc, bc = (np.array([float(v) for v in t.split(",")]) for t in (hc_s, bc_s))
        s = float(s_s)
        hl = bl = None
    else:
        hl = face_landmarks(hco, hz1 - 0.2 * (hz1 - hz0), hz0 + 0.3 * (hz1 - hz0))
        with bpy.data.libraries.load(OTHER) as (src, dst):
            dst.meshes = src.meshes[:1]
        bco = np.array([v.co for v in dst.meshes[0].vertices])
        bz1 = bco[:, 2].max()
        bl = face_landmarks(bco, bz1 - 0.12, bz1 - 0.4)
        hn, hc, bn, bc = (np.array(p) for p in (hl["nose"], hl["chin"], bl["nose"], bl["chin"]))
        s = (bn[2] - bc[2]) / (hn[2] - hc[2])
    t = bc - hc * s
    print("head landmarks", hl, "body", bl, "scale %.4f" % s)
    me = head.data
    for v in me.vertices:
        v.co = (np.array(head.matrix_world @ v.co) * s + t).tolist()
    head.matrix_world.identity()
    # below the cut the bust's own neck, shoulders and chest go; its hair (well clear of the body) stays
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    cut = bc[2] - OVERLAP
    bbm = bmesh.new()
    with bpy.data.libraries.load(OTHER) as (src, dst):
        dst.meshes = src.meshes[:1]
    bbm.from_mesh(dst.meshes[0])
    body_tree = BVHTree.FromBMesh(bbm)
    bm = bmesh.new()
    bm.from_mesh(me)
    drop = []
    for v in bm.verts:
        if v.co.z >= cut:
            continue
        hit = body_tree.find_nearest(v.co)
        if hit[0] is None or hit[3] < 0.012:
            drop.append(v)
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    # loose bits left behind (a strand tip, a piece of strap) go too
    bm.verts.ensure_lookup_table()
    seen, keep = set(), set()
    comps = []
    for v0 in bm.verts:
        if v0.index in seen:
            continue
        comp, stack = [], [v0]
        seen.add(v0.index)
        while stack:
            v = stack.pop()
            comp.append(v)
            for e in v.link_edges:
                w = e.other_vert(v)
                if w.index not in seen:
                    seen.add(w.index)
                    stack.append(w)
        comps.append(comp)
    big = max(len(c) for c in comps)
    bmesh.ops.delete(bm, geom=[v for c in comps if len(c) < 0.01 * big for v in c], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, head.name + "_placed.blend"))
    with open(os.path.join(OUT, "head_cut.txt"), "w") as fh:
        fh.write("%f %f\n" % (cut, bc[2]))
    print("DONE place, cut at z=%.3f" % cut)

elif STEP == "join":
    body = mesh_obj()
    cut, chin = (float(x) for x in open(os.path.join(OUT, "head_cut.txt")).read().split())
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(body.data)
    # the body's head goes from just under the chin up; long hair below that stays
    # optional 4th arg: the body's part ids; its old hair goes entirely (the new head brings its own)
    ids = np.load(argv[3]) if len(argv) > 3 else None
    bm.verts.ensure_lookup_table()
    gone = [v for v in bm.verts if v.co.z > chin - 0.01 or (ids is not None and ids[v.index] == 1)]
    bmesh.ops.delete(bm, geom=gone, context="VERTS")
    bm.to_mesh(body.data)
    bm.free()
    with bpy.data.libraries.load(OTHER) as (src, dst):
        dst.objects = [n for n in src.objects]
    for o in dst.objects:
        if o is not None and o.type == "MESH":
            bpy.context.scene.collection.objects.link(o)
            o.name = "head"
    # the body's old hair is near-black; pull it toward the new head's lighter violet so the seam reads less
    hm = bpy.data.materials.get("hair")
    if hm:
        nt = hm.node_tree
        tex = next(nd for nd in nt.nodes if nd.type == "TEX_IMAGE")
        bsdf = nt.nodes["Principled BSDF"]
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MIX"
        mix.inputs["Factor"].default_value = 0.6
        nt.links.new(tex.outputs["Color"], mix.inputs[6])
        mix.inputs[7].default_value = (0.088, 0.043, 0.133, 1)  # linear of sRGB (0.33, 0.24, 0.40)
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        # glTF can't carry a mix node: bake the tint into a copy of the image for this material
        img = tex.image
        w, h = img.size
        px = np.empty(w * h * 4, "f"); img.pixels.foreach_get(px); px = px.reshape(-1, 4)
        lin = np.clip(px[:, :3], 0, 1) ** 2.2
        lin = lin * 0.4 + np.array([0.088, 0.043, 0.133]) * 0.6
        px[:, :3] = lin ** (1 / 2.2)
        tinted = bpy.data.images.new(img.name + "_hairtint", w, h)
        tinted.pixels.foreach_set(px.ravel())
        tinted.pack()
        tex.image = tinted
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.nodes.remove(mix)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "merged.blend"))
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "merged.glb"), export_format="GLB",
                              export_image_format="JPEG", export_vertex_color="NONE")
    print("DONE join")
