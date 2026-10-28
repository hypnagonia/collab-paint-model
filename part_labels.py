"""3D part labels for a painted mesh: render many views, segment each render, vote labels onto vertices.

Step "render": blender -b MODEL_painted.blend -P part_labels.py -- render OUT_DIR
    writes OUT_DIR/views/view_XX.png + views.json (orthographic cameras)
Step "segment" (outside Blender): python segment_refs.py OUT_DIR/views/view_*.png  -> view_XX_seg.npy
    (segment_refs.py treats any *.png; the labels are 0 none, 1 hair, 2 skin, 3 clothing, 4 shoes)
Step "vote":   blender -b MODEL_painted.blend -P part_labels.py -- vote OUT_DIR
    writes OUT_DIR/part_labels.npy (one label per vertex of the painted mesh) + preview renders
"""
import bpy, json, math, os, sys
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
STEP, OUT = argv[0], argv[1]
VIEWS = os.path.join(OUT, "views")
SIZE = 768
ORTHO = float(os.environ.get("PL_ORTHO", 2.2))  # frame width in metres; smaller for a head-only mesh
CLASSES = ["none", "hair", "skin", "clothing", "shoes"]
COLORS = np.array([[0.2, 0.2, 0.2], [0.85, 0.2, 0.65], [0.95, 0.75, 0.6], [0.25, 0.55, 0.85], [0.5, 0.35, 0.15]])

scene = bpy.context.scene
obj = next(o for o in scene.objects if o.type == "MESH")
me = obj.data
zmax = max(v.co.z for v in me.vertices)
zmin = min(v.co.z for v in me.vertices)


def cameras():
    out = []
    for elev in (-15, 10, 35):
        for k in range(10):
            az = k * 36 + (18 if elev == 10 else 0)
            out.append({"az": az, "elev": elev})
    return out


def cam_matrix(az, elev):
    """World matrix of an orthographic camera looking at the figure's centre from (az, elev)."""
    a, e = math.radians(az), math.radians(elev)
    centre = Vector((0, 0, (zmin + zmax) / 2))
    d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))  # from centre to camera
    rot = (-d).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(centre + d * 6) @ rot


if STEP == "render":
    os.makedirs(VIEWS, exist_ok=True)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    cam = bpy.data.objects.new("pl_cam", bpy.data.cameras.new("pl_cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ORTHO
    scene.collection.objects.link(cam)
    scene.camera = cam
    cams = cameras()
    for i, c in enumerate(cams):
        cam.matrix_world = cam_matrix(c["az"], c["elev"])
        scene.render.filepath = os.path.join(VIEWS, "view_%02d.png" % i)
        bpy.ops.render.render(write_still=True)
    json.dump(cams, open(os.path.join(VIEWS, "views.json"), "w"))
    print("DONE render", len(cams))

elif STEP == "vote":
    cams = json.load(open(os.path.join(VIEWS, "views.json")))
    n = len(me.vertices)
    co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    nr = np.empty(n * 3, "f"); me.vertex_normals.foreach_get("vector", nr); nr = nr.reshape(-1, 3)
    bm_tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    score = np.zeros((n, len(CLASSES)))
    for i, c in enumerate(cams):
        lab = np.load(os.path.join(VIEWS, "view_%02d_seg.npy" % i))
        m = cam_matrix(c["az"], c["elev"])
        to_cam = np.array(m.col[2].xyz)  # camera looks down -Z, so +Z points back at the camera
        rel = co - np.array(m.translation)
        right, up = np.array(m.col[0].xyz), np.array(m.col[1].xyz)
        u = ((rel @ right) / ORTHO + 0.5) * SIZE
        v = (0.5 - (rel @ up) / ORTHO) * SIZE  # image row 0 is the top
        facing = nr @ to_cam
        cand = np.nonzero((facing > 0.1) & (u >= 0) & (u < SIZE) & (v >= 0) & (v < SIZE))[0]
        d = Vector(to_cam)
        for j in cand:
            p = Vector(co[j]) + Vector(nr[j]) * 0.002 + d * 0.002
            if bm_tree.ray_cast(p, d)[0] is None:
                l = lab[int(v[j]), int(u[j])]
                if l > 0:
                    score[j, l] += facing[j]
        print("view", i, "candidates", len(cand))
    label = score.argmax(1)
    label[score.max(1) == 0] = 0

    # smooth on the mesh: a few rounds of neighbour majority vote (fills unseen vertices too)
    ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
    for _ in range(10):
        votes = np.zeros((n, len(CLASSES)))
        np.add.at(votes, (ed[:, 0], label[ed[:, 1]]), 1)
        np.add.at(votes, (ed[:, 1], label[ed[:, 0]]), 1)
        votes[:, 0] = 0  # "none" never spreads
        votes[np.arange(n), label] += 1.5  # a vertex keeps its own label unless clearly outvoted
        label = np.where(votes.max(1) > 0, votes.argmax(1), label)
    # small islands (a skirt-coloured patch on a thigh, a navel read as clothing) take their surroundings' label
    from collections import deque
    adj = [[] for _ in range(n)]
    for a_, b_ in ed:
        adj[a_].append(b_)
        adj[b_].append(a_)
    seen = np.zeros(n, bool)
    for s0 in range(n):
        if seen[s0]:
            continue
        comp, border, q = [], [], deque([s0])
        seen[s0] = True
        while q:
            i = q.popleft()
            comp.append(i)
            for j in adj[i]:
                if label[j] != label[s0]:
                    border.append(label[j])
                elif not seen[j]:
                    seen[j] = True
                    q.append(j)
        if len(comp) < 300 and border:
            label[comp] = np.bincount(border, minlength=len(CLASSES)).argmax()
    np.save(os.path.join(OUT, "part_labels.npy"), label)
    print("labels:", {CLASSES[k]: int((label == k).sum()) for k in range(len(CLASSES))})

    # preview: vertex colours by part
    attr = me.color_attributes.new("part", "FLOAT_COLOR", "POINT")
    col = np.ones((n, 4), "f")
    col[:, :3] = COLORS[label]
    attr.data.foreach_set("color", col.ravel())
    me.color_attributes.active_color = attr
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX"
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = 500, 760
    cam = bpy.data.objects.new("pl_cam", bpy.data.cameras.new("pl_cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ORTHO
    scene.collection.objects.link(cam)
    scene.camera = cam
    for az in (0, 45, 90, 180, 270):
        cam.matrix_world = cam_matrix(az, 5)
        scene.render.filepath = os.path.join(OUT, "parts_%03d.png" % az)
        bpy.ops.render.render(write_still=True)
    print("DONE vote")
