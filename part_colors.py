"""Re-export the animated model with a per-vertex part colour (COLOR_0), for screen-space ink lines in a
cel-shaded scene: the page renders the parts as flat colours, rounds them, and draws a line wherever the part
changes from one pixel to the next. Because the colours are interpolated across triangles and rounded per pixel, the
borders come out as smooth curves, the same as the silhouette outline.

Part k is encoded in three bits: colour = (k & 1, k >> 1 & 1, k >> 2 & 1).
Parts: 0 skin, 1 hair, 2 top, 3 skirt, 4 bracers, 5 boots, 6 other, 7 horns.

Usage: blender -b NAME_rigged.blend -P part_colors.py -- OUT.glb
"""
import bpy, sys
import numpy as np

OUT = sys.argv[sys.argv.index("--") + 1]
obj = next(o for o in bpy.data.objects if o.type == "MESH")
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
me = obj.data
n = len(me.vertices)
M = np.array(obj.matrix_world)
co = np.empty(n * 3, "f"); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
x, y, z = co.T
zmax = z.max()
img = next(nd.image for m in me.materials for nd in m.node_tree.nodes if nd.type == "TEX_IMAGE" and nd.image)
W, H = img.size
px = np.empty(W * H * 4, "f"); img.pixels.foreach_get(px); px = px.reshape(H, W, 4)
uv = np.empty(len(me.loops) * 2, "f"); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
# a vertex sits on several UV islands at seams: average its loops' colours
c = np.zeros((n, 3)); cnt = np.zeros(n)
lc = px[np.clip((uv[:, 1] * H).astype(int), 0, H - 1), np.clip((uv[:, 0] * W).astype(int), 0, W - 1), :3]
np.add.at(c, lv, lc); np.add.at(cnt, lv, 1); c /= np.maximum(cnt, 1)[:, None]


def seg(p, a, b):
    a, b = np.array(a), np.array(b); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


fore = np.minimum.reduce([seg(co, (s * 0.23, 0.0, 1.14), (s * 0.31, -0.04, 0.82)) for s in (1, -1)])
skin = (c[:, 0] > c[:, 2] + 0.06) & (c[:, 0] > 0.2)
dark = c.max(1) < 0.3
violet = (c[:, 0] - c[:, 1] > 0.038) & (c[:, 2] > c[:, 1] + 0.03)
part = np.full(n, 6)
part[skin] = 0
part[~skin & dark & violet & (z > 1.05) & (fore > 0.07)] = 1
part[~skin & ~violet & (z > 1.18) & (z < 1.42) & (np.abs(x) < 0.2) & (y < -0.06)] = 2
part[~skin & (z < 1.06) & (z > 0.32)] = 3
part[(part == 6) & (z < 1.06) & (z > 0.32)] = 3
part[~skin & (fore < 0.075)] = 4
part[~skin & (z < 0.34)] = 5
part[(z > zmax - 0.16) & (c[:, 2] > c[:, 0] + 0.03) & ~skin] = 7
part[(part == 0) & (z > 0.05) & (z < 1.0)] = 3   # skin seen through the sheer skirt belongs to the skirt region
part[part == 6] = 1  # leftovers are mostly hair shadows

# clean: majority filter, then islands smaller than 80 vertices join their surroundings
ed = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
src = np.concatenate([ed[:, 0], ed[:, 1]]); dst = np.concatenate([ed[:, 1], ed[:, 0]])
for _ in range(3):
    votes = np.zeros((n, 8)); np.add.at(votes, (dst, part[src]), 1); votes[np.arange(n), part] += 1.5
    part = votes.argmax(1)
adj = [[] for _ in range(n)]
for a, b in ed:
    adj[a].append(b); adj[b].append(a)
seen = np.zeros(n, bool)
for i0 in range(n):
    if seen[i0]:
        continue
    comp, stack, border = [], [i0], []
    seen[i0] = True
    while stack:
        i = stack.pop(); comp.append(i)
        for j in adj[i]:
            if part[j] != part[i0]:
                border.append(part[j])
            elif not seen[j]:
                seen[j] = True; stack.append(j)
    if len(comp) < 80 and border:
        part[comp] = np.bincount(border, minlength=8).argmax()
print("parts", np.bincount(part, minlength=8))
col = np.stack([(part & 1), (part >> 1) & 1, (part >> 2) & 1], 1).astype("f")
for a in list(me.color_attributes):
    me.color_attributes.remove(a)
ca = me.color_attributes.new("part", "FLOAT_COLOR", "POINT")
c4 = np.ones((n, 4), "f"); c4[:, :3] = col; ca.data.foreach_set("color", c4.ravel())
me.color_attributes.active_color = ca
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True); rig.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_image_format="JPEG",
                          export_animations=True, export_force_sampling=True, export_frame_range=True,
                          export_vertex_color="ACTIVE")
print("DONE")
