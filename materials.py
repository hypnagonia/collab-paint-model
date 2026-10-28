"""Split the painted mesh into material slots by part and give each its own surface: sheer skirt, glossy horns
and hair, matte skin, leather gloves and boots, cloth top. All slots share the baked albedo.

Usage: blender -b NAME_painted.blend -P materials.py -- PART_LABELS.npy OUT_DIR
"""
import bpy, os, sys
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
PARTS, OUT = argv[0], argv[1]
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
NAME = obj.name
part = np.load(PARTS)  # 0 none, 1 hair, 2 skin, 3 clothing, 4 shoes  (or *_part_ids.npy from paint_parts.py)
IDS = PARTS.endswith("_part_ids.npy")
co = np.array([v.co for v in me.vertices])
z, ax = co[:, 2], np.abs(co[:, 0])
zmax = z.max()

# refine the four classes into materials by where they sit on the body
MATS = {  # name: (roughness, alpha, metallic, sheen-ish specular)
    "skin": (0.55, 1.0, 0.0), "hair": (0.38, 1.0, 0.0), "horn": (0.22, 1.0, 0.0),
    "top": (0.75, 1.0, 0.0), "skirt": (0.6, 0.5, 0.0), "gloves": (0.5, 1.0, 0.0), "boots": (0.45, 1.0, 0.0),
}
names = list(MATS)
cls = np.full(len(co), names.index("skin"))
cls[part == 1] = names.index("hair")
cls[part == 4] = names.index("boots")
cloth = (part == 3) | (part == 0)
cls[cloth & (z < 0.98 * zmax * 0.56)] = names.index("skirt")
cls[cloth & (z >= 0.98 * zmax * 0.56)] = names.index("top")
cls[cloth & (ax > 0.17) & (z > 0.45 * zmax) & (z < 0.7 * zmax)] = names.index("gloves")
# horns: the bits of "hair" that stick up above the scalp
hair = part == 1
crown = np.percentile(z[hair], 97) if hair.any() else zmax
cls[hair & (z > crown - 0.035)] = names.index("horn")
if IDS:
    # paint_parts ids: skin hair horn top skirt gloves boots lips skirt_on_body; only the free sheet is see-through
    MATS["skirt_on_body"] = (0.6, 1.0, 0.0)
    names = list(MATS)
    remap = [names.index(k) for k in ("skin", "hair", "horn", "top", "skirt", "gloves", "boots", "skin", "skirt_on_body")]
    cls = np.array(remap)[part]
print({n: int((cls == i).sum()) for i, n in enumerate(names)})

albedo = next(i for i in bpy.data.images if i.name.endswith("_albedo") and i.users)
me.materials.clear()
for n in names:
    rough, alpha, metal = MATS[n]
    m = bpy.data.materials.new(n)
    b = m.node_tree.nodes["Principled BSDF"]
    t = m.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = albedo
    m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Alpha"].default_value = alpha
    if alpha < 1:
        m.blend_method = "BLEND" if hasattr(m, "blend_method") else None
        m.use_backface_culling = False
    me.materials.append(m)

fv = np.empty(len(me.polygons) * 3, np.int64)
# faces may be quads/tris: take each face's first three vertices for a majority vote
idx = [p.vertices[:3] for p in me.polygons]
fl = np.array([np.bincount(cls[list(v)], minlength=len(names)).argmax() for v in idx])
me.polygons.foreach_set("material_index", fl)
me.update()

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_final.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + "_final.glb"), export_format="GLB",
                          export_image_format="JPEG", export_vertex_color="NONE")
print("DONE")
