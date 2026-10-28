"""Import the generated mesh, normalise it (feet on ground, 1.92 m tall, centred), decimate, render clay previews, export."""
import bpy, math, os, sys

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = argv[0], argv[1]
NAME = os.path.basename(SRC).split("_raw")[0] if "_raw" in SRC else "ranger"
TARGET_FACES = int(argv[2]) if len(argv) > 2 else 200000
YAW = float(argv[3]) if len(argv) > 3 else 0.0
HEIGHT = float(argv[4]) if len(argv) > 4 else 1.92

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.active_object
obj.name = NAME
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
obj.rotation_euler[2] += math.radians(YAW)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(bpy.context.scene.objects):
    if o.type != "MESH":
        bpy.data.objects.remove(o)

# normalise: 1.92 m tall, feet at z=0, centred on x/y
xs, ys, zs = zip(*(v.co for v in obj.data.vertices))
s = HEIGHT / (max(zs) - min(zs))
cx, cy, z0 = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, min(zs)
for v in obj.data.vertices:
    v.co.x, v.co.y, v.co.z = (v.co.x - cx) * s, (v.co.y - cy) * s, (v.co.z - z0) * s

faces = len(obj.data.polygons)
if faces > TARGET_FACES:
    d = obj.modifiers.new("decimate", "DECIMATE")
    d.ratio = TARGET_FACES / faces
    bpy.ops.object.modifier_apply(modifier="decimate")
for p in obj.data.polygons:
    p.use_smooth = True
print("faces", faces, "->", len(obj.data.polygons))

m = bpy.data.materials.new("clay")
bsdf = m.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.42, 0.40, 0.36, 1)
bsdf.inputs["Roughness"].default_value = 0.7
obj.data.materials.clear()
obj.data.materials.append(m)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "MATCAP"
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = "BOTH"
scene.render.resolution_x, scene.render.resolution_y = 640, 1000
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = 2.15
scene.collection.objects.link(cam)
scene.camera = cam
for tag, ang in (("front", 0), ("three_q", 40), ("side", 90), ("back", 180)):
    a = math.radians(ang)
    cam.location = (5 * math.sin(a), -5 * math.cos(a), 0.97)
    cam.rotation_euler = (math.pi / 2, 0, a)
    scene.render.filepath = os.path.join(OUT, "prev_%s.png" % tag)
    bpy.ops.render.render(write_still=True)
bpy.data.objects.remove(cam)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NAME + "_hy3d.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + "_hy3d.glb"), export_format="GLB",
                          export_normals=True, export_materials="EXPORT")
print("DONE")
