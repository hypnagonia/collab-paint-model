"""Pre-rendered backdrop for a fixed-camera scene (old survival-horror style): a dark-fantasy courtyard rendered once
from a fixed camera. The page draws this image behind a real-time 3D character seen through the same camera.

Writes OUT/backdrop.png and OUT/backdrop.json (camera position, look-at target, vertical fov, walkable area).
Usage: blender -b --factory-startup -P build_backdrop.py -- OUT_DIR
"""
import bpy, json, math, os, random, sys
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]
random.seed(4)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
eng = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
sc.render.engine = "CYCLES"; sc.cycles.samples = 96; sc.cycles.use_denoising = True; sc.cycles.device = "CPU"
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
try:
    sc.eevee.use_shadows = True
except Exception:
    pass


def mat(name, build):
    m = bpy.data.materials.new(name); nt = m.node_tree
    b = nt.nodes["Principled BSDF"]; build(nt, b); return m


def noise_ramp(nt, b, scale, stops, rough=0.9, voronoi=False):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    t = nt.nodes.new("ShaderNodeTexVoronoi" if voronoi else "ShaderNodeTexNoise")
    t.inputs["Scale"].default_value = scale
    if voronoi:
        t.feature = "DISTANCE_TO_EDGE"
    else:
        t.inputs["Detail"].default_value = 8
    nt.links.new(tc.outputs["Object"], t.inputs["Vector"])
    r = nt.nodes.new("ShaderNodeValToRGB")
    for i, (p, c) in enumerate(stops):
        e = r.color_ramp.elements[i] if i < 2 else r.color_ramp.elements.new(p)
        e.position = p; e.color = tuple((v / 255) ** 2.2 for v in c) + (1,)
    nt.links.new(t.outputs["Distance" if voronoi else "Fac"], r.inputs["Fac"])
    nt.links.new(r.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough


ground_m = mat("ground", lambda nt, b: noise_ramp(nt, b, 0.9, [(0.3, (20, 26, 18)), (0.55, (34, 42, 28)), (0.8, (48, 54, 38))]))
flag_m = mat("flag", lambda nt, b: noise_ramp(nt, b, 1.4, [(0.02, (12, 12, 14)), (0.08, (52, 50, 56)), (0.5, (70, 68, 74))], voronoi=True))
stone_m = mat("stone", lambda nt, b: noise_ramp(nt, b, 2.5, [(0.3, (40, 39, 44)), (0.6, (62, 60, 66)), (0.85, (82, 80, 86))]))
wood_m = mat("wood", lambda nt, b: noise_ramp(nt, b, 12, [(0.3, (30, 20, 14)), (0.7, (52, 36, 24))]))

def add(obj, m):
    # bake the scale in so procedural textures keep real-world size on stretched boxes
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(m); return obj

bpy.ops.mesh.primitive_plane_add(size=120); add(bpy.context.object, ground_m)
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0.003)); fp = bpy.context.object
fp.scale = (5.5, 9, 1); add(fp, flag_m)   # flagstone courtyard where she walks
# ruined colonnade at the back and sides
for i, x in enumerate([-6, -3, 0, 3, 6]):
    h = random.choice([3.8, 2.6, 4.2, 1.4, 3.5])
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.38, depth=h, location=(x, 6.5, h / 2)); add(bpy.context.object, stone_m)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 6.5, 0.15)); c = bpy.context.object; c.scale = (1.1, 1.1, 0.3); add(c, stone_m)
for x in (-7.2, 7.2):
    for y in (-1, 2.5):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 1.1)); w = bpy.context.object
        w.scale = (0.6, 3.2, random.uniform(1.6, 2.6)); w.rotation_euler.z = random.uniform(-0.05, 0.05); add(w, stone_m)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 8.5, 2.2)); w = bpy.context.object; w.scale = (16, 0.8, 4.4); add(w, stone_m)
# rubble and boulders
for _ in range(40):
    r = random.uniform(0.12, 0.6)
    x, y = random.uniform(-9, 9), random.uniform(-4, 8)
    if abs(x) < 5 and -4 < y < 4.5:
        continue
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=r, location=(x, y, r * 0.5))
    b = bpy.context.object; b.scale = (1, random.uniform(0.7, 1.3), random.uniform(0.5, 0.9)); b.rotation_euler = [random.uniform(0, 3) for _ in range(3)]
    add(b, stone_m)
# torches on posts
torches = [(-4.6, 3.5), (4.6, 3.5), (-4.6, -2.5), (4.6, -2.5)]
for x, y in torches:
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.07, depth=1.8, location=(x, y, 0.9)); add(bpy.context.object, wood_m)
    bpy.ops.object.light_add(type="POINT", location=(x, y, 1.95)); L = bpy.context.object.data
    L.energy = 380; L.color = (1.0, 0.55, 0.22); L.shadow_soft_size = 0.15
    fm = bpy.data.materials.new("fire"); nt = fm.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (1, 0.45, 0.12, 1); e.inputs["Strength"].default_value = 30
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o.inputs[0])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(x, y, 1.9)); add(bpy.context.object, fm)
# moon and sky
bpy.ops.object.light_add(type="SUN", rotation=(math.radians(55), 0, math.radians(-35))); m = bpy.context.object.data
m.energy = 1.4; m.color = (0.62, 0.68, 1.0); m.angle = math.radians(2)
w = bpy.data.worlds.new("sky"); sc.world = w; nt = w.node_tree
bg = nt.nodes["Background"]
grad = nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "EASING"
tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Rotation"].default_value = (0, math.radians(-90), 0)
nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs[0], grad.inputs["Vector"])
rp = nt.nodes.new("ShaderNodeValToRGB")
rp.color_ramp.elements[0].color = (0.035, 0.028, 0.055, 1); rp.color_ramp.elements[1].color = (0.004, 0.003, 0.008, 1)
nt.links.new(grad.outputs["Fac"], rp.inputs["Fac"]); nt.links.new(rp.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 1.0
# camera: high three-quarter view down the courtyard
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.sensor_fit = "VERTICAL"; cam.data.sensor_height = 24; cam.data.lens = 28
pos, target = Vector((0, -11.5, 5.2)), Vector((0, 1.5, 0.6))
cam.location = pos
cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
vfov = 2 * math.atan(cam.data.sensor_height / 2 / cam.data.lens)
sc.render.filepath = os.path.join(OUT, "backdrop.png")
bpy.ops.render.render(write_still=True)
json.dump({"pos": list(pos), "target": list(target), "vfov_deg": math.degrees(vfov), "aspect": 1600 / 900,
           "walk": {"xmin": -5.0, "xmax": 5.0, "ymin": -8.0, "ymax": 4.5}, "zup": True,
           "torches": [[x, y, 1.95] for x, y in torches]}, open(os.path.join(OUT, "backdrop.json"), "w"), indent=1)
print("DONE vfov %.1f" % math.degrees(vfov))
