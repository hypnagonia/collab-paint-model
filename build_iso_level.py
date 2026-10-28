"""Pre-rendered isometric level, larger than the screen, for a scrolling page: an orthographic camera at 30 deg
elevation / 45 deg azimuth renders a dark-fantasy ruin field once; the page scrolls this image under a real-time 3D
character drawn through an orthographic camera with the same direction and scale.

Writes OUT/iso_level.png and OUT/iso_level.json (camera axes, image centre, metres per pixel, walkable radius).
Usage: blender -b --factory-startup -P build_iso_level.py -- OUT_DIR
"""
import bpy, json, math, os, random, sys
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]
RES, ORTHO = 3200, 44.0
random.seed(7)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 64; sc.cycles.use_denoising = True; sc.cycles.device = "CPU"
sc.render.resolution_x = sc.render.resolution_y = RES


def mat(name, scale, stops, voronoi=False, rough=0.9):
    m = bpy.data.materials.new(name); nt = m.node_tree; b = nt.nodes["Principled BSDF"]
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
    return m


ground = mat("ground", 0.6, [(0.3, (18, 24, 16)), (0.55, (30, 38, 25)), (0.8, (44, 50, 34))])
flag = mat("flag", 1.5, [(0.02, (10, 10, 12)), (0.08, (50, 48, 54)), (0.5, (66, 64, 70))], voronoi=True)
stone = mat("stone", 2.5, [(0.3, (38, 37, 42)), (0.6, (58, 56, 62)), (0.85, (78, 76, 82))])
wood = mat("wood", 10, [(0.3, (26, 18, 13)), (0.7, (46, 32, 22))])
water = bpy.data.materials.new("water"); wb = water.node_tree.nodes["Principled BSDF"]
wb.inputs["Base Color"].default_value = (0.01, 0.012, 0.02, 1); wb.inputs["Roughness"].default_value = 0.05


def put(obj, m):
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(m); return obj


bpy.ops.mesh.primitive_plane_add(size=200); put(bpy.context.object, ground)
# a winding flagstone road and a plaza
pts = [(-16, -14), (-9, -8), (-3, -6), (2, -1), (2, 4), (8, 9), (15, 15)]
for k, ((x0, y0), (x1, y1)) in enumerate(zip(pts[:-1], pts[1:])):
    L = math.hypot(x1 - x0, y1 - y0)
    bpy.ops.mesh.primitive_plane_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, 0.004 + 0.003 * k))  # no z-fighting
    p = bpy.context.object; p.scale = (L + 1.6, 2.6, 1); p.rotation_euler.z = math.atan2(y1 - y0, x1 - x0); put(p, flag)
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=4.2, depth=0.02, location=(2, 1.5, 0.02)); put(bpy.context.object, flag)
# pond
bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=3.2, depth=0.02, location=(-9, 5, 0.01)); pnd = bpy.context.object; pnd.scale = (1.4, 1, 1); put(pnd, water)
# ruined walls, broken colonnades, tombstones, boulders, dead trees
for _ in range(16):
    x, y = random.uniform(-19, 19), random.uniform(-19, 19)
    if math.hypot(x - 2, y - 1.5) < 6:
        continue
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.9)); w = bpy.context.object
    w.scale = (random.uniform(2, 5), 0.5, random.uniform(1, 2.4)); w.rotation_euler.z = random.choice([0, math.pi / 2]) + random.uniform(-0.1, 0.1)
    w.location.z = w.scale.z / 2; put(w, stone)
for a in range(10):
    ang = a / 10 * 2 * math.pi
    x, y = 2 + 6 * math.cos(ang), 1.5 + 6 * math.sin(ang)
    h = random.choice([3.4, 2.2, 1.0, 3.0, 0.6])
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.35, depth=h, location=(x, y, h / 2)); put(bpy.context.object, stone)
for _ in range(26):
    x, y = random.uniform(8, 18), random.uniform(-18, -4)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.45)); t = bpy.context.object
    t.scale = (0.55, 0.16, 0.9); t.rotation_euler = (random.uniform(-0.15, 0.15), random.uniform(-0.1, 0.1), random.uniform(-0.4, 0.4)); put(t, stone)
for _ in range(70):
    r = random.uniform(0.15, 0.8); x, y = random.uniform(-20, 20), random.uniform(-20, 20)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=r, location=(x, y, r * 0.4))
    b = bpy.context.object; b.scale = (1, random.uniform(0.7, 1.3), random.uniform(0.5, 0.9)); b.rotation_euler = [random.uniform(0, 3) for _ in range(3)]; put(b, stone)
for _ in range(14):
    x, y = random.uniform(-19, 19), random.uniform(-19, 19)
    if math.hypot(x - 2, y - 1.5) < 6:
        continue
    h = random.uniform(2.5, 4.5)
    bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=0.22, radius2=0.04, depth=h, location=(x, y, h / 2)); put(bpy.context.object, wood)
    for _ in range(4):
        bl = random.uniform(0.8, 1.6); z = random.uniform(h * 0.4, h * 0.85); ang = random.uniform(0, 2 * math.pi)
        bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.07, radius2=0.01, depth=bl,
                                        location=(x + math.cos(ang) * bl * 0.4, y + math.sin(ang) * bl * 0.4, z + bl * 0.25))
        br = bpy.context.object; br.rotation_euler = (math.radians(55) * math.sin(ang), -math.radians(55) * math.cos(ang), 0); put(br, wood)
# torches along the road and the plaza
torch = []
for (x, y) in [(-12, -12), (-6, -5), (0, -5), (5, -0.5), (-1, 4), (5, 6), (10, 12), (-2.5, -1.5), (6.5, 3.5), (-9, 9)]:
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.07, depth=1.8, location=(x, y, 0.9)); put(bpy.context.object, wood)
    bpy.ops.object.light_add(type="POINT", location=(x, y, 1.95)); L = bpy.context.object.data
    L.energy = 500; L.color = (1.0, 0.55, 0.22); L.shadow_soft_size = 0.15
    fm = bpy.data.materials.new("fire"); nt = fm.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (1, 0.45, 0.12, 1); e.inputs["Strength"].default_value = 30
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o.inputs[0])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(x, y, 1.9)); bpy.context.object.data.materials.append(fm)
    torch.append([x, y, 1.95])
bpy.ops.object.light_add(type="SUN", rotation=(math.radians(55), 0, math.radians(-35))); m = bpy.context.object.data
m.energy = 3.2; m.color = (0.62, 0.68, 1.0); m.angle = math.radians(2)
w = bpy.data.worlds.new("sky"); sc.world = w; w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.026, 0.05, 1)

C0 = Vector((0, 0, 0))
el, az = math.radians(30), math.radians(45)
d = Vector((-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))  # from the target to the camera
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = ORTHO; cam.data.clip_end = 200
cam.location = C0 + d * 60
cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
mw = cam.matrix_world
right, up = Vector(mw.col[0][:3]).normalized(), Vector(mw.col[1][:3]).normalized()
sc.render.filepath = os.path.join(OUT, "iso_level.png")
bpy.ops.render.render(write_still=True)
json.dump({"res": RES, "ortho": ORTHO, "m_per_px": ORTHO / RES, "center": list(C0), "to_camera": list(d),
           "right": list(right), "up": list(up), "walk_radius": 18.5, "torches": torch, "zup": True},
          open(os.path.join(OUT, "iso_level.json"), "w"), indent=1)
print("DONE")
