"""Render the walk cycle as 2D sprites: N directions x M frames, transparent PNGs, orthographic camera.

Usage: blender -b ranger_rigged.blend -P render_sprites.py -- OUT_DIR SET_NAME ELEVATION_DEG DIRECTIONS FRAMES SIZE
Azimuth is the camera's angle around the character; folders are named by where the character faces on
screen: S = toward the viewer, N = away, E = screen-right, W = screen-left.
"""
import bpy, math, os, sys, json

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, NAME = argv[0], argv[1]
ELEV = float(argv[2])
DIRS = [int(d) for d in argv[3].split(",")]  # azimuths in degrees, 0 = facing the viewer
FRAMES = int(argv[4])
SIZE = int(argv[5])
CYCLE = 32

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

world = bpy.data.worlds.new("sprite_world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.36, 0.38, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
scene.world = world


def light(name, kind, energy, rot, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, kind)
    l.energy = energy
    l.color = color
    o = bpy.data.objects.new(name, l)
    o.rotation_euler = [math.radians(r) for r in rot]
    scene.collection.objects.link(o)
    return o


# lights are fixed in world space relative to the camera rig below, so every direction gets the same look
rig_lights = [light("key", "SUN", 3.2, (50, 0, 35), (1.0, 0.95, 0.88)),
              light("fill", "SUN", 0.9, (60, 0, -60), (0.85, 0.9, 1.0)),
              light("rim", "SUN", 2.2, (-50, 0, 180), (0.9, 0.95, 1.0))]

cam_data = bpy.data.cameras.new("sprite_cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.35
cam = bpy.data.objects.new("sprite_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
pivot = bpy.data.objects.new("pivot", None)
scene.collection.objects.link(pivot)
pivot.location = (0, 0, 0.98)
cam.parent = pivot
for l in rig_lights:
    l.parent = pivot

e = math.radians(ELEV)
cam.location = (0, -6 * math.cos(e), 6 * math.sin(e))
cam.rotation_euler = (math.pi / 2 - e, 0, 0)
# a raised camera sees the character lower in frame; nudge so feet and hood stay inside
pivot.location.z = 0.98 - 0.1 * math.sin(e)

meta = {"name": NAME, "elevation_deg": ELEV, "frame_size": SIZE, "frames": FRAMES,
        "fps_suggested": round(FRAMES / (CYCLE / 30), 1), "directions": []}
FACING = {0: "S", 45: "SW", 90: "W", 135: "NW", 180: "N", 225: "NE", 270: "E", 315: "SE"}
step = CYCLE / FRAMES
for az in DIRS:
    pivot.rotation_euler = (0, 0, math.radians(az))
    label = FACING.get(az % 360, "az%03d" % az)
    d = os.path.join(OUT, NAME, "walk_" + label)
    os.makedirs(d, exist_ok=True)
    for i in range(FRAMES):
        scene.frame_set(1 + int(round(i * step)))
        scene.render.filepath = os.path.join(d, "walk_%s_%02d.png" % (label, i))
        bpy.ops.render.render(write_still=True)
    meta["directions"].append({"facing": label, "azimuth_deg": az, "folder": "walk_" + label})
    print("rendered", NAME, az)
with open(os.path.join(OUT, NAME, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print("DONE")
