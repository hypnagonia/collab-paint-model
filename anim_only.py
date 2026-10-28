"""Export only the armature and its walk action (no mesh) so a page can swap walk styles on an already loaded model.

Usage: blender -b NAME_rigged.blend -P anim_only.py -- OUT.glb
"""
import bpy, sys

out = sys.argv[sys.argv.index("--") + 1]
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_animations=True,
                          export_force_sampling=True, export_frame_range=True, export_skins=False)
print("DONE")
