# hunyuan3d pipeline

Image → 3D character pipeline on a Mac (Apple MPS + Blender), built around Tencent Hunyuan3D-2.

- `gen_sheet.py` — split a character sheet into views and generate the shape (Hunyuan3D-2mv)
- `finish_mesh.py` — normalise, decimate, clay previews
- `segment_refs.py`, `part_labels.py` — hair / skin / clothing labels on photos and on the mesh
- `paint_mesh.py`, `retexture.py`, `retex_hybrid.py` — photo projection painting, part-aware repaint
- `merge_head.py`, `trim_body_hair.py`, `mirror_region.py`, `symmetrize_upper.py` — mesh surgery
- `rig_*.py`, `mocap_walk.py`, `anim_only.py` — armature, weights, mocap retarget (CMU / Bandai Namco)
- `render_sprites.py`, `make_sheets.py`, `lowpoly.py` — 2D sprite sheets, low-poly export
- `prep_paint_job.py` + `colab/hunyuan_paint_v5.ipynb` — Hunyuan3D-2 Paint on free Colab (T4, 12 GB RAM):
  the mesh is cleaned, reduced and UV-unwrapped on the Mac; the notebook loads the paint models one by one
  straight onto the GPU so it fits the free tier

Hunyuan3D-2 models are under the Tencent Hunyuan non-commercial license; Bandai Namco motion data is
non-commercial (CC BY-NC-ND 4.0).
