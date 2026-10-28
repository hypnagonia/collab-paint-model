"""Nose tip and chin from the mesh's centre-line profile (character faces -Y). Imported by merge_head.py,
or run in Blender to print: blender -b MODEL.blend -P face_profile.py"""
import numpy as np


def face_landmarks(co, search_top, search_bottom):
    """co: (n,3) vertices. Returns dict(nose=(x,y,z), chin=(x,y,z)) using the front profile near x of the nose."""
    # head centre line from the skull above the face (hair locks hang in front of the face sides)
    top = co[(co[:, 2] < search_top + 0.06) & (co[:, 2] > search_top - 0.02)]
    hx = (np.percentile(top[:, 0], 10) + np.percentile(top[:, 0], 90)) / 2
    band = (co[:, 2] < search_top) & (co[:, 2] > search_bottom) & (np.abs(co[:, 0] - hx) < 0.025)
    b = co[band]
    nose = b[np.argmin(b[:, 1])]
    xc = nose[0]
    col = co[(np.abs(co[:, 0] - xc) < 0.012) & (co[:, 2] < nose[2]) & (co[:, 2] > nose[2] - 0.2)]
    zs = np.arange(nose[2], nose[2] - 0.2, -0.004)
    front = np.array([col[np.abs(col[:, 2] - z) < 0.003][:, 1].min() if (np.abs(col[:, 2] - z) < 0.003).any()
                      else np.nan for z in zs])
    # chin: last level before the front surface falls back > 3 cm behind the nose (under the jaw)
    chin_i = None
    for i, f in enumerate(front):
        if np.isfinite(f) and f > nose[1] + 0.03 and i > 5:
            chin_i = i - 1
            break
    chin_z = zs[chin_i] if chin_i is not None else nose[2] - 0.07
    chin_y = np.nanmin(front[max(0, chin_i - 3):chin_i + 1]) if chin_i is not None else nose[1] + 0.01
    return {"nose": nose.tolist(), "chin": [float(xc), float(chin_y), float(chin_z)]}


if __name__ == "__main__":
    import bpy
    o = next(o for o in bpy.data.objects if o.type == "MESH")
    co = np.array([v.co for v in o.data.vertices])
    zmax = co[:, 2].max()
    print("LANDMARKS", face_landmarks(co, zmax - 0.12, zmax - 0.4))
