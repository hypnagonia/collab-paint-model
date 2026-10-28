"""Pack rendered walk frames into sprite sheets (one row per direction) + per-direction strips + a preview GIF."""
import json, os, sys
from PIL import Image

ROOT = sys.argv[1]
ORDER = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
for name in sorted(os.listdir(ROOT)):
    base = os.path.join(ROOT, name)
    if not os.path.isfile(os.path.join(base, "meta.json")):
        continue
    meta = json.load(open(os.path.join(base, "meta.json")))
    n, size = meta["frames"], meta["frame_size"]
    dirs = [d for d in ORDER if os.path.isdir(os.path.join(base, "walk_" + d))]
    sheet = Image.new("RGBA", (n * size, len(dirs) * size))
    bbox = [size, size, 0, 0]
    gif = []
    for r, d in enumerate(dirs):
        strip = Image.new("RGBA", (n * size, size))
        for i in range(n):
            im = Image.open(os.path.join(base, "walk_" + d, "walk_%s_%02d.png" % (d, i)))
            b = im.getbbox()
            if b:
                bbox = [min(bbox[0], b[0]), min(bbox[1], b[1]), max(bbox[2], b[2]), max(bbox[3], b[3])]
            strip.paste(im, (i * size, 0))
            sheet.paste(im, (i * size, r * size))
        strip.save(os.path.join(base, "walk_%s_strip.png" % d))
    sheet.save(os.path.join(base, "walk_sheet.png"))
    # animated preview: all directions side by side at half size
    for i in range(n):
        fr = Image.new("RGBA", (len(dirs) * size // 2, size // 2), (128, 132, 128, 255))
        for c, d in enumerate(dirs):
            im = Image.open(os.path.join(base, "walk_" + d, "walk_%s_%02d.png" % (d, i))).resize((size // 2, size // 2))
            fr.alpha_composite(im, (c * size // 2, 0))
        gif.append(fr.convert("RGB"))
    gif[0].save(os.path.join(base, "preview.gif"), save_all=True, append_images=gif[1:],
                duration=int(1000 / meta["fps_suggested"]), loop=0)
    meta.update({"sheet": "walk_sheet.png", "sheet_rows": dirs, "sheet_columns": "frame 0..%d" % (n - 1),
                 "content_bbox_px": bbox})
    json.dump(meta, open(os.path.join(base, "meta.json"), "w"), indent=2)
    print(name, "sheet", sheet.size, "content bbox", bbox, "of", size)
