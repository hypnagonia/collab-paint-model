"""Face bounding box (ATR "Face" class) in a reference crop, as fractions of the crop size.

Usage: python face_box.py CROP.png [CROP2.png ...]  -> prints JSON {path: [x0, y0, x1, y1]} (y from the top)
"""
import json
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "mattmdjaga/segformer_b2_clothes"
proc = AutoImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
out = {}
for path in sys.argv[1:]:
    rgba = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    bg.alpha_composite(rgba)
    with torch.no_grad():
        logits = model(**proc(images=bg.convert("RGB"), return_tensors="pt")).logits
    lab = torch.nn.functional.interpolate(logits, size=rgba.size[::-1], mode="bilinear").argmax(1)[0].numpy()
    ys, xs = np.nonzero(lab == 11)
    w, h = rgba.size
    out[path] = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
print(json.dumps(out))
