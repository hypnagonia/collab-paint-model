"""Semantic labels for reference crops: hair / skin / clothing / shoes, saved as palette PNGs next to them.

Uses SegFormer-B2 fine-tuned on human clothing (ATR labels). Output class ids:
0 none, 1 hair, 2 skin (face, arms, legs), 3 clothing, 4 shoes
Usage: python segment_refs.py ref_front_crop.png [ref_right_crop.png ...]
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "mattmdjaga/segformer_b2_clothes"
# ATR: 0 Background 1 Hat 2 Hair 3 Sunglasses 4 Upper-clothes 5 Skirt 6 Pants 7 Dress 8 Belt 9 Left-shoe
# 10 Right-shoe 11 Face 12 Left-leg 13 Right-leg 14 Left-arm 15 Right-arm 16 Bag 17 Scarf
TO_CLASS = np.array([0, 3, 1, 3, 3, 3, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 3, 3], np.uint8)
PALETTE = [40, 40, 40, 200, 60, 160, 240, 190, 150, 60, 140, 200, 120, 90, 40]

proc = AutoImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
for path in sys.argv[1:]:
    rgba = Image.open(path).convert("RGBA")
    # the model expects a photo-like background; put the cut-out on mid grey
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    bg.alpha_composite(rgba)
    with torch.no_grad():
        logits = model(**proc(images=bg.convert("RGB"), return_tensors="pt")).logits
    up = torch.nn.functional.interpolate(logits, size=rgba.size[::-1], mode="bilinear", align_corners=False)
    lab = TO_CLASS[up.argmax(1)[0].numpy()]
    lab[np.array(rgba)[..., 3] < 128] = 0
    out = Image.fromarray(lab, "P")
    out.putpalette(PALETTE + [0] * (768 - len(PALETTE)))
    dst = path[:-len("_crop.png")] + "_seg.png" if path.endswith("_crop.png") else path[:-4] + "_seg.png"
    out.save(dst)
    np.save(dst.replace(".png", ".npy"), lab)  # row 0 = top
    print(dst, {n: round(float((lab == i).mean()), 3) for i, n in enumerate(["none", "hair", "skin", "cloth", "shoes"])})
