# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

from PIL import Image

src_path = "src/r10_xArtboard 16.png"        # the image you want to mask (RGB/RGBA)
mask_path = "src/16_Mask.png"        # grayscale mask (white keep, black discard)
out_path = "output/masked16.png"

img = Image.open(src_path).convert("RGBA")
mask = Image.open(mask_path).convert("L")     # 0..255

# Use the mask as alpha: white -> opaque, black -> transparent
r, g, b, a = img.split()
a2 = Image.composite(a, Image.new("L", img.size, 0), mask)  # keep a where mask is white, else 0
out = Image.merge("RGBA", (r, g, b, a2))

out.save(out_path)