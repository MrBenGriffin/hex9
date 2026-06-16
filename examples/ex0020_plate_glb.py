# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Supporting System Test - No octahedral projection or H9
This follows ex0010_plate_px.py (which loaded a png and displayed it).
This loads a Plate Carrée png, converts it to latitude/longitude, and displays it.
Needless to say, in this case, we could just use Basemap - but the point is to demonstrate
16 Jun 2026 0.1.3a0 (passed) 118.1s
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from PIL import Image  # Pillow for clean image saving
import numpy as np


def show_global(pts, proj='ortho', alpha=1.0):
    """Display GCD points on the globe with minimal whitespace"""
    x, y, z = pts.coords[:, 0], pts.coords[:, 1], pts.coords[:, 2]
    cols = pts.samples.astype(np.float64) / 255

    fig = plt.figure(figsize=(12, 12), dpi=300)
    # Use a transparent background for the figure
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c=cols, s=2, alpha=alpha, antialiased=True)

    # Remove all axes and panes
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Force the aspect ratio to be equal so the globe is a sphere
    # Adjust the camera distance to zoom in
    ax.set_box_aspect([1, 1, 1], zoom=1.4)

    # Tighten the limits to the sphere's bounds (assuming unit sphere)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 5. Save with tight_layout and no padding
    plt.tight_layout(pad=0)
    fig.savefig(f"output/ex0020_global.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)

    print(f'fig saved at output/ex0020_global.png')
    plt.close(fig)


def run():
    """
    Load a photo, adopt into PlatePixel points,
    transform to Sphere via GCD (latitude/longitude).
    Then display onto globe
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')  # PlatePixel (configured as Plate Carrée here)

    # Load as uint8 RGBA (matplotlib.imread often yields float32 0..1, which breaks roundtrips/saving)
    file = 'src/tissot_3600x1800.png'
    pil_img = Image.open(file).convert('RGBA')
    img = np.array(pil_img)
    print('img loaded', img.shape, img.dtype)

    # Adopt with an explicit full-sphere Plate Carrée extent so lon/lat mapping is correct.
    # extent ordering is (lon_min, lat_min, lon_max, lat_max) == (xmin, ymin, xmax, ymax)
    pc_extent = (-180.0, -90.0, 180.0, 90.0)
    pc_px = p_pix.adopt(img, extent=pc_extent, y_up=True, center=True)

    print(f'img adopted to Points')
    sp_ll = reg.project(pc_px, ['p_pix', 'g_gcd', 'c_ell'])  # project image as ECEF
    print(f'img projected to g_gcd')
    show_global(sp_ll)  # Display it as an overlay on the globe.
    # sp_pl = reg.project(sp_ll, ['g_gcd', 'p_pix'])  # Roundtrip back to Pixels
    # out = p_pix.image(sp_pl)
    #
    # # Ensure uint8 for saving.
    # if out.dtype != np.uint8:
    #     out = np.clip(out, 0, 255).astype(np.uint8)
    #
    # mode = 'RGBA' if (out.ndim == 3 and out.shape[2] == 4) else 'RGB'
    # pic = Image.fromarray(out, mode=mode)
    # pic.save('output/ex0020_flat.png')  # PNG keeps alpha if present


if __name__ == '__main__':
    run()
