# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
import matplotlib.pyplot as plt
from hhg9 import Registrar
from hhg9.projections import PlatePixelGCD

if __name__ == '__main__':

    reg = Registrar()

    p_pix = reg.domain('p_pix')  # PlatePixel
    c_sph = reg.domain('c_ell')  # EllipsoidCartesian(reg)  # Cartesian Spherical (xyz)
    g_sph = reg.domain('g_gcd')  # GeneralGCD(reg)  # Cartesian Spherical (xyz)
    # pg = PlatePixelGCD(reg)

    # Create dummy image of shape (18, 36, 3)
    h, w = 1800, 3600
    img = np.zeros((h, w, 1))
    p0 = p_pix.adopt(img)  # Shape: (648, 5)

    # Project to lat/lon and back
    l1 = reg.project(p0, [p_pix, g_sph])
    p1 = reg.project(l1, [g_sph, p_pix])

    # Compute pixel round-trip error
    original_px = np.array(p0.coords, dtype=np.uint64)
    projects_px = np.array(p1.coords, dtype=np.uint64)

    px_error = np.linalg.norm(original_px - projects_px, axis=1)
    p1.samples = px_error
    p2 = p_pix.image(p1)
    plt.imshow(p2, origin='lower')
    plt.show()

    plt.imshow(p2, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()

    error_img = px_error.reshape(h, w)
    plt.imshow(error_img, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()
