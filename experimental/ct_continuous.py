# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import pickle
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from hhg9.h9 import H9K


class AuthalicWarp:
    def __init__(self, source_pts, target_pts):
        """
        Builds a C1-continuous warp field from Source -> Target.

        Parameters:
        source_pts (Nx2): The perfect, regular grid (a_p)
        target_pts (Nx2): The Sinkhorn-optimized grid (x_prime)
        """
        print(f"Building Clough-Tocher Interpolator ({len(source_pts)} points)...")
        # We build two separate interpolators: one for X-shift, one for Y-shift.
        # This is often more robust than interpolating the absolute position directly.
        diff = target_pts - source_pts

        self.dx_interp = CloughTocher2DInterpolator(source_pts, diff[:, 0])
        self.dy_interp = CloughTocher2DInterpolator(source_pts, diff[:, 1])
        print("Warp Ready.")

    def __call__(self, xy):
        """
        Warps an array of points (Mx2).
        Returns mapped points (Mx2).
        """
        xy = np.asarray(xy)
        if xy.ndim == 1: xy = xy[None, :]  # Handle single point

        # 1. Predict Displacement
        dx = self.dx_interp(xy)
        dy = self.dy_interp(xy)

        # 2. Handle 'Out of Bounds' (NaNs)
        # CT returns NaN if a point is outside the convex hull of the source grid.
        # For a map projection, this usually means the point is off-map.
        # We fill NaNs with 0 (no warp) or handle gracefully.
        mask_nan = np.isnan(dx) | np.isnan(dy)
        if np.any(mask_nan):
            # print(f"Warning: {np.sum(mask_nan)} points outside warp domain.")
            dx[mask_nan] = 0.0
            dy[mask_nan] = 0.0

        # 3. Apply
        return xy + np.stack([dx, dy], axis=1)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


# --- USAGE EXAMPLE (Append to your Feedback Loop script) ---
if __name__ == '__main__':
    # ... (After your loop finishes and you have the final x_prime) ...
    f_name = 'output/data/fb_iter22_1000000.npz'
    repo = np.load(f_name, allow_pickle=True)

    a_p = repo['a_p']  # original value
    t_p = repo['t_p']  # target value.
    layer = repo['layer']
    grid = repo['grid']
    octant_id = repo['octant_id']
    cmp = repo['cmp']

    # 1. Create the Warp
    warp_field = AuthalicWarp(a_p, t_p)

    # 2. Save it for later (So you don't have to re-run Sinkhorn)
    warp_field.save(f"output/H9_L{layer}_Warp.pkl")

    # 3. Test it: Warp a regular grid of lines to see the curvature
    # Create a grid of lines in the Source Domain
    test_x = np.linspace(H9K.limits.TL, H9K.limits.TR, 90)
    test_y = np.linspace(H9K.limits.VF, H9K.limits.VC, 90)

    # Visualize how straight lines become curves
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Plot warped horizontal lines
    for y in test_y:
        line = np.stack([np.linspace(-1, 1, 100) * H9K.limits.TR, np.full(100, y)], axis=1)
        # Filter to keep inside triangle (simple box check for demo)
        warped_line = warp_field(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], 'k-', alpha=0.3)

    # Plot warped vertical lines
    for x in test_x:
        # Create vertical span
        line = np.stack([np.full(100, x), np.linspace(H9K.limits.VF, H9K.limits.VC, 100)], axis=1)
        warped_line = warp_field(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], 'k-', alpha=0.3)

    ax.set_title(f"Continuous Authalic Warp Field (L{layer})")
    plt.savefig(f"output/warp_check_L{layer}.jpg")
    plt.close()
    print("Warp check saved.")

