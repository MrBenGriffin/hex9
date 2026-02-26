# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CloughTocher2DInterpolator

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9_RA, H9O
from hhg9.h9.region import regions_xy
from hhg9.h9.polygon import hex_poly_layer
import matplotlib as mpl


# --- LOAD L4 FINAL ---
LAYER = 4
DATA_FILE = "output/L4_Final_Ironed.npz"  # Your Polished Master
data = np.load(DATA_FILE)
grid = data['grid']

# --- REBUILD TRIANGULATION ---
# (Need t_grid indices)
grid_struct = np.load(f"grid_l{LAYER}.npz")
t_grid = grid_struct['grid'] # indices [N, 3]
cmp = grid_struct['cmp']

# --- CALCULATE ERROR ---
rg = Registrar()
b_oct, g_gcd = rg.domain('b_oct'), rg.domain('g_gcd')
dpts = Points(grid, b_oct, cmp)
gpts = rg.project(dpts, [b_oct, g_gcd])

# Get centroids of triangles for plotting
tri_coords = grid[t_grid] # [N_tri, 3, 2]
centroids = np.mean(tri_coords, axis=1) # [N_tri, 2]

# Calculate Area Ratios
t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
ideal = np.mean(areas)
error = np.abs(areas / ideal - 1.0) # Absolute % Error

# --- PLOT ---
plt.figure(figsize=(12, 12))
# Plot only triangles with > 2% error to see the "Bad Zones"
mask = error > 0.02

plt.scatter(centroids[~mask, 0], centroids[~mask, 1], s=1, c='lightgray', alpha=0.1, label="< 2% Error")
sc = plt.scatter(centroids[mask, 0], centroids[mask, 1], s=10, c=error[mask], cmap='inferno', vmin=0.02, vmax=0.10)
plt.colorbar(sc, label="Abs Area Error (2% to 10%)")
plt.title(f"L{LAYER} Error Hotspots (p99={np.quantile(error, 0.99):.3f})")
plt.axis('equal')
plt.savefig(f"output/L{LAYER}_Error_Map.png")
print("Saved Error Map to output/L4_Error_Map.png")