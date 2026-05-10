# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
from math import sqrt
from hhg9.h9.ellipsoid import ellipsoid_area_wgs84

if __name__ == '__main__':
    earth = ellipsoid_area_wgs84() # 510_065_621_724_154.6
    hex_0 = earth / 12
    tri_0 = earth / 8
    l_hex = hex_0
    l_tri = tri_0
    h_areas = np.zeros((64,), dtype=np.float64)
    t_areas = np.zeros((64,), dtype=np.float64)
    for i in range(20):
        h_areas[i] = l_hex
        h_a = sqrt(2)*3**(1/4)*sqrt(l_hex)/3   #
        l_hex /= 9
        print(f'layer: {i}, hex_area: {h_areas[i]}, side: {h_a}')
    accuracy = 38  # accuracy is nanometres.
