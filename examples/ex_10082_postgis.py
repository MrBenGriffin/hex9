# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

lat, lon = 0,0

import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import hex_pack

reg = Registrar()  # Manage Domains & Projections
g_gcd = reg.domain('g_gcd')
b_oct = reg.domain('b_oct')
b_pts = reg.project(Points(np.array([[lat, lon]]), g_gcd), [g_gcd, b_oct])
adr = hex_pack(b_pts, 29)
fdx = [f'{a:0x}' for a in adr[0]]
str = ' '.join(fdx)

print(str)