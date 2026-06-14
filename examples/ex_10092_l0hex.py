# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import numpy as np
from hhg9 import Registrar, Points
import hhg9.h9.region as rg
from hhg9.h9 import H9O

if __name__ == '__main__':
    reg = Registrar()
    # b_oct = reg.domain('b_oct')
    stonehenge = np.array([[51.1787980000210725, -1.8261898473293335]])
    g_pt = Points(stonehenge, 'g_gcd')
    b_pt = reg.project(g_pt, ['g_gcd', 'b_oct'])
    # now have b_oct.
    oc, mo = b_pt.cm()
    cx = rg.xy_regions(b_pt.coords, mo, 0, ensure)
    c2 = rg.H9R.mcc2[mo, cx[:, 1]]
    r0 = H9O.l0hex_by_id[oc, c2]
    print(c2)