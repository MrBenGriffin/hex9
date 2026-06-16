# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Test Reference Neighbours.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 5.8s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed - but questionable)
16 Dec 2025 0.1.0a3 (passed - but of questionable value)
12 Oct 2025 (passed)
"""
import numpy as np
from hhg9 import Registrar
from hhg9.h9.region import region_neighbours

if __name__ == '__main__':
    rg = [
        #                                          (idx  imo, npm, c2, sib, PM, Legal)
        # [(0x5F, 0x5F, 0x2b), (0x3e, 0x39, 0x2b)]
        [(0x49, 0x3a, 0x2b), (0x49, 0x2a, 0x3e)],  # 01  [0,  0,  0,   1],  0   √√
        # [(0x5F, 0x5F, 0x49), (0x34, 0x16, 0x49)]
        [(0x49, 0x21, 0x49), (0x49, 0x25, 0x34)],  # 03  [0,  0,  1,   1],  0   √√
        # [(0x5F, 0x5F, 0x35), (0x25, 0x25, 0x21)]
        [(0x49, 0x3a, 0x21), (0x49, 0x39, 0x16)],  # 05  [0,  0,  2,   1],  0   √√

        [(0x49, 0x26, 0x2b), (0x16, 0x39, 0x3e)],  # 06  [0,  1,  0,   0],  0   √√
        [(0x16, 0x35, 0x2b), (0x16, 0x25, 0x3e)],  # 07  [0,  1,  0,   1],  1   √√
        [(0x49, 0x3a, 0x49), (0x16, 0x25, 0x34)],  # 08  [0,  1,  1,   0],  0   √√
        [(0x16, 0x35, 0x49), (0x16, 0x39, 0x34)],  # 09  [0,  1,  1,   1],  1   √√
        [(0x49, 0x35, 0x21), (0x16, 0x2a, 0x16)],  # 10  [0,  1,  2,   0],  0   √√
        [(0x16, 0x35, 0x21), (0x16, 0x34, 0x16)],  # 11  [0,  1,  2,   1],  1   √√

        [(0x16, 0x34, 0x3e), (0x49, 0x21, 0x2b)],  # 12  [1,  0,  0,   0],  1   √√
        [(0x49, 0x25, 0x3e), (0x49, 0x35, 0x2b)],  # 13  [1,  0,  0,   1],  0   √√
        [(0x16, 0x25, 0x34), (0x49, 0x3a, 0x49)],  # 14  [1,  0,  1,   0],  1   √√
        [(0x49, 0x25, 0x34), (0x49, 0x21, 0x49)],  # 15  [1,  0,  1,   1],  0   √√
        [(0x16, 0x16, 0x16), (0x49, 0x21, 0x21)],  # 16  [1,  0,  2,   0],  1   √√
        [(0x49, 0x25, 0x16), (0x49, 0x26, 0x21)],  # 17  [1,  0,  2,   1],  0   √√

        # [(0x35, 0x5F, 0x3e), (0x3e, 0x39, 0x3e)],
        [(0x16, 0x25, 0x3e), (0x16, 0x35, 0x2b)],  # 19  [1,  1,  0,   1],  1   √√
        # [(0x35, 0x5F, 0x34), (0x34, 0x16, 0x25)],
        [(0x16, 0x2a, 0x34), (0x16, 0x26, 0x49)],  # 21  [1,  1,  1,   1],  1   √√
        # [(0x35, 0x5F, 0x16), (0x25, 0x25, 0x16)],
        [(0x16, 0x39, 0x16), (0x16, 0x3a, 0x21)],  # 23  [1,  1,  2,   1],  1   √√

    ]
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    np.set_printoptions(formatter={'int': lambda x: f'0x{x:02x}'})
    for _uri, _ref in rg:
        uri = np.array([_uri], dtype=np.uint8)
        ref = np.array([_ref], dtype=np.uint8)
        nbr, nmo = region_neighbours(uri)
        ub2, nm2 = region_neighbours(nbr)
        if np.any(nbr != ref) or np.any(ub2 != uri):
            if np.any(uri != ub2):
                print(f'RTripErr URI:{uri}-> NB:{nbr} vs REF:{ref}; NB:{nbr}-> RT:{ub2} vs URI:{uri}')
            else:
                print(f'Mismatch  NB:{uri}-> NB:{nbr} vs REF:{ref}; NB:{nbr}-> RT:{ub2} vs URI:{uri}')
        else:
            continue
            # print(f'Matched! URI:{uri}-> NB:{nbr}')
    print('All tested')
