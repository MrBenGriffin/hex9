# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - This is proof-of-concept for uint64 packing.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 0.1s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed - with rewrite)
"""
import numpy as np
import hhg9.algorithms.packing as pk


def run():
    """Proof of concept"""
    rx = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 3, 3, 1, 1, 2, 9]])
    xc = pk.u64_pack(rx)
    cv = pk.u64_layers(xc)
    print(f'xc: {xc[0][0]:0x}')
    print("nibbles:", cv[0, :8])


if __name__ == "__main__":
    run()
