"""
Part of the H9 project
This is proof-of-concept for uint64 packing.
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
