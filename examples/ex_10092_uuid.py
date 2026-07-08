# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
from hhg9.h9.uuid_address import h9_enc_ext, h9_label
from hhg9 import Registrar
from hhg9.h9.grid import HexMesh

if __name__ == '__main__':
    LAYER = 4
    reg = Registrar()
    mesh = HexMesh.create(LAYER, reg)
    ulabels = [h9_label(u, False) for u in mesh.addrs]
    nx = np.array(ulabels)  # human-readable, not-roundtrippable
    unq, cnt = np.unique(nx, return_counts=True)
    dset = unq[cnt > 1]
    print(f'{nx.shape[0]} addresses (without tail); {dset.shape[0]} duplicates')
