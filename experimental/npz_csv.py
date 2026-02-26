# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path

import numpy as np
import csv


if __name__ == '__main__':
    f_name = 'grid_idx.npz'
    # f_name = 'output/c3__rg0014_tf0250_cn1165_ct1070_L3.npz'
    data = np.load(f_name)
    for key, value in data.items():
        if key in ['xy_vert', 'pts']:
            fn = Path(f's_{key}.csv')
            if not fn.exists():
                with fn.open('w', newline='') as f:
                    for np_row in value:
                        val = [float(npf) for npf in np_row]
                        csv.writer(f).writerow(val)
                    f.close()
            break
        else:
            print(f"{key}: skipped")

