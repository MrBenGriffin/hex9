# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path

from hhg9 import Registrar
import csv

def run(net, src, out):
    with open('ref_data.csv.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(0, len(data), CHUNK):
            chunk = data[i:i + CHUNK]
            lats, lons, pops = chunk[:, 0], chunk[:, 1], chunk[:, 2]
            uuids, _ = h9_encode(lats, lons)
            bins = {L: h9_bin(uuids, L) for L in LAYERS}

            min_lat = min(min_lat, lats.min())
            max_lat = max(max_lat, lats.max())
            min_lon = min(min_lon, lons.min())
            max_lon = max(max_lon, lons.max())

            for j, (uuid, pop) in enumerate(zip(uuids, pops)):
                w.writerow([str(uuid), pop] + [str(bins[L][j]) for L in LAYERS])

    print(f"Bounds: {min_lon},{min_lat} → {max_lon},{max_lat}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Project lat to net and back')
    p.add_argument('--net', type=str, default='mortar')
    p.add_argument('--src', type=Path, default=Path('ref_data.csv'))
    p.add_argument('--out', type=Path, default=Path('net_result.csv'))
    args = p.parse_args()
    run(args.net, src=args.src, out=args.out)
