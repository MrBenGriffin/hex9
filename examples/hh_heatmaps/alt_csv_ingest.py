"""
Part of the H9 project - Preparation 0001
Load a CSV file and store it as numpy.
In this case, we are loading the Meta General Population csv.
https://data.humdata.org/organization/meta
This is data points, with fractional population counts.
longitude,latitude,_general_2020
Last Tested 17 March 2026 √
"""
import numpy as np
import csv
from hhg9.h9.uuid_address import h9_encode, h9_bin


if __name__ == '__main__':
    data = np.load('src/gbr_lat_lon_pop.npy')
    CHUNK = 100_000
    LAYERS = range(5, 10)

    min_lat, max_lat = np.inf, -np.inf
    min_lon, max_lon = np.inf, -np.inf

    with open('gbr_uuid_pop.csv', 'w', newline='') as f:
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

