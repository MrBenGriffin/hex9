import math
import numpy as np
import csv
import scipy as sp
import scipy.spatial

def ll_xyz(lat, lon) -> tuple:
    phi, theta = np.radians(lat), np.radians(lon)
    x = math.cos(phi) * math.cos(theta)
    y = math.cos(phi) * math.sin(theta)
    z = math.sin(phi)  # z is 'up'
    return x, y, z


if __name__ == '__main__':
    # "Lat", "Lon", "Population"
    data = dict()
    lats, lons, pops = set(), set(), set()
    fn = 'assets/population_gbr_2019-07-01.csv'
    reader = csv.reader(open(fn, 'r'), quoting=csv.QUOTE_ALL, lineterminator='\n')
    for row in reader:
        lat, lon, pop = float(row[0]), float(row[1]), float(row[2])
        lats.add(lat)
        lons.add(lon)
        pops.add(pop)
        xyz = ll_xyz(lat, lon)
        data[xyz] = pop
    llats = sorted(lats)
    llons = sorted(lons)
    lpops = sorted(pops)
    print(f'len lats: {len(lats)}, lons: {len(lons)}, pops: {len(pops)}')
    print(f'min lats: {min(llats)}, lons: {min(llons)}, pops: {min(lpops)}')
    print(f'max lats: {max(llats)}, lons: {max(llons)}, pops: {max(lpops)}')

    kvec = [list(k) for k in data.keys()]
    keys = np.array(kvec, dtype=float) # fails.
    tree = sp.spatial.KDTree(keys)

    # make 100 random unit vectors suitable for a sphere and normalise.
    n = 100
    v = np.random.normal(size=(n, 3))
    vn = [0, 0, 1] + v / np.linalg.norm(v, axis=1, keepdims=True)

    for pt in vn:
        dx, k = tree.query(pt, workers=-1, k=1)
        kx = tuple(tree.data[k])
        print(f'kx: {kx}')
        print(f'pop: {data[kx]}')

        # llp = np.array(data)

