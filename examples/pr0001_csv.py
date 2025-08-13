"""
Part of the H9 project - Preparation 0001
Load a CSV file and store it as numpy.
In this case, we are loading the Meta General Population csv.
https://data.humdata.org/organization/meta
This is 33k data points, with fractional population counts.
longitude,latitude,_general_2020
Last Tested 12 August 2025 √
"""
import numpy as np

if __name__ == '__main__':
    file = 'jpn'
    src = f'src/{file}_general_2020.csv'
    data = np.genfromtxt(src, delimiter=',', skip_header=1)
    # stored as lon;lat;population!
    records = data.shape[0]
    output = f'src/{file}_lon_lat_pop.npy'
    print(f'Converted {src} to {records} rows. Writing {output}')
    np.save(output, data)
