# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - uses H9 addresses and round-trips them.
This:
(a) Generates a Reference set of GCD random (seeded) addresses
(b) Projects them onto Octahedral Barycentric coordinates.
(c.1) Projects them back to GCD for roundtrip testing
(c.2.a) Renders them as H9 Hex labels
(c.2.b) Reverts the labels back to Octahedral Barycentric.
(c.2.c) Projects the reverted back to GCD for roundtrip testing.
(d) Measures, via GeographicLib Inverse, the ∂ distance from Reference
(e) Logs the results into a CSV file.
# ⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
02 Mar 2026 0.1.1a1 (passed; tidied somewhat.)
26 Dec 2025 0.1.0a4 (passed; accuracy has changed to area m^2)
16 Dec 2025 0.1.0a3 (passed; after rewriting formatter)
25 Nov 2025 (passed)
"""
import csv
import time
import numpy as np
from pathlib import Path
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd
from hhg9.h9 import H9K
from hhg9.h9.addressing import reg_hex_digits, TailStyle
from hhg9.h9.classifier import location
from hhg9.h9.region import xy_regions


class CSVLogger:
    """Append rows to a CSV with a fixed header; creates parent dirs as needed."""
    HEADER = [
        "lat_in", "lon_in", "oct_in", "bry_x", "bry_y", "gcd_err_val", "grid_addr", "grid_err_val", "loc"
    ]

    def __init__(self, path, header=None):
        self.path = Path(path) if path else None
        self.header = self.HEADER
        self._fh = None
        self._w = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._w = csv.writer(self._fh)
            self._w.writerow(self.header)

    def write(self, row_dict):
        """Write a row"""
        if not self._w:
            return
        row = [row_dict.get(k, "") for k in self.header]
        self._w.writerow(row)

    def close(self):
        """Close logging"""
        if self._fh:
            self._fh.close()
            self._fh = None
            self._w = None


def pt_loc(coords, mo):
    """Identify where points lie in their octants."""
    x, y = coords[:, 0], coords[:, 1]
    ẋ = H9K.R3 * x
    return location(ẋ, y, mo)


def run(reg, logger, refs, layers=36):
    h9f = OctahedralH9(reg)
    h9f.width = layers
    lat = refs.coords[:, 0]
    lon = refs.coords[:, 1]
    # b_rys = reg.project(refs, ['g_gcd', 'c_ell', 'c_oct', 'b_oct'])  # components filled
    print('100k is likely to take about 30s')

    print('projecting g_gcd->b_oct')
    b_rys = reg.project(refs, ['g_gcd', 'b_oct'])  # components filled
    b_rys.domain.register_format(h9f)
    print('projecting b_oct->g_gcd')
    b_rtp = reg.project(b_rys, ['b_oct', 'g_gcd'])
    # Get the octant id and mode of each point (discarding mode here).
    oc, mo = b_rys.cm()
    locs = pt_loc(b_rys.coords, mo)
    # Now generate the set of h9f labels for each point, and convert to np.array
    print('generating b_oct (h9) labels')
    labels = f'{b_rys:h9.{layers}}'  # This is reversible addresses.
    label_vec = np.array(labels.splitlines())
    # the h9f formatter h9f is used to revert this array back to barycentric.
    print('converting (h9) labels to b_oct ')
    h9_r = h9f.revert(labels)  # Convert from addresses.
    # we now take these and project them back to GCD.
    print('projecting converted labels in b_oct->g_gcd ')
    l_rtp = reg.project(h9_r, ['b_oct', 'g_gcd'])
    # We will use the GeographicLib Inverse to find ∂-distances (metres)
    # and multiply the values by 1e+9 such that our ∂-distances are in nanometres.
    print('calculating ∂-distances wgs84 distances between reference and reverted b_oct in nm (nanometres)')
    b_deltas = wgs84(refs.coords, b_rtp.coords) * 1e+9
    print('calculating ∂-distances wgs84 distances between reference and reverted label b_oct in nm (nanometres)')
    l_deltas = wgs84(refs.coords, l_rtp.coords) * 1e+9
    print('storing results in csv')
    for i in range(refs.coords.shape[0]):
        # Now write to the logger.
        logger.write({
            "lat_in": f"{refs.coords[i][0]:.16f}", "lon_in": f"{refs.coords[i][1]:.16f}",
            "bry_x": f"{b_rys.coords[i][0]:.16f}", "bry_y": f"{b_rys.coords[i][1]:.16f}",
            "oct_in": int(oc[i]),
            "lat_rt_gb": f"{b_rtp.coords[i][0]:.16f}", "lon_rt_gb": f"{b_rtp.coords[i][1]:.16f}",
            "gcd_err": f'{b_deltas[i]:.6f}nm',
            "gcd_err_val": f'{b_deltas[i]:.6f}',
            "grid_addr": label_vec[i],
            "lat_rt_ga": f"{l_rtp.coords[i][0]:.16f}", "lon_rt_ga": f"{l_rtp.coords[i][1]:.16f}",
            "grid_err": f'{l_deltas[i]:.6f}nm',
            "grid_err_val": f'{l_deltas[i]:.6f}',
            "loc": f'{locs[i]}',
        })
    b_min = float(np.min(b_deltas))
    b_max = float(np.max(b_deltas))
    b_p90 = np.percentile(b_deltas, 90)
    b_p99 = np.percentile(b_deltas, 99)
    l_min = float(np.min(l_deltas))
    l_max = float(np.max(l_deltas))
    l_p90 = np.percentile(l_deltas, 90)
    l_p99 = np.percentile(l_deltas, 99)
    print(f'b_min: {b_min:.6f}, b_max: {b_max:.6f}, b_p90: {b_p90:.6f}, b_p99: {b_p99:.6f}')
    print(f'l_min: {l_min:.6f}, l_max: {l_max:.6f}, l_p90: {l_p90:.6f}, l_p99: {l_p99:.6f}')


if __name__ == '__main__':
    accuracy = 1e-22  # in meters^2
    reg = Registrar()  # Manage Domains & Projections
    seed = 4007
    samples = 100_000
    ake = reg.projection('oct_ell')
    layers = ake.set_accuracy(accuracy)
    np.random.seed(seed)
    base = Path(__file__).parent

    log_file = base.joinpath(f'logs/L{layers}_{samples}_{seed}.csv')
    main_logger = CSVLogger(log_file)
    start_time = time.perf_counter()

    locr = gcd_rnd(samples)
    gpts = Points(locr, 'g_gcd')
    run(reg, main_logger, gpts, layers)
    main_logger.close()
    elapsed = time.perf_counter() - start_time
    print(f'Results written to {log_file.relative_to(base)}\n'
          f'Completed {len(gpts)} points in {elapsed:.3f} seconds'
          f' ({elapsed / len(gpts):.6f} sec/pt)')
