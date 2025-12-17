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
16 December 2025 0.1.0a3 (passed; after rewriting formatter)
25 November 2025 (passed)
"""
import csv
import time
import numpy as np
from pathlib import Path
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd
from hhg9.h9.addressing import reg_hex_digits, TailStyle
from hhg9.h9.region import xy_regions


class CSVLogger:
    """Append rows to a CSV with a fixed header; creates parent dirs as needed."""
    HEADER = [
        "lat_in", "lon_in", "oct_in", "bry_x", "bry_y", "gcd_err_val", "grid_addr", "grid_err_val",
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


def run(reg, logger, refs, layers=36):
    h9f = OctahedralH9(reg)
    lat = refs.coords[:, 0]
    lon = refs.coords[:, 1]
    # Resolve Ambiguity flags (vectorised)
    ε = 1e-15
    ambiguous = (
        (np.isclose(np.abs(lat), 90.0, atol=ε)) |
        (np.isclose(lat, 0.0, atol=ε) & (np.isclose(np.abs(lon), 180.0, atol=ε) | np.isclose(lon, 0.0, atol=ε)))
    ).astype(np.uint8)
    # 'ambiguous' marks any points which are on vertices.
    # such points might have multiple addresses.
    # project the GCD reference points onto barycentric octahedral.
    b_rys = reg.project(refs, ['g_gcd', 'b_oct'])  # components filled
    # Get the domain managing the barycentric, and register the h9f grid format.
    b_rys.domain.register_format(h9f)
    # project barycentric octahedral back to GCD
    b_rtp = reg.project(b_rys, ['b_oct', 'g_gcd'])
    # Get the octant id and mode of each point (discarding mode here).
    oc, _mo = b_rys.cm()
    b_oct = b_rys.domain
    # for depth in range(30):
    #     cx = xy_regions(b_rys.coords, _mo, depth)
    #     hd = reg_hex_digits(cx, oc, b_oct, TailStyle.key)
    #     print(f'depth={depth}, cx={cx[0]}, hx={hd[0]}')

    # depth=0, cx=[22 42], hx=[7 5]
    # depth=1, cx=[22 42 57], hx=[7 7 1]
    # Labels are depth+2 long.  Consider layer 0:  they still need 0...B.
    # However, for the purpose of reliable keys we need the c2 of parent/mode of parent
    # Now generate the set of h9f labels for each point, and convert to np.array
    labels = f'{b_rys:h9.{layers}}'  # This is reversible addresses.
    label_vec = np.array(labels.splitlines())
    # the h9f formatter h9f is used to revert this array back to barycentric.
    h9_r = h9f.revert(labels)  # Convert from addresses.
    # we now take these and project them back to GCD.
    l_rtp = reg.project(h9_r, ['b_oct', 'g_gcd'])
    # We will use the GeographicLib Inverse to find ∂-distances (metres)
    # and multiply the values by 1e+9 such that our ∂-distances are in nanometres.
    b_deltas = wgs84(refs.coords, b_rtp.coords) * 1e+9
    l_deltas = wgs84(refs.coords, l_rtp.coords) * 1e+9
    for i in range(refs.coords.shape[0]):
        # Now write to the logger.
        logger.write({
            "lat_in": f"{refs.coords[i][0]:.16f}", "lon_in": f"{refs.coords[i][1]:.16f}",
            "bry_x": f"{b_rys.coords[i][0]:.16f}", "bry_y": f"{b_rys.coords[i][1]:.16f}",
            "oct_in": int(oc[i]),
            "ambiguous": ambiguous[i],
            "lat_rt_gb": f"{b_rtp.coords[i][0]:.16f}", "lon_rt_gb": f"{b_rtp.coords[i][1]:.16f}",
            "gcd_err": f'{b_deltas[i]:.6f}nm',
            "gcd_err_val": f'{b_deltas[i]:.6f}',
            "grid_addr": label_vec[i],
            "lat_rt_ga": f"{l_rtp.coords[i][0]:.16f}", "lon_rt_ga": f"{l_rtp.coords[i][1]:.16f}",
            "grid_err": f'{l_deltas[i]:.6f}nm',
            "grid_err_val": f'{l_deltas[i]:.6f}',
        })


if __name__ == '__main__':
    accuracy = 1e-9  # in meters
    reg = Registrar()  # Manage Domains & Projections
    seed = 1234512
    samples = 100_000
    layers = 36  # 36 reaches mathematical limits on 64-bit floats.
    np.random.seed(seed)
    base = Path(__file__).parent

    log_file = base.joinpath(f'logs/L{layers}_{samples}_{seed}.csv')
    main_logger = CSVLogger(log_file)
    start_time = time.perf_counter()

    locr = gcd_rnd(samples)
    gpts = Points(locr, 'g_gcd')
    run(reg, main_logger, gpts, layers=layers)
    main_logger.close()
    elapsed = time.perf_counter() - start_time
    print(f'Results written to {log_file.relative_to(base)}\n'
          f'Completed {len(gpts)} points in {elapsed:.3f} seconds'
          f' ({elapsed / len(gpts):.6f} sec/pt)')
