# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - uses H9 addresses and round-trips them.
This:
(a) Generates a Reference set of GCD random (seeded) addresses
(b) Projects them onto Octahedral Barycentric coordinates.
(c.1) Projects them back to GCD for roundtrip testing
(c.2.a) Renders them as H9 UUID
(c.2.b) Reverts the UUID back to Octahedral Barycentric.
(c.2.c) Projects reverted back to GCD for roundtrip testing.
(d) Measures, via GeographicLib Inverse, the ∂ distance from Reference
(e) Logs the results into a CSV file.
# ⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 303.4s / 13.9s acc
19 May 2026 (passed)
"""
import csv
import json
import time
import numpy as np
from pathlib import Path
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd
from hhg9.h9.uuid_address import h9_enc, h9_dec, h9_enc_ext


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


class CSVLogger:
    """Append rows to a CSV with a fixed header; creates parent dirs as needed."""
    HEADER = [
        "lat_in", "lon_in", "oid", "mode", "braw_x", "braw_y", "bary_x", "bary_y",  "gcd_err", "uuid_addr", "uuid_err"
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


def run(reg, logger, refs, b_oct):
    print('10k is likely to take about 12s')
    print('projecting g_gcd->b_oct')
    good = refs.domain.valid(refs)
    if np.any(~good):
        print('Invalid LL points detected')
        g_bad = refs.select(~good)
        for ll in g_bad:
            print(f'g_gcd: {ll.coords}')
        refs = refs.select(good)
    b_rys = reg.project(refs, ['g_gcd', 'b_oct'])  # components filled
    good = b_rys.domain.valid(b_rys)
    if np.any(~good):
        idx = np.flatnonzero(~good)
        print(f'Invalid Barycentric points detected: {idx}')
        g_bad = refs.select(~good)
        b_bad = b_rys.select(~good)
        for (ll, by) in zip(g_bad, b_bad):
            print(f'g_gcd: {ll.coords} -> b_oct: {by.coords}, {by.oid}')
        refs = refs.select(good)
        b_rys = b_rys.select(good)
    b_raw = reg.project(b_rys, ['b_oct', 'b_raw'])  # step back one.
    print('projecting b_oct->g_gcd')
    b_rtp = reg.project(b_rys, ['b_oct', 'g_gcd'])
    print('calculating ∂-distances wgs84 distances between reference and reverted b_oct in nm (nanometres)')
    b_deltas = wgs84(refs.coords, b_rtp.coords) * 1e+9
    # Get the octant id and net_mode of each point (discarding net_mode here).
    oc, mo = b_rys.cm()
    uuid_by = h9_enc_ext(b_rys, oc, mo)
    uuid_rt = h9_dec(uuid_by, b_oct)
    u_rtp = reg.project(uuid_rt, ['b_oct', 'g_gcd'])
    print('calculating ∂-distances wgs84 distances between reference and reverted uuid b_oct in nm (nanometres)')
    u_deltas = wgs84(refs.coords, u_rtp.coords) * 1e+9
    max_delta = np.max(u_deltas)
    print(f'Max uuid ∂: {max_delta}; storing results in csv')
    for i in range(refs.coords.shape[0]):
        # Now write to the logger.
        logger.write({
            "lat_in": f"{refs.coords[i][0]:.16f}", "lon_in": f"{refs.coords[i][1]:.16f}",
            "oid": int(oc[i]),
            "mode": int(mo[i]),
            "braw_x": f"{b_raw.coords[i][0]:.16f}", "braw_y": f"{b_raw.coords[i][1]:.16f}",
            "bary_x": f"{b_rys.coords[i][0]:.16f}", "bary_y": f"{b_rys.coords[i][1]:.16f}",
            "gcd_err": f'{b_deltas[i]:.12f}',
            "uuid_addr": str(uuid_by[i]),
            "uuid_err": f'{u_deltas[i]:.12f}',
        })


if __name__ == '__main__':
    from hhg9.h9.region import recover_stats_reset, recover_stats_report
    reg = Registrar()  # Manage Domains & Projections
    locs = json_load('../assets/locations.json')
    ll = list()
    for region in locs:
        for spot in locs[region]:
            val = locs[region][spot]
            lat, lon = tuple(val)
            ll.append((float(lat), float(lon)))
    seed = 4007
    samples = 100_000
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    recover_stats_reset()

    np.random.seed(seed)
    base = Path(__file__).parent
    log_file = base.joinpath(f'output/logs/{samples}_{seed}_uuid.csv')
    main_logger = CSVLogger(log_file)
    loc_csv = np.array(ll)
    print(f'Loaded {loc_csv.shape[0]} addresses from json')
    loc_rnd = gcd_rnd(samples)
    locr = np.vstack((loc_csv, loc_rnd))
    gpts = Points(locr, g_gcd)
    start_time = time.perf_counter()
    run(reg, main_logger, gpts, b_oct)
    main_logger.close()
    elapsed = time.perf_counter() - start_time
    print(f'Results written to {log_file.relative_to(base)}\n'
          f'Completed {len(gpts)} points in {elapsed:.3f} seconds'
          f' ({elapsed / len(gpts):.6f} sec/pt)')
    recover_stats_report()
