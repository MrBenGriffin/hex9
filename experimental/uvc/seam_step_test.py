"""
Validate step_region_oct's seam crossing: where the within-octant step dies at the
boundary, the octant-aware step should land on a geometrically ADJACENT cell in the
oid_nb octant.
"""
import numpy as np
from build_stepper import (make_chain, step_region, step_region_oct,
                           reg, b, g, oid_nb)
from hhg9.h9 import H9O
from hhg9.h9.addressing import reg_hex_digits, hex_decode, TailStyle
nb_c2p = H9O.nb_c2p


def to_xyz(chain, oct):
    hv = reg_hex_digits(np.asarray(chain, dtype=np.uint8)[None, :], oct, b, TailStyle.reversible)
    ptsb = hex_decode(hv, b)
    geo = reg.project(ptsb, [b, g])
    la, lo = np.radians(geo.coords[0])
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


# sample many points; collect unit spacing from within-octant steps, and test seam steps
lats = np.linspace(-80, 80, 25); lons = np.linspace(-175, 175, 40)
spacings, seam = [], []
for la in lats:
    for lo in lons:
        chain, pb, oc = make_chain(la, lo)
        oc = int(oc[0]); src = to_xyz(chain, oc)
        for e in range(3):
            w = step_region(chain, e)
            if w is not None:
                spacings.append(np.linalg.norm(to_xyz(w, oc) - src))
            else:
                r = step_region_oct(chain, oc, e)
                if r is not None:
                    ch2, oc2 = r
                    d = np.linalg.norm(to_xyz(ch2, oc2) - src)
                    seam.append((oc2 == int(oid_nb[oc, e]), oc2 != oc, d, bool(nb_c2p[oc, e])))

s = np.median(spacings)
print(f"unit cell spacing (median within-octant step): {s:.4f}")
print(f"within-octant steps sampled: {len(spacings)}")
print(f"seam steps tested: {len(seam)}")
if seam:
    adj = sum(d < 1.6 * s for _, _, d, _ in seam)
    print(f"  landed adjacent (<1.6 unit): {adj}/{len(seam)}")
    for preserve in (True, False):
        grp = [d for _, _, d, p in seam if p == preserve]
        if grp:
            a = sum(x < 1.6 * s for x in grp)
            tag = "PRESERVE (axial)" if preserve else "SWAP (diagonal)"
            print(f"  {tag:<18}: adjacent {a}/{len(grp)}   "
                  f"dist quantiles {np.round(np.quantile(np.array(grp)/s,[0,.5,1]),2)}")
