import sys; sys.path.insert(0, '.')
import numpy as np
import uuid as uuid_mod
from hhg9.h9.uuid_address import h9_bin, _batch_int_to_nibbles
from hhg9.h9.region import region_neighbours, H9R
from hhg9.h9 import H9C, H9K
from hhg9.h9.lattice import H9C as LC

# A known L29-diverging full uuid (input), and its Python L29 bin.
hexs = ["26365137507244657866710401541662",
        "98685408477266574787733774022064",
        "a0628402342087286432274114160164"]
for hx in hexs:
    u = uuid_mod.UUID(hex=hx)
    pb = h9_bin([u], 29)[0]
    nb = _batch_int_to_nibbles([u.int], n=32)[0]
    pbn = _batch_int_to_nibbles([pb.int], n=32)[0]
    print(f"\n=== full {hx}")
    print(f"  full nibbles    : {''.join(format(x,'x') for x in nb)}")
    print(f"  py bin L29      : {''.join(format(x,'x') for x in pbn)}")
    print(f"  full nib[27,28,29,30,31] = {nb[27]},{nb[28]},{nb[29]},{nb[30]} tail={nb[31]:x}")
    print(f"  pyL29 nib[29,30,31]      = {pbn[29]},{pbn[30]:x},{pbn[31]:x}")

# Reconstruct the L29 region chain from geometry for these by brute: we don't have
# coords. Instead decode->reproject. Use h9_decode then re-derive regions.
from hhg9 import Registrar, Points
from hhg9.h9.uuid_address import h9_decode
from hhg9.h9.region import xy_regions
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
reg = Registrar(); g=reg.domain('g_gcd'); b=reg.domain('b_oct')
for hx in hexs:
    u = uuid_mod.UUID(hex=hx)
    lat, lon = h9_decode([u])
    bp = reg.project(Points(np.column_stack([lat, lon]), g), [g, b])
    coords = bp.coords; oc, mo = bp.cm()
    L = 29
    regions = xy_regions(coords, mo, L)[0]
    x, y = coords[0,0], coords[0,1]
    loc = location(np.array([H9K.R3*x]), np.array([y]), np.array([mo[0]]))[0]
    print(f"\n=== {hx}")
    print(f"  region chain (cids) len={len(regions)}: {[hex(int(v)) for v in regions]}")
    print(f"  modes per cid: {[int(H9C.mode[v]) for v in regions]}")
    print(f"  I=regions[-2]={hex(int(regions[-2]))} mode={int(H9C.mode[regions[-2]])}  C=regions[-1]={hex(int(regions[-1]))}")
    print(f"  location={loc} (EXT={loc==BaryLoc.EXT})")
    nbr, c2 = region_neighbours(regions[None, :])
    print(f"  region_neighbours -> nbr={[hex(int(v)) for v in nbr[0]]}  c2={int(c2[0])}")
    print(f"  changed? {not np.array_equal(regions, nbr[0])}")
