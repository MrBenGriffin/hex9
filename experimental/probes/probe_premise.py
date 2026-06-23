# Decisive test of the PORT PLAN premise: are the 521 L29-diverging cells resolved
# by Python's region_neighbours COMBINATORIAL fold (gate mode(I)==1, "active"), or
# by the GEOMETRIC decode->centroid->reclassify path inside h9_bin?
import sys; sys.path.insert(0, '.')
import numpy as np, ctypes
from hhg9.h9.uuid_address import h9_encode, h9_bin, _batch_int_to_nibbles, UUID_DEPTH
from hhg9.h9.region import xy_regions
from hhg9.h9 import H9C, H9K
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9 import Registrar, Points
from hhg9.h9.uuid_address import h9_dec

DY='/Users/ben/Documents/Projects/GeoPlegma/libhex9/build-l30/libhex9.dylib'
lib=ctypes.CDLL(DY); u16=ctypes.c_uint8*16
lib.hex9_bin.argtypes=[u16,ctypes.c_int,u16]
def cbin(b,L):
    o=u16(); lib.hex9_bin(u16(*b),L,o); return bytes(o)

rng=np.random.default_rng(1); n=3000
lats=rng.uniform(-89.5,89.5,n); lons=rng.uniform(-179.5,179.5,n)
uuids=h9_encode(lats,lons); L=29
py=h9_bin(list(uuids),L)
full_nibs=_batch_int_to_nibbles([u.int for u in uuids],n=32)
py_nibs=_batch_int_to_nibbles([p.int for p in py],n=32)

# chain h9_bin actually uses: decode -> b_oct -> xy_regions at L
reg=Registrar(); g=reg.domain('g_gcd'); b=reg.domain('b_oct')
pts=h9_dec(list(uuids), b)            # exactly what h9_bin does internally
coords=pts.coords; oc=pts.oid
mo=H9C.mode  # not used directly
# replicate h9_bin_pts mode handling: oc->mode via H9O
from hhg9.h9 import H9O
pmode = H9O.oid_mo[oc]
regions = xy_regions(coords, pmode, L)          # (N, L+2)
imode = H9C.mode[regions[:,-2]]                  # mode of I (penultimate)
x,y = coords[:,0], coords[:,1]
locs = location(H9K.R3*x, y, pmode)
active = (locs!=BaryLoc.EXT)&(locs!=BaryLoc.UDF)&(imode==1)

div = np.array([cbin(u.bytes,L)!=p.bytes for u,p in zip(uuids,py)])
ndiv = int(div.sum())
print(f"diverging cells (C vs py) at L{L}: {ndiv}/{n}")
print(f"  of diverging: mode(I)==1 (region_nbr active gate): {int(imode[div].sum()==imode[div].astype(bool).sum()) if False else int((imode[div]==1).sum())}")
print(f"  of diverging: active(mode1 & in-scope):            {int(active[div].sum())}")
print(f"  of diverging: mode(I)==0 (NOT folded by Python):   {int((imode[div]==0).sum())}")
print(f"  of diverging: full-uuid nibble[{L}] in wing 6/7/8: {int(np.isin(full_nibs[div,L],[6,7,8]).sum())}")
print(f"  of diverging: py    bin nibble[{L}] in wing 6/7/8: {int(np.isin(py_nibs[div,L],[6,7,8]).sum())}")
print(f"  of diverging: py    bin nibble[{L}] in 0..5:       {int(np.isin(py_nibs[div,L],[0,1,2,3,4,5]).sum())}")
# Does the geometric L29 reclassification ever produce the wing at the terminus?
print(f"\n  (whole corpus) active cells: {int(active.sum())}; diverging cells: {ndiv}")
print(f"  overlap active&diverging: {int((active&div).sum())}")
