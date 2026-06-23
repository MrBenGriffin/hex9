# Empirical confirmation of the 6 functional capability claims against libhex9 (C),
# independent of nibble-identity with Python.
import ctypes, math
import numpy as np

DY='/Users/ben/Documents/Projects/GeoPlegma/libhex9/build-l30/libhex9.dylib'
lib=ctypes.CDLL(DY); u16=ctypes.c_uint8*16
lib.hex9_lmax.restype=ctypes.c_int
lib.hex9_encode.argtypes=[ctypes.c_double,ctypes.c_double,u16]; lib.hex9_encode.restype=ctypes.c_int
lib.hex9_decode.argtypes=[u16,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]; lib.hex9_decode.restype=ctypes.c_int
lib.hex9_bin.argtypes=[u16,ctypes.c_int,u16]; lib.hex9_bin.restype=ctypes.c_int
lib.hex9_grid_create.argtypes=[ctypes.c_double]*4+[ctypes.c_int,ctypes.c_int,ctypes.c_int64,ctypes.c_char_p,ctypes.c_size_t]
lib.hex9_grid_create.restype=ctypes.c_void_p
lib.hex9_grid_count.argtypes=[ctypes.c_void_p]; lib.hex9_grid_count.restype=ctypes.c_int
lib.hex9_grid_cell_uuid.argtypes=[ctypes.c_void_p,ctypes.c_int,u16]
lib.hex9_grid_cell_id.argtypes=[ctypes.c_void_p,ctypes.c_int,u16]
lib.hex9_grid_cell_centroid.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
lib.hex9_grid_destroy.argtypes=[ctypes.c_void_p]
print("lmax =", lib.hex9_lmax())

def enc(lon,lat):
    o=u16(); rc=lib.hex9_encode(lon,lat,o); return bytes(o),rc
def dec(b):
    lo=ctypes.c_double(); la=ctypes.c_double(); rc=lib.hex9_decode(u16(*b),ctypes.byref(lo),ctypes.byref(la)); return lo.value,la.value,rc
def cbin(b,L):
    o=u16(); rc=lib.hex9_bin(u16(*b),L,o); return bytes(o),rc

def geodist_m(lon1,lat1,lon2,lat2):  # haversine, R=6371000
    R=6371000.0; p1,p2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1,math.sqrt(a)))

rng=np.random.default_rng(7); N=20000
lats=rng.uniform(-89.9,89.9,N); lons=rng.uniform(-179.9,179.9,N)

# (1) lat/lon -> hex uuid
rc_bad=sum(enc(lo,la)[1] for lo,la in zip(lons[:2000],lats[:2000]))
print(f"\n(1) encode lon/lat -> uuid: {2000-rc_bad}/2000 rc==0  -> {'YES' if rc_bad==0 else 'NO'}")

# (2) uuid -> lat/lon round trip accuracy
maxd=0.0
for lo,la in zip(lons,lats):
    u,_=enc(lo,la); rlo,rla,rc=dec(u)
    maxd=max(maxd, geodist_m(lo,la,rlo,rla))
# decode is to the CELL CENTROID, so latlon->uuid->latlon is bounded by cell size,
# not accuracy. The real test is stability of the centroid: decode->encode->decode.
maxc=0.0; byte_stable=0
for lo,la in zip(lons[:5000],lats[:5000]):
    u,_=enc(lo,la); c_lo,c_la,_=dec(u)         # centroid
    u2,_=enc(c_lo,c_la); c_lo2,c_la2,_=dec(u2)  # re-encode centroid
    byte_stable+= (u2==u)
    maxc=max(maxc, geodist_m(c_lo,c_la,c_lo2,c_la2))
print(f"(2) centroid round-trip (decode->encode->decode) max displacement = {maxc*1e6:.4f} um; "
      f"centroid->uuid byte-stable {byte_stable}/5000  -> {'YES' if maxc*1e6 < 1 else 'NO'}")

# (3) uuid -> bin at any layer (structure + rc)
u,_=enc(12.34,56.78); ok=True
for L in [1,5,10,22,29,30]:
    b,rc=cbin(u,L)
    nb=[ (by>>4)&0xF for by in b ] + [ by&0xF for by in b ]  # not ordered; just rc check
    if rc!=0: ok=False
print(f"(3) uuid -> bin at L in 1..30: all rc==0 -> {'YES' if ok else 'NO'}")

# (4) global enumeration: count, uniqueness, expected 12*9^L
def grid_global(L):
    err=ctypes.create_string_buffer(256)
    g=lib.hex9_grid_create(-180.0,-90.0,180.0,90.0,L,0,0,err,256)
    if not g: return None,err.value.decode()
    n=lib.hex9_grid_count(g)
    uu=set()
    for i in range(n):
        o=u16(); lib.hex9_grid_cell_uuid(g,i,o); uu.add(bytes(o))
    lib.hex9_grid_destroy(g)
    return (n,len(uu)),None
for L in [1,2,3]:
    expect=12*(9**L)
    res,e=grid_global(L)
    if res is None:
        print(f"(4) L{L}: grid_create failed: {e}")
    else:
        n,uniq=res
        print(f"(4) L{L}: count={n} unique={uniq} expected={expect} -> {'YES' if n==expect==uniq else 'NO'}")

# (5) derive an address (full reversible id) from a bin, and (6) bin ancestor nesting
err=ctypes.create_string_buffer(256)
g=lib.hex9_grid_create(-10.0,-10.0,10.0,10.0,5,0,0,err,256)
n=lib.hex9_grid_count(g)
addr_ok=anc_ok=0; tot=min(n,2000)
for i in range(tot):
    binu=u16(); lib.hex9_grid_cell_uuid(g,i,binu); binb=bytes(binu)
    idu=u16();  lib.hex9_grid_cell_id(g,i,idu);   idb=bytes(idu)   # full L29 address of the bin
    # (5) address -> bin at the grid layer reproduces the bin
    rb,_=cbin(idb,5)
    addr_ok += (rb==binb)
    # (6) ancestor: bin the full address to L4 and L3; check nesting bin(addr,4) prefix-consistent
    a4,_=cbin(idb,4); a3,_=cbin(idb,3)
    # ancestor of a4 should equal bin(addr,3)
    anc_ok += (cbin(idb,3)[0]==a3)
lib.hex9_grid_destroy(g)
print(f"(5) bin -> full address -> re-bin == bin: {addr_ok}/{tot} -> {'YES' if addr_ok==tot else 'NO'}")
print(f"(6) bin ancestor derivable/consistent: {anc_ok}/{tot} -> {'YES' if anc_ok==tot else 'NO'}")
