# Python(hhg9) <-> C(libhex9) byte-equality cross-check for the reconciled tail.
#
# PRIMARY (projection-free): feed Python-generated UUIDs into C's combinatorial
#   hex9_bin (the identity_from_uuid nibble-walk, NOT a geometric re-encode) and
#   compare byte-for-byte against Python h9_bin at every layer. This isolates the
#   tail/bin/sentinel LAYOUT — no projection in the loop — so any diff is a real
#   layout divergence (the thing we just reconciled), not float noise.
#
# SECONDARY (end-to-end, shallow): same lat/lon through each side's full pipeline
#   (project -> encode -> bin) at L1..6, where 2e-15 projection agreement cannot
#   flip a nibble. Confirms the two independent engines converge.
#
# Zero diffs => the "deviation from the Python" is closed, and this corpus seeds
# the Rust conformance suite.
import sys, ctypes
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from hhg9.h9.uuid_address import h9_encode, h9_bin, h9_label, h9_from_label

DYLIB = next((p for p in (
    Path('/Users/ben/Documents/Projects/GeoPlegma/libhex9/build-l30/libhex9.dylib'),
    Path('/Users/ben/Documents/Projects/GeoPlegma/libhex9/build/libhex9.dylib'),
) if p.exists()), None)
assert DYLIB, "no libhex9.dylib found"

lib = ctypes.CDLL(str(DYLIB))
u16 = ctypes.c_uint8 * 16
lib.hex9_lmax.restype = ctypes.c_int
lib.hex9_encode.argtypes = [ctypes.c_double, ctypes.c_double, u16]
lib.hex9_bin.argtypes = [u16, ctypes.c_int, u16]
lib.hex9_label_key.argtypes = [u16, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
LMAX = lib.hex9_lmax()
print(f"lib={DYLIB.parent.name}  lmax={LMAX}")


def c_bin(uuid_bytes, layer):
    out = u16()
    rc = lib.hex9_bin(u16(*uuid_bytes), layer, out)
    return (bytes(out), rc)


def c_label(uuid_bytes, layer):
    buf = ctypes.create_string_buffer(64)
    n = lib.hex9_label_key(u16(*uuid_bytes), layer, buf, 64)
    return buf.value.decode() if n >= 0 else f"<rc={n}>"


def c_encode(lon, lat):
    out = u16()
    lib.hex9_encode(lon, lat, out)
    return bytes(out)


def primary(n, layers):
    rng = np.random.default_rng(1)
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)
    uuids = h9_encode(lats, lons)                       # Python's UUIDs
    print(f"\nPRIMARY (projection-free) n={n}")
    for L in layers:
        py = h9_bin(list(uuids), L)                     # Python bins
        bin_bad = lbl_bad = 0
        for pu, u in zip(py, uuids):
            cb, rc = c_bin(u.bytes, L)
            if cb != pu.bytes:
                bin_bad += 1
            if c_label(u.bytes, L) != h9_label(pu):
                lbl_bad += 1
        flag = "" if (bin_bad == lbl_bad == 0) else "  <-- DIFF"
        print(f"  L{L:<2d}: bin diff {bin_bad:>5d}/{n}   label diff {lbl_bad:>5d}/{n}{flag}")


def secondary(n, layers):
    rng = np.random.default_rng(2)
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)
    py_full = h9_encode(lats, lons)
    print(f"\nSECONDARY (end-to-end, shallow) n={n}")
    for L in layers:
        py = h9_bin(list(py_full), L)
        bad = 0
        for lo, la, pu in zip(lons, lats, py):
            cb, _ = c_bin(c_encode(lo, la), L)
            if cb != pu.bytes:
                bad += 1
        flag = "" if bad == 0 else "  <-- DIFF"
        print(f"  L{L:<2d}: bin diff {bad:>5d}/{n}{flag}")


def dump_diffs(n, L, k=6):
    """Nibble-level dump of the first k bin diffs at layer L (projection-free)."""
    rng = np.random.default_rng(1)                       # same seed as primary()
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)
    uuids = h9_encode(lats, lons)
    py = h9_bin(list(uuids), L)
    print(f"\nDIFF DUMP L{L} (nibbles 0..31; ^ marks differing positions)")
    shown = 0
    for pu, u in zip(py, uuids):
        cb, _ = c_bin(u.bytes, L)
        if cb == pu.bytes:
            continue
        pn = [b for byte in pu.bytes for b in ((byte >> 4) & 0xF, byte & 0xF)]
        cn = [b for byte in cb for b in ((byte >> 4) & 0xF, byte & 0xF)]
        mark = ''.join('^' if a != b else ' ' for a, b in zip(pn, cn))
        print(f"  py {''.join(format(x,'x') for x in pn)}")
        print(f"  C  {''.join(format(x,'x') for x in cn)}")
        print(f"     {mark}")
        shown += 1
        if shown >= k:
            break


def trace(lat, lon, note=""):
    """Per-layer L0..LMAX bin trace, Python vs C, for one point (projection-free)."""
    u = h9_encode([lat], [lon])[0]
    print(f"\nTRACE ({lat:.5f}, {lon:.5f}) {note}\n  uuid={u.hex}")
    print(f"  {'L':>2} | {'python label':<34} | {'C label':<34} |")
    for L in range(0, LMAX + 1):
        py = h9_bin([u], L)[0]
        cb, rc = c_bin(u.bytes, L)
        pl, cl = h9_label(py), c_label(u.bytes, L)
        print(f"  {L:>2} | {pl:<34} | {cl:<34} | {'ok' if cb == py.bytes else 'DIFF'}")


def pick_and_trace(n=4000, L=29):
    """Find 2 points that diverge at layer L plus 1 clean control, and trace each."""
    rng = np.random.default_rng(11)
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)
    uuids = h9_encode(lats, lons)
    py = h9_bin(list(uuids), L)
    div, clean = [], []
    for i, (pu, u) in enumerate(zip(py, uuids)):
        cb, _ = c_bin(u.bytes, L)
        (div if cb != pu.bytes else clean).append(i)
        if len(div) >= 2 and len(clean) >= 1:
            break
    for i in div[:2]:
        trace(lats[i], lons[i], note=f"(diverges at L{L})")
    if clean:
        trace(lats[clean[0]], lons[clean[0]], note="(control: clean at L%d)" % L)


def _tail_fields(nib31):
    """Decode tail nibble per amended layout (p_mo<<3)|(c2<<1)|r_mo."""
    return ((nib31 >> 3) & 1, (nib31 >> 1) & 3, nib31 & 1)   # p_mo, c2, r_mo


def characterise(n, L=29):
    """Over diverging cells at layer L: do r_mo/p_mo ever change, or only c2?
    Tabulate c2 transitions and terminal-body-nibble deltas (the c2-seed test)."""
    rng = np.random.default_rng(1)
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)
    uuids = h9_encode(lats, lons)
    py = h9_bin(list(uuids), L)
    rmo_chg = pmo_chg = c2_chg = total = 0
    c2_trans, termnib = {}, {}
    for pu, u in zip(py, uuids):
        cb, _ = c_bin(u.bytes, L)
        if cb == pu.bytes:
            continue
        total += 1
        p_pmo, p_c2, p_rmo = _tail_fields(pu.bytes[15] & 0x0F)
        c_pmo, c_c2, c_rmo = _tail_fields(cb[15] & 0x0F)
        rmo_chg += (p_rmo != c_rmo)
        pmo_chg += (p_pmo != c_pmo)
        c2_chg += (p_c2 != c_c2)
        c2_trans[(p_c2, c_c2)] = c2_trans.get((p_c2, c_c2), 0) + 1
        # terminal body nibble (nibble L, MSB-first): byte L//2, hi/lo
        pn = (pu.bytes[L // 2] >> 4 if L % 2 == 0 else pu.bytes[L // 2] & 0xF)
        cn = (cb[L // 2] >> 4 if L % 2 == 0 else cb[L // 2] & 0xF)
        termnib[(pn, cn)] = termnib.get((pn, cn), 0) + 1
    print(f"\nCHARACTERISE L{L}: {total} diverging cells")
    print(f"  r_mo changed: {rmo_chg}   p_mo changed: {pmo_chg}   c2 changed: {c2_chg}")
    print(f"  c2 transitions (py->C): " +
          ", ".join(f"{a}->{b}:{c}" for (a, b), c in sorted(c2_trans.items())))
    print(f"  terminal-nibble (py->C), top: " +
          ", ".join(f"{a:x}->{b:x}:{c}" for (a, b), c in
                    sorted(termnib.items(), key=lambda kv: -kv[1])[:8]))


if __name__ == '__main__':
    primary(3000, [1, 2, 4, 8, 15, 22, 29, 30])
    secondary(3000, [1, 2, 3, 4, 5, 6])
    dump_diffs(3000, 29)
    pick_and_trace()
    characterise(6000, 29)
