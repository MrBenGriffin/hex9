"""Scratch: test whether including body[:,1] (not treating it as a sentinel)
makes the forward->inverse round-trip identity hold."""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import hex_layer, TailStyle
from hhg9.h9 import HEX_LUTS
from hhg9.h9.uvc import _LUT, _REV
from hhg9.h9.tail import tail_unpack_reversible


def fwd(hx, lo_col):
    """Forward including body cols [lo_col .. depth-1]."""
    body = hx[:, :-1]
    tail = hx[:, -1]
    sz, depth = body.shape
    c2, _r, c_mo = tail_unpack_reversible(tail)
    hr = HEX_LUTS.hex_reg
    uv_stack = []
    cm = c_mo.copy(); c2t = c2.copy()
    for i in range(depth - 1, lo_col - 1, -1):
        d = body[:, i]
        uv_stack.append(_LUT[d, cm, c2t].astype(np.int32))
        rmc = hr[d, cm, c2t]; cm = rmc[:, 1]; c2t = rmc[:, 2]
    qr = np.zeros((sz, 2), dtype=np.int32)
    for uv in reversed(uv_stack):
        qr = 3 * qr + uv
    return qr[:, 0], qr[:, 1], c2


def inv(u, v, r_mo, levels):
    """DFS recover `levels` digits leaf->root; return digs or None."""
    qr_root = np.array([u, v], dtype=np.int32)
    u_mod3 = int(u) % 3
    if u_mod3 < 0: u_mod3 += 3
    starts = {(cm, c2v) for cm in range(2) for c2v in range(3) if (cm, c2v, u_mod3) in _REV}
    for lcm, lc2 in starts:
        stack = [(levels, qr_root.copy(), lcm, lc2, [])]
        while stack:
            lvl, qr, cm, c2v, digs = stack.pop()
            if lvl == 0:
                if cm == r_mo and qr[0] == 0 and qr[1] == 0:
                    return digs
                continue
            m = int(qr[0]) % 3
            if m < 0: m += 3
            for d, p_mo, p_c2, uv in _REV.get((cm, c2v, m), []):
                rem = qr - np.array(uv, dtype=np.int32)
                if rem[0] % 3 == 0 and rem[1] % 3 == 0:
                    stack.append((lvl - 1, rem // 3, p_mo, p_c2, digs + [d]))
    return None


reg = Registrar(); g = reg.domain('g_gcd'); b = reg.domain('b_oct')
for lo in (1, 2):
    print(f"=== include cols [{lo}..depth-1]  (lo=2 is current code, lo=1 is the fix) ===")
    for lat, lon, lab in [(0, 10, 'A'), (0, 20, 'B'), (0, 30, 'C')]:
        pt = Points(np.array([[float(lat), float(lon)]]), domain=g)
        pb = reg.project(pt, [g, b])
        hv = hex_layer(pb, layer=4, tail_style=TailStyle.reversible)
        body = hv[:, :-1]; depth = body.shape[1]
        u, v, c2 = fwd(hv, lo)
        _c2, r_mo, _p = tail_unpack_reversible(hv[:, -1])
        levels = depth - lo
        digs = inv(int(u[0]), int(v[0]), int(r_mo[0]), levels)
        want = [int(x) for x in body[0, lo:]]            # root->leaf
        got = list(reversed(digs)) if digs else None     # root->leaf
        ok = got == want
        print(f"  {lab} uv=({int(u[0]):+d},{int(v[0]):+d}) want{want} got{got} {'OK' if ok else 'MISMATCH'}")
    print()
