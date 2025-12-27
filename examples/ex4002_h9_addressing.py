# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Proof of concepts taken from AKOctahedral

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd
from hhg9.h9 import HEX_LUTS, H9_RA, H9O
from hhg9.h9.addressing import reg_pack, reg_unpack, hex_str_encode, hex_str_decode, neighbours, hex_pack, hex_unpack, \
    reg_hex_digits, hex_digits_reg
from hhg9.algorithms.packing import u64_layers
from hhg9.h9.region import regions_xy, xy_regions


def poc_region_encode():
    """PoC region encode"""
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    samples = 1000
    scheme = H9_RA
    locr = gcd_rnd(samples)
    refs = Points(locr, 'g_gcd')
    b_refs = reg.project(refs, ['g_gcd', 'b_oct'])  # components filled
    adx = reg_pack(b_refs, 14, reg)
    # oc, mo = b_refs.cm()
    # cx = rg.xy_regions(b_refs.coords, mo, 14)
    # packer = region_packer(pack_fn=u64_pack, unpack_fn=u64_layers, octant_mode=lambda o: np.take(b_oct.oid_mo, o))
    # r_scheme = H9_RA
    # rx = r_scheme.cell2rid[cx]
    # adx = packer.encode(rx, octants=oc)
    # xc = u64_pack(rx)
    cv = u64_layers(adx)
    print(f'xc: {adx[0][0]:0x}')
    print("nibbles:", cv[0, :8])
    prt = reg_unpack(adx, reg, scheme)
    # ocr, dec = packer.decode(adx)
    # sanity: decoded proto (col 0) must equal mapping(octant)
    # assert np.array_equal(dec[:, 0], np.take(b_oct.oid_mo, oc)), "proto restored from octant mismatch"
    # assert np.array_equal(ocr, oc), "octant restore failed"
    # cells = scheme.rid2cell[dec]
    # reg_mo_rt = rg.regions_xy(cells)
    # reg_rt = reg_mo_rt[:, :2]  # just want the x,y.
    # prt = Points(reg_rt, b_oct, components=ocr)
    r_tp = reg.project(prt, ['b_oct', 'g_gcd'])  # components filled
    b_deltas = wgs84(refs.coords, r_tp.coords) * 1e+0
    print(f"mean(delta): {np.mean(b_deltas)}m; max(b_deltas): {np.max(b_deltas)}m")


def poc_stonehenge_h9():
    """hex9 address of stonehenge."""
    reg = Registrar()
    ref = Points(np.array([[51.1787980000210725, -1.8261898473293335]]), 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])  # components filled
    ad = hex_str_encode(b_ref)
    print(ad[0])
    ptb = hex_str_decode(ad, reg)
    #  reference value (from above)                      51.1787980000210725, -1.8261898473293335
    b_bak = reg.project(ptb, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    rx = wgs84(ref.coords, b_bak.coords) * 1e+9
    print(f'{rx[0]}nm')  #


def poc_hex_str_rnd():
    """hex9 address of rnd."""
    reg = Registrar()
    locr = gcd_rnd(1)
    ref = Points(locr, 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])  # components filled
    depth = 8
    h9x = hex_str_encode(b_ref, depth)
    print(f'{h9x}')
    n_ref = neighbours(b_ref, depth)
    h9n = hex_str_encode(n_ref, depth)
    k_ref = neighbours(n_ref, depth)
    h9k = hex_str_encode(k_ref, depth)
    for (o, n, k, p, c) in zip(h9x, h9n, h9k, b_ref.coords, b_ref.components):
        print(f'{o}={k} via {n}, ({p}):{c}')
    pbk = hex_str_decode(h9x, reg)
    b_bak = reg.project(pbk, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    rx = wgs84(ref.coords, b_bak.coords) * 1e+9
    print(f'{rx}')  #


def poc_hex_pak():
    """hex9 address of rnd."""
    reg = Registrar()
    locr = gcd_rnd(100)
    ref = Points(locr, 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])  # components filled
    packed = hex_pack(b_ref, 12)  # 3 metres - not great!
    rtp = hex_unpack(packed, reg)
    b_bak = reg.project(rtp, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    rx = wgs84(ref.coords, b_bak.coords)
    print(f'{rx}')  #


if __name__ == "__main__":
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    exm = np.array([[0x16, 0x39, 0x3A, 0x35]])
    oid = 4
    vx_m = regions_xy(exm)
    vx = vx_m[:, :-1]
    cmp = H9O.oid_cmp[oid]
    pts = Points(vx, b_oct, cmp)
    o_id, mode = pts.cm()
    ggg = reg.project(pts, ['b_oct', 'g_gcd'])

    rn = xy_regions(pts.coords, mode, 2)
    assert np.all(rn == exm)

    hex_reg = HEX_LUTS.hex_reg
    scheme = H9_RA
    rid = scheme.cell2rid[rn[0]]
    print(f"rn: {rn[0]}; rid: {rid}; octant: {o_id}")

    po = reg_hex_digits(rn, o_id, b_oct)
    print(f"octant.0: {o_id} cells.0: {rn[0]} => reg_hex_digits() => hex_digits.0: {po[0]}")
    oc, ba = hex_digits_reg(b_oct, po)
    print(f"hex_digits.0: {po[0]} => hex_digits_reg() => octant.1: {oc} cells.1: {ba[0]}")
    vb = reg_hex_digits(ba, oc, b_oct)
    print(f"octant.1: {oc} cells.1: {ba[0]} => reg_hex_digits() => hex_digits.1: {vb[0]}")

    exit(0)

    vx_m = regions_xy(exm)

    print("\npoc_hex_str_rnd")
    poc_hex_str_rnd()

    print("\npoc_hex_pak")
    poc_hex_pak()

    print("\npoc_stonehenge_h9")
    poc_stonehenge_h9()

    print("\npoc_region_encode")
    poc_region_encode()
