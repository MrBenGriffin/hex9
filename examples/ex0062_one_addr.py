# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
uses single address and walks through it.
16 Jun 2026 0.1.3a0 (passed) 6.7s
13 Mar 2026 0.1.1a1 (passed)
02 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
21 Dec 2025 0.1.0a3 (passed; after rewriting formatter)

"""
import json
import numpy as np
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9, DMS
from hhg9.algorithms.distance import wgs84
from hhg9.h9 import H9_RA, TailStyle
from hhg9.h9.addressing import Style, reg_hex_digits
import hhg9.h9.region as rg
from hhg9.h9.uuid_address import h9_enc, h9_dec, h9_bin_pts

if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9f formats.
    """
    reg = Registrar()  # Manage Domains & Projections
    h9f = OctahedralH9(reg)  # formatter.
    dms = DMS(reg)
    b_oct = reg.domain('b_oct')
    b_oct.no_lib()
    g_gcd = reg.domain('g_gcd')
    b_oct.register_format(h9f)
    g_gcd.register_format(dms)
    # b_oct.no_warp()

    # name = 'Null Antipodes'
    # name = 'Greenwich test'
    ll = np.atleast_2d([
        [0.014836,   179.99999]
        #[32.8010970000000000, -19.4490900000000000]
        # [55.94480923423753, -3.187282666241978]
        # [52.520933833691204,   13.405313674395048970],
    #   43486836-6783-ffff-ffff-fffffffffff2
    ])

    #   -- A point well inside oct 0 (Berlin-ish, away from any seam)
    #   SET hex9.encoder = 'nn';
    #   SELECT h9_encode(ST_SetSRID(ST_MakePoint(13.405313674395048970, 52.520933833691204), 4326)) AS nn;
    #   SET hex9.encoder = 'containment';
    #   SELECT h9_encode(ST_SetSRID(ST_MakePoint(13.405313674395048970, 52.520933833691204), 4326)) AS containment;
    #print(w.do(np.array([[0.20786011,-0.26302651]]), mo=1))
    pos = Points(np.array(ll), g_gcd)
    # print(name)
    print(f'{pos:dms} (Reference Coordinate)')
    bel = reg.project(pos, [g_gcd, 'c_ell'])
    print(f'ECEF: {bel.coords[0,0]:.8f},{bel.coords[0,1]:.8f},{bel.coords[0,2]:.8f}')

    boc = reg.project(pos, ['c_ell', 'c_oct'])
    print(f'OCTA: {boc.coords[0, 0]:.8f},{boc.coords[0, 1]:.8f},{boc.coords[0, 2]:.8f}, face:{boc.oid[0]} (via AK)')

    brw = reg.project(boc, ['c_oct', 'b_oct', 'b_raw'])
    oc0, mo0 = brw.cm()
    print(f'BRAW: {brw.coords[0, 0]:.8f},{brw.coords[0, 1]:.8f}, octant:{oc0}, mode:{mo0}')

    bro = reg.project(boc, ['c_oct', 'b_oct'])
    oc1, mo1 = bro.cm()
    print(f'BARY: {bro.coords[0, 0]:.8f},{bro.coords[0, 1]:.8f}, octant:{oc1}, mode:{mo1}')

    bry = reg.project(pos, [g_gcd, b_oct])  #
    oc2, mo2 = bry.cm()
    print(f'Direct: {bry.coords[0,0]:.8f},{bry.coords[0, 1]:.8f}, octant:{oc2}, mode:{mo2}')

    uuid = h9_enc(bry)
    print(f'UUID FD: {uuid}')
    rxd = h9_dec(uuid, b_oct)
    uuid_rt = h9_enc(rxd)
    print(f'UUID RT: {uuid_rt}')
    oc3, mo3 = rxd.cm()
    print(f'UUID b_oct RT: {rxd.coords[0, 0]:.8f},{rxd.coords[0, 1]:.8f}, octant:{oc3}, mode:{mo3}')

    cxx = rg.xy_regions(bry.coords, mo3)
    cl = ' '.join([f'{i:02x}' for i in list(cxx[0])])
    print(f'Cells: {cl}')
    rxx = H9_RA.cell2rid[cxx]
    cl = ' '.join([f'{i:01X}' for i in list(rxx[0])])
    print(f'Regions: {cl}')

    hxr = reg_hex_digits(cxx, oc3, b_oct, TailStyle.reversible)   # TailStyle.reversible = default.
    hxk = reg_hex_digits(cxx, oc3, b_oct, TailStyle.key)
    xr = ' '.join([f'{i:01X}' for i in list(hxr[0])])
    xk = ' '.join([f'{i:01X}' for i in list(hxk[0])])
    print(f'Hex digits (rev): {xr}')
    print(f'Hex digits (key): {xk}')

    rtp = reg.project(bry, [b_oct, g_gcd])  # sph rt..
    dif = wgs84(pos.coords, rtp.coords) * 1e+9

    cxx = rg.xy_regions(bry.coords, mo3)
    xym = rg.regions_xy(cxx)
    uvr = Points(xym[:, :2], oid=oc3, domain=b_oct)
    rgx = [H9_RA.cell2rid[i] for i in cxx]
    rga = [''.join([f'{a:x}' for a in p]) for p in rgx]
    hxx = h9f.format(bry, None, 'r35')
    ub1 = h9f.revert(hxx, Style.UR64)
    rrp = reg.project(uvr, [b_oct, g_gcd])
    ltp = reg.project(ub1, [b_oct, g_gcd])
    ltp.domain = g_gcd
    rif = wgs84(pos.coords, rrp.coords) * 1e+9
    lif = wgs84(pos.coords, ltp.coords) * 1e+9

    print(f'Regions {cxx}')
    print(f'{pos:dms} (Reference Coordinates)')
    print(f'{ltp:dms} (Label GCD Coordinates)')

    print(f'∂{dif[0]:.6f}nm (roundtrip via GCD<->Barycentric)')
    print(f'∂{rif[0]:.6f}nm (roundtrip via GCD<->Bary Regions)')
    print(f'∂{lif[0]:.6f}nm (roundtrip via GCD<->Hex9 Label)')
    print(f'H9.adr:{bry:h9}')
    print(f'H9.key:{bry:h9.k}')
    uuid = h9_enc(bry)
    bbk = h9_dec(uuid, b_oct) # decode from uuid of h9_enc
    for layer in range(30, 0, -1):
        uuid_b = h9_bin_pts(bbk, layer)
        pt_b = h9_dec(uuid_b, b_oct)
        ltp = reg.project(pt_b, [b_oct, g_gcd])
        uif = wgs84(pos.coords, ltp.coords)
        print(f'UUID {uuid} - binned to L{layer}: {uuid_b} ∂x: {uif[0]:.18f}m')


    print(f'Reference BRY: {bry.coords[0, 0]:.18f},{bry.coords[0, 1]:.18f}')
    print(f'Label RT  BRY: {uvr.coords[0, 0]:.18f},{uvr.coords[0, 1]:.18f}')
    print()
