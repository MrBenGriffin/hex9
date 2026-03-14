# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - uses H9 addresses and round-trips them.
This is a processor-intensive analysis of all edge-cases, measuring behaviour
on each seam, and alongside it. These are then depicted as graphs and summarised
in console messages. Seams are important edge cases (literally!)
This is a bit of a dogs dinner, thanks to AI assistance.
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES.
These values represent SUB-MILLIMETRE projection fidelity checks.
"""
import matplotlib.pyplot as plt
import numpy as np
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.h9 import H9O
from hhg9.h9.region import xy_regions
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from geographiclib.geodesic import Geodesic


WGS84 = Geodesic.WGS84  # or your custom ellipsoid

# --- diagnostics config -----------------------------------------------------
# Set RAW_MODE=True to remove most training wheels (minimal filtering).
RAW_MODE = True
# If not None, values above this nm threshold are treated as alias spikes.
# Set to None in RAW_MODE to disable alias clipping entirely.
ALIAS_CAP_NM = None  # e.g., 10 for 1 m; None disables
# Number of endpoints to elide from each end when plotting/fit (0 keeps all)
ELIDE_VERTICES = 0 if RAW_MODE else 1
# ---------------------------------------------------------------------------
# Given: arrays t (hex_layer,), nm_L (hex_layer,), nm_R (hex_layer,), nb_nm (hex_layer,), beta (hex_layer,), lat0 (hex_layer,)


def _fit_slope(x, y):
    X = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef[1], r2  # slope, R^2


def plot_edge(edge_name, t, nm_L, nm_R, nb_nm, beta, lat0, n_bad=0):
    """
    :param edge_name:
    :param t:
    :param nm_L:
    :param nm_R:
    :param nb_nm:
    :param beta:
    :param lat0:
    :param n_bad:
    :return:
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # 1) Round-trip vs t
    ax = axes[0]
    ax.plot(t, nm_L, label='left (nm)')
    ax.plot(t, nm_R, label='right (nm)', linestyle='--')
    k = 2  # trim
    slL, r2L = _fit_slope(t[k:-k], nm_L[k:-k])
    slR, r2R = _fit_slope(t[k:-k], nm_R[k:-k])
    mlL, mlR = slL * 1e-12, slR * 1e-12
    ax.set_title(f'{edge_name}: round-trip vs t  | slope L={mlL:.3f}km, R={mlR:.3f}km')
    ax.set_xlabel('t (vertex→vertex)')
    ax.set_ylabel('nm')
    ax.legend()
    ax.grid(True)

    # 2) Neighbour drift vs t
    ax = axes[1]
    ax.plot(t, nb_nm)
    slN, r2N = _fit_slope(t, nb_nm)
    ax.set_title(f'neighbour drift vs t | slope={slN:.3f} nm')
    ax.set_xlabel('t')
    ax.set_ylabel('nm')
    ax.grid(True)

    # 3) Signed cross-edge bias (nm), mean near 0
    ax = axes[2]
    ax.plot(t, beta)
    slB, r2B = _fit_slope(t, beta)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_title(f'signed cross-edge bias β vs t | slope={slB:.3e} (nm)')
    ax.set_xlabel('t')
    ax.set_ylabel('β')
    ax.grid(True)

    fig.suptitle(f'Seam diagnostics: {edge_name}  (filtered outliers: {n_bad})', fontsize=12)
    return fig


def elide_vertices(*arrs, drop=0):
    """Return copies of arrays with `drop` points removed from each end.
    If drop == 0, returns the inputs unchanged. Returns a tuple, not a generator.
    """
    if drop == 0:
        return arrs
    out = []
    for a in arrs:
        out.append(a[drop:-drop])
    return tuple(out)


def seam_litmus(b_on_seq, mode, depth=12, delta=1e-12, engine=None):
    """b_on_seq: (hex_layer,2) bary on-seam; engine: H9Engine"""
    xy = np.asarray(b_on_seq)
    # tangent & normal in b_oct
    tvec = np.gradient(xy, axis=0)
    tnorm = np.linalg.norm(tvec, axis=1, keepdims=True)
    tnorm[tnorm == 0] = 1.0
    that = tvec / tnorm
    nhat = np.stack([-that[:, 1], that[:, 0]], axis=1)

    bL = xy + delta * nhat
    bR = xy - delta * nhat

    regsL = xy_regions(bL, mode, depth=depth)
    regsR = xy_regions(bR, mode, depth=depth)

    # leaf region id (last filled col); if your layout is [root, ..., leaf]
    leafL = regsL[:, -1]
    leafR = regsR[:, -1]
    mism = leafL == leafR  # they should differ across the boundary!
    return leafL, leafR, mism


def analyse_edge(edge_name, gcd_st, bry_st, rtp_st):
    """
    gcd_st comes from stitch_seam and may include:
      's' (meters), 'azi', 'left_az', 'right_az'
    """
    m_to_nm = 1e+9  # meters -> nanometres

    # --- parameterization along edge ---
    if 's' in gcd_st:
        s = np.asarray(gcd_st['s'], dtype=float)
        # avoid div-by-zero if degenerate edge
        denom = (s[-1] - s[0]) if s.size > 1 else 1.0
        t = (s - s[0]) / denom
    else:
        # fallback: evenly spaced
        n = gcd_st['on'].shape[0]
        t = np.linspace(0.0, 1.0, n)

    # --- round-trip distances (each flank) ---
    d_L_m = wgs84(gcd_st['L'], rtp_st['L'])
    d_R_m = wgs84(gcd_st['R'], rtp_st['R'])
    nm_L = d_L_m * m_to_nm
    nm_R = d_R_m * m_to_nm

    # --- neighbor spacing delta (after - before) ---
    nb0_nm = wgs84(gcd_st['L'], gcd_st['R']) * m_to_nm
    nb1_nm = wgs84(rtp_st['L'], rtp_st['R']) * m_to_nm

    # --- distance-based alignment of round-trips (robust on EA) ---
    # Distances from each flank to each round-tripped candidate
    dLL_m = wgs84(gcd_st['L'], rtp_st['L'])  # L -> Lrt
    dLR_m = wgs84(gcd_st['L'], rtp_st['R'])  # L -> Rrt
    dRR_m = wgs84(gcd_st['R'], rtp_st['R'])  # R -> Rrt
    dRL_m = wgs84(gcd_st['R'], rtp_st['L'])  # R -> Lrt

    # Choose the closer round-trip for each flank
    use_Lrt_for_L = dLL_m <= dLR_m
    use_Rrt_for_R = dRR_m <= dRL_m

    rtpL_aligned = np.where(use_Lrt_for_L[:, None], rtp_st['L'], rtp_st['R'])
    rtpR_aligned = np.where(use_Rrt_for_R[:, None], rtp_st['R'], rtp_st['L'])

    # --- geodesic cross-track beta (meters -> nm) ---
    azi_on = np.asarray(gcd_st['azi'], dtype=float)
    # azimuths from on-edge -> each aligned round-trip
    invL = [WGS84.Inverse(lat0, lon0, lat1, lon1)
            for (lat0, lon0), (lat1, lon1) in zip(gcd_st['on'], rtpL_aligned)]
    invR = [WGS84.Inverse(lat0, lon0, lat1, lon1)
            for (lat0, lon0), (lat1, lon1) in zip(gcd_st['on'], rtpR_aligned)]
    azi_on_to_Lrt = np.array([inv['azi1'] for inv in invL], dtype=float)
    azi_on_to_Rrt = np.array([inv['azi1'] for inv in invR], dtype=float)

    # signed cross-track components (meters)
    dXT_L_m = d_L_m * np.sin(np.radians(azi_on_to_Lrt - azi_on))
    dXT_R_m = d_R_m * np.sin(np.radians(azi_on_to_Rrt - azi_on))

    # β in nanometres
    beta_nm = 0.5 * (dXT_L_m - dXT_R_m) * 1e9
    # use beta_nm for plotting/summary instead of the b_oct-derived beta
    beta = beta_nm  # replace the old beta array

    # Final distances (meters -> nm)
    d_L_m = np.where(use_Lrt_for_L, dLL_m, dLR_m)
    d_R_m = np.where(use_Rrt_for_R, dRR_m, dRL_m)
    nm_L  = d_L_m * m_to_nm
    nm_R  = d_R_m * m_to_nm

    # (Optional) instrumentation — how often did we “swap sides”?
    side_swaps = int((~use_Lrt_for_L).sum() + (~use_Rrt_for_R).sum())
    print(f"[{edge_name}] distance-based swaps: {side_swaps}")

    nb_nm = nb1_nm - nb0_nm

    # --- build individual masks ---
    is_nanL = ~np.isfinite(nm_L)
    is_nanR = ~np.isfinite(nm_R)

    # seam alias (absurd spikes) threshold in nm
    if ALIAS_CAP_NM is None:
        aliasL = np.zeros_like(t, dtype=bool)
        aliasR = np.zeros_like(t, dtype=bool)
    else:
        aliasL = nm_L > ALIAS_CAP_NM
        aliasR = nm_R > ALIAS_CAP_NM

    # vertex/end trimming
    elide = ELIDE_VERTICES
    at_ends = np.zeros_like(t, dtype=bool)
    at_ends[:elide] = True
    at_ends[-elide:] = True

    # combine (anything that trips on either flank is dropped)
    # combine drop mask: in RAW_MODE, only NaNs and (optional) ends are dropped; else also aliases
    if RAW_MODE:
        drop = is_nanL | is_nanR
    else:
        drop = is_nanL | is_nanR | aliasL | aliasR | at_ends

    n_bad = int(np.count_nonzero(drop))
    keep = ~drop

    # convenience for polar/equator patterns
    lat0 = np.asarray(gcd_st['on'])[:, 0]

    # --- diagnostics per edge ---
    n_tot = t.size
    print({
        'edge': edge_name,
        'tot': int(n_tot),
        'drop_counts': {
            'nanL': int(is_nanL.sum()),
            'nanR': int(is_nanR.sum()),
            'aliasL': int(aliasL.sum()),
            'aliasR': int(aliasR.sum()),
            'ends': int(at_ends.sum()),
        },
        'kept': int((~drop).sum())
    })

    # --- graceful fallback ---
    if keep.sum() == 0:
        # relax: keep all finite, non-endpoint points
        keep = np.isfinite(nm_L) & np.isfinite(nm_R) & ~at_ends
        if keep.sum() == 0:
            # last resort: keep all finite (so we can still show a plot)
            keep = np.isfinite(nm_L) & np.isfinite(nm_R)
    # (re)compute cleaned views after any fallback adjustments
    t_clean     = t[keep]
    nmL_clean   = nm_L[keep]
    nmR_clean   = nm_R[keep]
    nb_clean    = nb_nm[keep]
    beta_clean  = beta[keep]
    lat0_clean  = lat0[keep]

    # use *_core for slopes & summary; use full arrays for the lines if you want
    # slL, r2L = _fit_slope(t_clean, nmL_clean) if t_clean.size else (0.0, 0.0)
    # slR, r2R = _fit_slope(t_clean, nmR_clean) if t_clean.size else (0.0, 0.0)
    # slN, r2N = _fit_slope(t_clean, nb_clean)   if t_clean.size else (0.0, 0.0)
    slB, r2B = _fit_slope(t_clean, beta_clean) if t_clean.size else (0.0, 0.0)

    def _summary(x):
        return dict(median=float(np.median(x)) if x.size else 0.0,
                    p95=float(np.percentile(x, 95)) if x.size else 0.0,
                    mx=float(np.max(x)) if x.size else 0.0,
                    frac_over_1nm=float(np.mean(x > 1.0)) if x.size else 0.0)

    summary = {
        'edge': edge_name,
        'filtered': n_bad,
        'kept': int(np.count_nonzero(keep)),
        'L': _summary(nmL_clean),
        'R': _summary(nmR_clean),
        'drift': _summary(nb_clean),
        'beta': {'slope': float(slB), 'median': float(np.median(beta_clean)) if beta_clean.size else 0.0}
    }
    print(summary)

    # Convenience: on-edge latitude (for pole/equator patterns)
    # lat0 = np.asarray(gcd_st['on'])[:, 0]

    # include the seam metadata in the returned dict if present
    out = dict(
        t=t,
        nm_L=nm_L, nm_R=nm_R,
        nb_nm=nb_nm, nb0_nm=nb0_nm, nb1_nm=nb1_nm,
        beta=beta,
        lat0=lat0,
        t_clean=t_clean,
        nmL_clean=nmL_clean, nmR_clean=nmR_clean,
        nb_clean=nb_clean, beta_clean=beta_clean,
        lat0_clean=lat0_clean,
        filtered=n_bad, kept=int(np.count_nonzero(keep)),
    )
    if 's' in gcd_st: out['s'] = s
    if 'azi' in gcd_st: out['azi'] = np.asarray(gcd_st['azi'])
    if 'left_az' in gcd_st: out['left_az'] = np.asarray(gcd_st['left_az'])
    if 'right_az' in gcd_st: out['right_az'] = np.asarray(gcd_st['right_az'])
    return out


def vert_ll(label, partner_lon=None):
    """Given a vertex return it's latitude and longitude."""
    if label == 'A': return (0.0, 0.0)
    if label == 'P': return (0.0, 180.0)  # Let GeographicLib normalise
    if label == 'E': return (0.0, 90.0)
    if label == 'W': return (0.0, -90.0)
    if label == 'N': return (90.0, float(partner_lon))
    if label == 'S': return (-90.0, float(partner_lon))
    raise ValueError(f"Unknown vertex {label}")


def edge_endpoints_ll(edge_name):
    """edge_name is canonical alphabetical like 'AE','EN','PS', etc."""
    a, b = edge_name[0], edge_name[1]
    # If a pole is present, use the other vertex's lon for the pole
    if a in 'NS' and b in 'AEWP':
        _, lon = vert_ll(b)
        return vert_ll(a, lon), vert_ll(b)
    if b in 'NS' and a in 'AEWP':
        _, lon = vert_ll(a)
        return vert_ll(a), vert_ll(b, lon)
    # No pole on this edge
    return vert_ll(a), vert_ll(b)


def stitch_seam(edge_name, step_km=1.0, offset_m=0.1, ellipsoid=WGS84):
    """Given an edge name, e.g. NE, return seam stitches along it.
    Returns a dict with:
      - 'edge': edge name
      - 'endpoints': ((lat1,lon1), (lat2,lon2))
      - 's': 1D array of along-edge distances (meters) corresponding to each point
      - 'azi': forward azimuths at each on-edge point (degrees)
      - 'left_az' / 'right_az': left/right normal bearings (degrees)
      - 'on', 'L', 'R': (hex_layer,2) arrays of lat/lon for the on-edge points and the
                         left/right offsets (offset_m) at each sample
    """
    (lat1, lon1), (lat2, lon2) = edge_endpoints_ll(edge_name)
    line = ellipsoid.InverseLine(lat1, lon1, lat2, lon2)
    total_m = line.s13
    n = max(1, int(np.ceil(total_m / (step_km * 1000.0))))
    s_vals = np.linspace(0.0, total_m, n + 1)

    on_edge = []
    left_off = []
    right_off = []
    azi_list = []
    laz_list = []
    raz_list = []

    for s in s_vals:
        pos = line.Position(s, Geodesic.LATITUDE | Geodesic.LONGITUDE | Geodesic.AZIMUTH)
        lat, lon, azi = pos['lat2'], pos['lon2'], pos['azi2']  # forward azimuth toward endpoint 2

        # Left/right normals (GeographicLib uses degrees)
        left_az = (azi - 90.0) % 360.0
        right_az = (azi + 90.0) % 360.0

        # Offset to each side by offset_m
        left = WGS84.Direct(lat, lon, left_az, offset_m)
        right = WGS84.Direct(lat, lon, right_az, offset_m)

        on_edge.append((lat, lon))
        left_off.append((left['lat2'], left['lon2']))
        right_off.append((right['lat2'], right['lon2']))
        azi_list.append(azi)
        laz_list.append(left_az)
        raz_list.append(right_az)

    return {
        'edge': edge_name,
        'endpoints': ((lat1, lon1), (lat2, lon2)),
        's': np.asarray(s_vals, dtype=float),
        'azi': np.asarray(azi_list, dtype=float),
        'left_az': np.asarray(laz_list, dtype=float),
        'right_az': np.asarray(raz_list, dtype=float),
        'on': np.array(on_edge),
        'L': np.array(left_off),
        'R': np.array(right_off),
    }


def run():
    """
        Convert
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    b_oct = reg.domain('b_oct')

    h9f = OctahedralH9(reg)  # formatter.
    g_gcd.register_format(DMS(reg))
    g_gcd.register_format(DecimalDegrees(reg))
    c_ell.register_format(DecimalCartesian(reg))
    b_oct.register_format(h9f)

    edge_list = H9O.edges_by_id.flatten()

    # Projections/Transforms. Bary and Net are loaded by the domains.
    # edge_list = [k for k in c_oct.edges]
    gcd_stitches = {''.join(k): stitch_seam(k, 1.0, 0.1) for k in edge_list}
    globe = []
    lengths = []
    for ed in gcd_stitches.values():
        for line in ['on', 'L', 'R']:
            globe.append(ed[line])
            lengths.append(ed[line].shape[0])
    split_idx = np.cumsum(lengths)[:-1]
    pts = Points(np.vstack(globe), g_gcd)
    bry = reg.project(pts, [g_gcd, b_oct])

    bary = np.split(bry.coords, split_idx)
    rtp = reg.project(bry, [b_oct, g_gcd])
    r_gcd = np.split(rtp.coords, split_idx)
    bary_stitches = {}
    rtp_stitches = {}
    idx = 0  # walks through the split arrays in groups of 3: on, L, R
    for k, ed in gcd_stitches.items():
        # each edge contributes 3 sequences in this order: 'on', 'L', 'R'
        bary_stitches[k] = {
            'edge': ed['edge'],
            's': ed['s'],
            'azi': ed['azi'],
            'left_az': ed['left_az'],
            'right_az': ed['right_az'],
            'on': bary[idx + 0],
            'L': bary[idx + 1],
            'R': bary[idx + 2],
        }
        rtp_stitches[k] = {
            'edge': ed['edge'],
            's': ed['s'],
            'azi': ed['azi'],
            'left_az': ed['left_az'],
            'right_az': ed['right_az'],
            'on': r_gcd[idx + 0],
            'L': r_gcd[idx + 1],
            'R': r_gcd[idx + 2],
        }
        idx += 3

    for k in gcd_stitches:
        gcd_s, bry_s, rtp_s = gcd_stitches[k], bary_stitches[k], rtp_stitches[k]
        diag = analyse_edge(k, gcd_s, bry_s, rtp_s)
        fig = plot_edge(k,
                        diag['t_clean'], diag['nmL_clean'], diag['nmR_clean'],
                        diag['nb_clean'], diag['beta_clean'], diag['lat0_clean'],
                        n_bad=diag['filtered'])
        fig.savefig(f"output/ex0059_{k}.png", dpi=100)
        plt.close(fig)
        print(f'fig saved at output/ex0059_{k}.png')


if __name__ == '__main__':
    run()
