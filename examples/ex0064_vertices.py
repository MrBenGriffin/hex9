# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - uses H9 addresses and round-trips them.
This is processor intensive, and works on 100_000 (Seeded) random
points at each vertex, round-trips them, and then draws a heatmap
demonstrating variation across the entire vertex.
Vertices are, like seams, special edge cases.
Looks at radii of 3,000km, 10,000km, and 5m.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
02 Mar 2026 0.1.1a1 (passed, fixed measurement issue.)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import matplotlib.pyplot as plt
import numpy as np
import re


from hhg9 import Registrar, Points
from hhg9.algorithms.geometry import ellipsoid_f_grad, ortho_basis_from_normal
from hhg9.algorithms.pickers import geodesic_cap_rnd_ecef
from hhg9.projections import AKOctahedralEllipsoid
from geographiclib.geodesic import Geodesic
import cartopy.crs as ccrs
from scipy.stats import gaussian_kde, binned_statistic_2d


WGS84 = Geodesic.WGS84  # or custom ellipsoid


def geodesic_distance_nm(lat1, lon1, lat2, lon2):
    """Ellipsoidal geodesic distance on WGS84, returned in nanometres."""
    lat1 = np.asarray(lat1, dtype=float).ravel()
    lon1 = np.asarray(lon1, dtype=float).ravel()
    lat2 = np.asarray(lat2, dtype=float).ravel()
    lon2 = np.asarray(lon2, dtype=float).ravel()
    if not (lat1.size == lon1.size == lat2.size == lon2.size):
        raise ValueError('lat/lon arrays must have the same size')

    s_m = np.fromiter(
        (WGS84.Inverse(a, b, c, d)['s12'] for a, b, c, d in zip(lat1, lon1, lat2, lon2)),
        dtype=float,
        count=lat1.size,
    )
    return (s_m * 1e9).reshape((-1,))


def slug(s: str) -> str:
    """Quick conversion of string to file_slug."""
    s = s.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_-]+", "_", s)    # optional: keep only safe chars
    s = re.sub(r"_+", "_", s).strip("_")   # optional: collapse underscores
    return s


def _compute_nw_grid(u_coords, v_coords, values, *,
                     grid_resolution=200j, bw_method=None,
                     clip_percentile=99.0, den_floor_rel=1e-3,
                     mask_edge=True, boundary_correction='reflect',
                     reflect_margin_frac=0.075, taper_start_frac=None):
    """
    Core Nadaraya–Watson computation returning full diagnostics.
    If `taper_start_frac` is None, no feathering is applied (masked areas are set to NaN only).
    Returns: dict with keys 'grid_u','grid_v','nw','num','den','mask','r_lim','ess_rel','binned_mean','binned_count'.
    """
    # Radius estimate
    r_samples = np.hypot(u_coords, v_coords)
    r_lim = np.percentile(r_samples, clip_percentile) * 1.02

    # Optional reflection near boundary
    use_reflect = (boundary_correction == 'reflect')
    if use_reflect:
        r = r_samples
        theta = np.arctan2(v_coords, u_coords)
        margin = reflect_margin_frac * r_lim
        near_edge = r >= (r_lim - margin)
        if np.any(near_edge):
            r_ref = 2.0 * r_lim - r[near_edge]
            u_ref = r_ref * np.cos(theta[near_edge])
            v_ref = r_ref * np.sin(theta[near_edge])
            u_aug = np.concatenate([u_coords, u_ref])
            v_aug = np.concatenate([v_coords, v_ref])
            vals_aug = np.concatenate([values, values[near_edge]])
        else:
            u_aug, v_aug, vals_aug = u_coords, v_coords, values
    else:
        u_aug, v_aug, vals_aug = u_coords, v_coords, values

    dataset = np.vstack([u_aug, v_aug])

    # Grid
    padding = np.max(np.abs(np.concatenate([u_coords, v_coords]))) * 0.1
    umin, umax = u_coords.min() - padding, u_coords.max() + padding
    vmin, vmax = v_coords.min() - padding, v_coords.max() + padding
    grid_u, grid_v = np.mgrid[umin:umax:grid_resolution, vmin:vmax:grid_resolution]
    grid_pts = np.vstack([grid_u.ravel(), grid_v.ravel()])

    # KDEs
    kde_num = gaussian_kde(dataset, weights=vals_aug, bw_method=bw_method)
    kde_den = gaussian_kde(dataset, weights=np.ones_like(vals_aug), bw_method=bw_method)

    num_raw = kde_num(grid_pts)
    den_raw = kde_den(grid_pts)

    n = vals_aug.shape[0]
    v_sum = np.sum(vals_aug, dtype=float)
    num = num_raw * v_sum
    den = den_raw * n

    den_floor = den_floor_rel * np.nanmax(den)
    valid = den > den_floor

    out = np.full_like(num, np.nan)
    np.divide(num, den, out=out, where=valid)
    nw = out.reshape(grid_u.shape)

    # Edge mask and feather
    if mask_edge:
        r_grid = np.sqrt(grid_u**2 + grid_v**2)
        edge_mask = r_grid <= r_lim
        # Optional feathering: only if explicitly requested
        if taper_start_frac is not None:
            r0 = float(taper_start_frac) * r_lim
            if r0 < r_lim:
                band = (r_grid >= r0) & (r_grid <= r_lim)
                t = (r_grid[band] - r0) / max(1e-12, (r_lim - r0))
                feather = 0.5 * (1.0 + np.cos(np.pi * t))
                nw = nw.copy()
                nw[band] = nw[band] * feather
        # Finally, apply the hard mask to NaN the outside
        nw = np.where(edge_mask, nw, np.nan)
        mask = edge_mask

    # Effective sample size proxy (relative)
    den_grid = den.reshape(grid_u.shape)
    ess_rel = den_grid / (np.nanmax(den_grid) if np.isfinite(den_grid).any() else 1.0)

    # Binned mean / count for a robust non-parametric sanity check
    bins = int(np.round(np.sqrt(np.abs(grid_resolution.real * grid_resolution.imag)))) if hasattr(grid_resolution, 'real') else 100
    bins = max(40, min(200, bins))
    H_mean, u_edges, v_edges, _ = binned_statistic_2d(u_coords, v_coords, values, statistic='mean', bins=bins, range=[[umin, umax], [vmin, vmax]])
    H_count, _, _, _ = binned_statistic_2d(u_coords, v_coords, values*0+1, statistic='sum', bins=bins, range=[[umin, umax], [vmin, vmax]])

    return {
        'grid_u': grid_u, 'grid_v': grid_v, 'nw': nw,
        'num': num.reshape(grid_u.shape), 'den': den_grid,
        'mask': mask, 'r_lim': r_lim, 'ess_rel': ess_rel,
        'binned_mean': H_mean, 'binned_count': H_count,
        'u_edges': u_edges, 'v_edges': v_edges
    }


def plot_cartesian_kde(u_coords, v_coords, values, *,
                       grid_resolution=200j, bw_method=None,
                       title="Heatmap", cmap='viridis',
                       clip_percentile=99.0, den_floor_rel=1e-3,
                       mask_edge=True, name='base',
                       boundary_correction='reflect',
                       reflect_margin_frac=0.075,
                       taper_start_frac=None,
                       ):
    """
    Plots a 2D heatmap directly from Cartesian coordinates
    using a Nadaraya–Watson smoothed mean.
    Masked areas are set to NaN so the colorbar reflects only valid data.
    """
    res = _compute_nw_grid(
        u_coords, v_coords, values,
        grid_resolution=grid_resolution,
        bw_method=bw_method,
        clip_percentile=clip_percentile,
        den_floor_rel=den_floor_rel,
        mask_edge=mask_edge,
        boundary_correction=boundary_correction,
        reflect_margin_frac=reflect_margin_frac,
        taper_start_frac=taper_start_frac,
    )

    grid_u, grid_v, grid_z = res['grid_u'], res['grid_v'], res['nw']

    fig, ax = plt.subplots(figsize=(12, 12))
    pc = ax.contourf(grid_u, grid_v, grid_z, levels=150, cmap=cmap)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label="Estimated mean ∂ (nm) — Nadaraya–Watson")
    ax.set_title(title)
    ax.set_xlabel("Tangent U-axis (m)")
    ax.set_ylabel("Tangent V-axis (m)")
    f_name = slug(name)
    plt.savefig(f"output/ex0064_{f_name}_kde.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/ex0064_{f_name}_kde.png')


def plot_cartesian_kde_diagnostics(u_coords, v_coords, values, *,
                                   grid_resolution=200j, bw_method=None,
                                   clip_percentile=99.0, den_floor_rel=1e-3,
                                   mask_edge=True, name='base',
                                   boundary_correction='reflect',
                                   reflect_margin_frac=0.075,
                                   taper_start_frac=None,
                                   ):
    """Four-panel diagnostics: NW mean, denominator (ESS proxy), numerator, and binned mean; plus a radial edge profile."""
    res = _compute_nw_grid(
        u_coords, v_coords, values,
        grid_resolution=grid_resolution,
        bw_method=bw_method,
        clip_percentile=clip_percentile,
        den_floor_rel=den_floor_rel,
        mask_edge=mask_edge,
        boundary_correction=boundary_correction,
        reflect_margin_frac=reflect_margin_frac,
        taper_start_frac=taper_start_frac,
    )
    gu, gv = res['grid_u'], res['grid_v']
    nw, num, den, ess = res['nw'], res['num'], res['den'], res['ess_rel']
    Hm = res['binned_mean']
    ue, ve = res['u_edges'], res['v_edges']
    f_name = slug(name)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    im0 = axs[0,0].contourf(gu, gv, nw, 150)
    axs[0,0].set_title('NW smoothed mean (nm)')
    axs[0,0].set_aspect('equal'); fig.colorbar(im0, ax=axs[0,0])

    im1 = axs[0,1].contourf(gu, gv, den, 150)
    axs[0,1].set_title('Denominator (scaled density)')
    axs[0,1].set_aspect('equal'); fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[1,0].contourf(gu, gv, num, 150)
    axs[1,0].set_title('Numerator (weighted density)')
    axs[1,0].set_aspect('equal'); fig.colorbar(im2, ax=axs[1,0])

    extent = [ue[0], ue[-1], ve[0], ve[-1]]
    im3 = axs[1,1].imshow(Hm.T, origin='lower', extent=extent, aspect='equal')
    axs[1,1].set_title('Binned mean (nm)')
    fig.colorbar(im3, ax=axs[1,1])

    # Radial edge profile of NW mean in the outer 5% ring
    r_lim = res['r_lim']
    r_grid = np.sqrt(gu**2 + gv**2)
    band = (r_grid >= 0.95*r_lim) & (r_grid <= r_lim)
    theta = np.arctan2(gv, gu)
    theta_band = theta[band]; nw_band = nw[band]
    # Aggregate by angle
    nbins = 72
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    idx = np.digitize(theta_band, bins) - 1
    prof = np.full(nbins, np.nan)
    for k in range(nbins):
        vals = nw_band[idx==k]
        if vals.size:
            prof[k] = np.nanmedian(vals)
    fig2, ax2 = plt.subplots(figsize=(10,3))
    centers = 0.5*(bins[:-1]+bins[1:])
    ax2.plot(np.degrees(centers), prof, marker='.')
    ax2.set_xlabel('Angle (deg)'); ax2.set_ylabel('NW median (outer ring, nm)')
    ax2.set_title('Edge ring angular profile — check for "glow" directions')
    ax2.grid(True, alpha=0.3)
    plt.savefig(f"output/ex0064_{f_name}_kde_diag.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/ex0064_{f_name}_kde_diag.png')


def plot_global_error_map(lons, lats, errors_nm, center_lon, center_lat, title, name, *,
                          c_clip=(0.5, 99.5),
                          force_vmin=None,
                          force_vmax=None,
                          ):
    """
    Plots error data on a globe using an Orthographic projection.
    Good for large radii.
    """
    fig = plt.figure(figsize=(12, 12), dpi=300, frameon=False)

    # Create a map projection centered on our test point
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=center_lon, central_latitude=center_lat)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    img = plt.imread('src/bm_3600x1800.jpg')
    ax.imshow(img, origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
    ax.coastlines(resolution='50m', color='white', alpha=0.25)
    ax.set_global()  # Zoom out to see the whole globe

    errors_nm = np.asarray(errors_nm, dtype=float)
    finite = np.isfinite(errors_nm)
    if not np.any(finite):
        raise ValueError('errors_nm contains no finite values')

    if force_vmin is None or force_vmax is None:
        lo_p, hi_p = c_clip
        vmin = np.percentile(errors_nm[finite], lo_p) if force_vmin is None else float(force_vmin)
        vmax = np.percentile(errors_nm[finite], hi_p) if force_vmax is None else float(force_vmax)
    else:
        vmin, vmax = float(force_vmin), float(force_vmax)

    # Guard against degenerate ranges
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = np.nanmin(errors_nm[finite])
        vmax = np.nanmax(errors_nm[finite])

    scatter = ax.scatter(
        lons, lats,
        c=errors_nm,
        cmap='viridis',
        ec='none', s=5,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax,
    )
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
    cbar.set_label(f'Round-trip Error (nm)  [color scale {vmin:.3g}–{vmax:.3g} nm]')

    # Helpful console stats for sanity
    mn = float(np.nanmin(errors_nm[finite]))
    mx = float(np.nanmax(errors_nm[finite]))
    p50 = float(np.percentile(errors_nm[finite], 50))
    p99 = float(np.percentile(errors_nm[finite], 99))
    print(f'errors_nm stats: min={mn:.6g}  p50={p50:.6g}  p99={p99:.6g}  max={mx:.6g}')

    ax.set_title(f'LAEA: {title}')
    f_name = slug(name)
    plt.savefig(f"output/ex0064_{f_name}_laea.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/ex0064_{f_name}_laea.png')


def run():
    """
        Runner for vertex error graphs.
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')   # GCD Spherical Domain (latitude/longitude)
    c_ell = reg.domain('c_ell')   # Cartesian Ellipsoid (xyz) ECEF
    c_oct = reg.domain('c_oct')   # Cartesian Octahedron (xyz)
    b_oct = reg.domain('b_oct')   # Octahedron Barycentric-origin xy

    # Projections/Transforms. Bary and Net are loaded by the domains.
    # EllipsoidGCD(reg)  # g_sph <=> c_sph
    # ak = AKOctahedralEllipsoid(reg)  # c_sph <=> (c_oct <=> b_oct)
    # ak.set_accuracy(1e-9)  # 1nm area
    # GCDBary(reg)

    vertices = {
        # "Russia": np.array([35.264, 45.0], dtype=float),
        "Null Antipodes": np.array([0.0, 180.0], dtype=float),
        "Null Island": np.array([0.0, 0.0], dtype=float),
        "East Equator": np.array([0.0, 90.0], dtype=float),
        "West Equator": np.array([0.0, -90.0], dtype=float),
        "North Pole": np.array([90.0, 0.0], dtype=float),
        "South Pole": np.array([-90.0, 0.0], dtype=float),
    }

    # elliptical axes
    a = c_ell.a
    b = c_ell.b
    print(f"c_ell axes: a={a:.3f}  b={b:.3f};  WGS84 a={WGS84.a:.3f}  f={WGS84.f:.12g}")
    # 3_000_000 =  local
    # 10_000_000 = hemisphere.
    # 20_000_000 = global. , 3_000_000, 10_000_000.0, 5.0

    for radius in [20_000_000, 0.1, 5.0]:
        for name, (lat, lon) in vertices.items():
            centre = Points(np.array([[lat, lon]]), g_gcd)
            ll_pts = geodesic_cap_rnd_ecef(reg, lat, lon, 100_000, radius, seed=4325325)
            ref = reg.project(ll_pts, [g_gcd, c_ell])

            # Round-trip back to lat/lon (so we can inspect the worst offenders)
            pll_ll = reg.project(ll_pts, [g_gcd, b_oct, g_gcd]).coords

            # And round-trip back to ECEF for a robust 3D distance metric
            pll = Points(pll_ll, g_gcd)
            pll = reg.project(pll, [g_gcd, c_ell])

            d_m = np.linalg.norm(pll.coords - ref.coords, axis=1)
            nm_pt = d_m * 1e9  # nanometre

            # Also compute geodesic distance between original and round-tripped lat/lon (WGS84)
            lat1, lon1 = ll_pts.coords[:, 0], ll_pts.coords[:, 1]
            lat2, lon2 = pll_ll[:, 0], pll_ll[:, 1]
            nm_geod = geodesic_distance_nm(lat1, lon1, lat2, lon2)

            # Sanity: at micron scale, chord and geodesic should agree extremely closely.
            finite = np.isfinite(nm_geod) & np.isfinite(nm_pt)
            if np.any(finite):
                diff = nm_geod[finite] - nm_pt[finite]
                print(
                    f"distance sanity (nm): chord p50={np.percentile(nm_pt[finite], 50):.6g}  "
                    f"geod p50={np.percentile(nm_geod[finite], 50):.6g}  "
                    f"(geod-chord) p50={np.percentile(diff, 50):.6g}  maxabs={np.max(np.abs(diff)):.6g}"
                )

            # Prefer geodesic for reporting/plots to match ex0061
            nm_pt = nm_geod

            # Print the worst round-trips for inspection/debugging
            top_k = 10
            worst = np.argsort(nm_pt)[-top_k:][::-1]
            print(f"\nWorst {top_k} round-trips for {name} (radius={radius} m):")
            for i in worst:
                lat1, lon1 = ll_pts.coords[i, 0], ll_pts.coords[i, 1]
                lat2, lon2 = pll_ll[i, 0], pll_ll[i, 1]
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                print(
                    f"  i={int(i):6d}  err_nm={nm_pt[i]:10.6f}  "
                    f"orig=({lat1: .16f}, {lon1: .16f})  "
                    f"rt=({lat2: .16f}, {lon2: .16f})  "
                    f"d=({dlat:+.3e}°, {dlon:+.3e}°)"
                )

            if radius > 200.0:
                la, lo = ll_pts.coords[:, 0], ll_pts.coords[:, 1]
                sort_indices = np.argsort(nm_pt)
                lo_s = lo[sort_indices]
                la_s = la[sort_indices]
                errs = nm_pt[sort_indices]
                f_name = f'{name}_{radius/1000}km'
                plot_global_error_map(
                    lo_s, la_s, errs, center_lon=lon, center_lat=lat,
                    title=f"c_ell/b_oct Projection ∂nm at '{name}'; over radius {radius}", name=f_name,
                    c_clip=(0.5, 99.5),
                )
            else:
                # do small.
                c_xyz = reg.project(centre, [g_gcd, c_ell]).coords[0]
                _, grad_c = ellipsoid_f_grad(c_xyz, a, b)
                n_hat = grad_c / np.linalg.norm(grad_c)
                u_hat, v_hat = ortho_basis_from_normal(n_hat)
                xyz_relative = ref.coords - c_xyz

                coords_2d_u = xyz_relative @ u_hat
                coords_2d_v = xyz_relative @ v_hat
                # plot_cartesian_kde_diagnostics(coords_2d_u, coords_2d_v, nm_pt)
                f_name = f'{name}_{radius}m'
                plot_cartesian_kde(
                    coords_2d_u,
                    coords_2d_v,
                    nm_pt,
                    title=f"c_ell/b_oct Projection ∂nm at '{name}'; over radius {radius}", name=f_name
                )


if __name__ == '__main__':
    run()
