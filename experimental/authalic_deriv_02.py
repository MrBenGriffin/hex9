# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
    using cell identification of an authalic triangular grid
    on WGS84 with minimal solves per cell.

    seams fixed (meridians / equator edges),
    spine fixed (45° meridian)
    edges are geodesic segments
    per-triangle area target fixed
    symmetry enforced
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, unique
from pathlib import Path
from typing import Tuple
import numpy as np
from geographiclib.geodesic import Geodesic
from geographiclib.polygonarea import PolygonArea
import time


@unique
class AF(IntEnum):
    """
    Enumeration of various Point-Solving Algorithms.
    """
    FIXED = 1  # pre-defined octant vertex (3)
    SEAMM = 2  # 1D-mirrored seam; (2)
    SEAMG = 3  # 1D-general seam; 3*dim_size - 7
    SPINE = 4  # 1D-spine; dim_size//2 - 1
    TWINS = 5  # 2D-mirrored spine; dim_size//2 - 1
    INNER = 6  # 2D-general; All the rest.
    ALIAS = 7  # Mirror-solves
    QTWIN = 8  # Twin Solution for TWIN on equator
    SEAMQ = 9  # 1D-equator seam; (equator points excluding fixed / qtwin)


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


class AreaOracle:
    def __init__(self, geod: Geodesic | None = None):
        self.geod = geod or Geodesic.WGS84

    def tri_area(self, ta: LatLon, tb: LatLon, tc: LatLon) -> float:
        poly = PolygonArea(self.geod, False)
        poly.AddPoint(ta.lat, ta.lon)
        poly.AddPoint(tb.lat, tb.lon)
        poly.AddPoint(tc.lat, tc.lon)
        _num, _perim, area = poly.Compute(False, True)
        return abs(area)


def point_on_geodesic_from_to(geod: Geodesic, a: LatLon, b: LatLon, s_from_a_m: float) -> LatLon:
    """Return the latlon of a point between a,b in metres."""
    inv = geod.Inverse(a.lat, a.lon, b.lat, b.lon)
    line = geod.Line(a.lat, a.lon, inv["azi1"])
    pos = line.Position(s_from_a_m)
    return LatLon(pos["lat2"], pos["lon2"])


def meridian_point_from_pole(geod: Geodesic, lon: float, s_from_pole_m: float) -> LatLon:
    """Given a meridian, find the line"""
    pole = LatLon(90.0, lon)
    line = geod.Line(pole.lat, pole.lon, 180.0)  # due south
    pos = line.Position(s_from_pole_m)
    return LatLon(pos["lat2"], pos["lon2"])


def _mirror_lon(lon: float, *, lon_mirror: float = 90.0) -> float:
    """Mirror longitude across the spine meridian halfway across the octant.

    For an octant bounded by lon=0 and lon=90, mirroring across the spine lon=45
    maps lon -> 90 - lon.
    """
    return lon_mirror - lon


def solve_latlon_2d(
        *,
        residuals_fn,
        lat_bounds: Tuple[float, float] = (0.0, 90.0),
        lon_bounds: Tuple[float, float] = (0.0, 90.0),
        lat0: float | None = None,
        lon0: float | None = None,
        max_iter: int = 50,
        f_scale: float = 1.0,
        f_tol_rel: float = 1e-12,
        step_damp: float = 0.5,
        d_lat: float = 1e-7,
        d_lon: float = 1e-7,
) -> LatLon:
    """Generic 2D damped-Newton solver in (lat, lon).

    Parameters
    ----------
    residuals_fn:
        Callable (lat, lon) -> (f1, f2). The solver finds (lat, lon) such that
        f1 == 0 and f2 == 0.
    lat_bounds, lon_bounds:
        Hard clamps applied each iteration. Use tight bounds to pick the intended
        branch when multiple solutions exist.
    lat0, lon0:
        Optional initial guess. Defaults to midpoints of the bounds.
    f_scale:
        Characteristic scale of residuals (e.g. target area in m^2). Used only
        for tolerance scaling.

    Returns
    -------
    LatLon:
        The solved point.
    """

    lat = 0.5 * (lat_bounds[0] + lat_bounds[1]) if lat0 is None else lat0
    lon = 0.5 * (lon_bounds[0] + lon_bounds[1]) if lon0 is None else lon0

    def clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    for _ in range(max_iter):
        f1, f2 = residuals_fn(lat, lon)
        # Guard against NaNs from degenerate triangles or invalid inputs.
        if not np.isfinite(f1) or not np.isfinite(f2):
            # Nudge toward the centre of the search box.
            lat = 0.5 * (lat_bounds[0] + lat_bounds[1])
            lon = 0.5 * (lon_bounds[0] + lon_bounds[1])
            continue
        scale = max(1.0, float(f_scale))
        if max(abs(f1), abs(f2)) <= f_tol_rel * scale:
            return LatLon(lat, lon)

        # Jacobian via central differences
        f1p, f2p = residuals_fn(lat + d_lat, lon)
        f1m, f2m = residuals_fn(lat - d_lat, lon)
        j11 = (f1p - f1m) / (2.0 * d_lat)
        j21 = (f2p - f2m) / (2.0 * d_lat)

        f1p, f2p = residuals_fn(lat, lon + d_lon)
        f1m, f2m = residuals_fn(lat, lon - d_lon)
        j12 = (f1p - f1m) / (2.0 * d_lon)
        j22 = (f2p - f2m) / (2.0 * d_lon)

        det = j11 * j22 - j12 * j21
        if not np.isfinite(det):
            lat = 0.5 * (lat_bounds[0] + lat_bounds[1])
            lon = 0.5 * (lon_bounds[0] + lon_bounds[1])
            continue
        if det == 0.0:
            # Degenerate Jacobian: take a tiny, safe step.
            lat = clamp(lat - step_damp * 1e-5 * (1.0 if f1 > 0 else -1.0), *lat_bounds)
            lon = clamp(lon - step_damp * 1e-5 * (1.0 if f2 > 0 else -1.0), *lon_bounds)
            continue

        # Newton step: J * [dlat, dlon] = -f
        dlat = (-f1 * j22 + f2 * j12) / det
        dlon = (f1 * j21 - f2 * j11) / det

        lat_new = clamp(lat + step_damp * dlat, *lat_bounds)
        lon_new = clamp(lon + step_damp * dlon, *lon_bounds)

        f1n, f2n = residuals_fn(lat_new, lon_new)
        if max(abs(f1n), abs(f2n)) > max(abs(f1), abs(f2)):
            damp = step_damp
            improved = False
            for _k in range(7):
                damp *= 0.5
                lat_try = clamp(lat + damp * dlat, *lat_bounds)
                lon_try = clamp(lon + damp * dlon, *lon_bounds)
                f1t, f2t = residuals_fn(lat_try, lon_try)
                if max(abs(f1t), abs(f2t)) <= max(abs(f1), abs(f2)):
                    lat_new, lon_new = lat_try, lon_try
                    improved = True
                    break
            if not improved:
                # Nudge longitude slightly to escape flat spots.
                lon_new = clamp(lon + 1e-4 * (1.0 if f1 < 0 else -1.0), *lon_bounds)

        lat, lon = lat_new, lon_new

    return LatLon(lat, lon)


class Lattice:
    """Lattice Construct"""
    def __init__(self, layer, oracle: AreaOracle) -> None:
        self.layer = layer
        self.oracle = oracle
        self.oa = None  # octant area
        self.max_iter: int = 96
        self.rel_area_tol: float = 1e-13
        self.abs_s_tol_m: float = 1e-9
        self.step_damp: float = 0.5
        self.tri_num = 9 ** layer
        self.dim_size = 3 ** layer + 1
        self.vert_num = self.dim_size*(self.dim_size+1)//2
        self.ll = np.full([self.vert_num, 2], dtype=np.float64, fill_value=-1.0)
        self.af = np.zeros([self.vert_num], dtype=np.uint8)
        self.par = np.full([self.vert_num, 2],  dtype=int, fill_value=-1)
        self.sib = np.full([self.vert_num],  dtype=int, fill_value=-1)
        self.c_area = np.full((self.vert_num, 2), np.nan)
        self.c_rel = np.full((self.vert_num, 2), np.nan)
        self.t2i = None
        self.i2t = None
        self.valid = None
        self._init_maps()  # i2t, t2i
        self._init_grid()  # ll, ft
        self._init_par_sib()  # par, sib
        self.pole = LatLon(*self.ll[0])
        a = self.oa
        for level in range(layer):
            a /= 9.0  # L1 microtriangle area target
        self.target_area = a
        self.tri_grid = self._tri_grid()
        self.tri_area = np.full((self.tri_grid.shape[0],), np.nan)
        self.tri_rel = np.full((self.tri_grid.shape[0],), np.nan)

    def _init_maps(self) -> None:
        ck_sum = 2 * (self.dim_size - 1)
        self.t2i = np.full(
            (self.dim_size, self.dim_size, self.dim_size),
            fill_value=-1, dtype=np.int32
        )
        self.i2t = np.full(
            (self.vert_num, 3),
            fill_value=-1, dtype=np.int32
        )
        idx = 0
        for p in range(self.dim_size):  # 0..D-1
            l_min = max(0, ck_sum - p - (self.dim_size - 1))
            l_max = min(self.dim_size - 1, ck_sum - p)
            for l in range(l_min, l_max + 1):
                r = ck_sum - (p + l)
                if 0 <= r < self.dim_size:
                    self.t2i[p, l, r] = idx
                    self.i2t[idx] = p, l, r
                    idx += 1
        assert self.vert_num - idx == 0  # should equal D*(D+1)//2

    def ti(self, mask: np.ndarray) -> np.ndarray:
        """Helper: convert a 3D boolean mask (in (tri_points,l,r) space) to 1D vertex indices."""
        return self.t2i[mask]

    def _init_grid(self) -> None:
        tm = self.dim_size - 1
        p = np.arange(self.dim_size)[:, None, None]
        l = np.arange(self.dim_size)[None, :, None]
        r = np.arange(self.dim_size)[None, None, :]
        valid = self.t2i >= 0

        # INNER - set all valid to this first.
        self.af[self.ti(valid)] = AF.INNER

        # SEAMG - left boundary meridian seam only: lon == 0 (r == tm)
        seam_mask = valid & (r == tm)
        self.af[self.ti(seam_mask)] = AF.SEAMG

        # SPINE - down the centre.
        spine_mask = valid & (r == l)
        self.af[self.ti(spine_mask)] = AF.SPINE

        # TWINS - the row immediately adjacent to the spine (one off: r == l-1)
        twins_mask = valid & (r == (l + 1))
        self.af[self.ti(twins_mask)] = AF.TWINS

        # SEAMQ - equator seam (tri_points == tm). This is handled separately from SEAMG.
        seamq_mask = valid & (p == tm)
        self.af[self.ti(seamq_mask)] = AF.SEAMQ

        # QTWIN - set equator twin tri_points == tm
        seam2_mask = valid & (p == tm) & (r == (l + 1))
        self.af[self.ti(seam2_mask)] = AF.QTWIN

        # SEAMM - row 1 - but don't want alias.
        tops_mask = valid & (p == 1)
        self.af[self.ti(tops_mask)] = AF.SEAMM

        # ALIAS - entire right side
        alias_mask = valid & (r < l)
        self.af[self.ti(alias_mask)] = AF.ALIAS

        # FIXED - octant vertices.
        vx = [0, tm, tm], [tm, 0, tm], [tm, tm, 0]  # pole, left, right.
        lx = (90., 0), (0, 0), (0, 90.)             # pole, left, right
        for v, l in zip(vx, lx):
            j = self.t2i[tuple(v)]
            self.ll[j] = l
            self.af[j] = AF.FIXED
        # now set area.
        pole = LatLon(*lx[0])
        lp = LatLon(*lx[1])
        rp = LatLon(*lx[2])
        self.oa = self.oracle.tri_area(pole, lp, rp)

    def _init_par_sib(self) -> None:
        """Populate `par` and `sib` lookup tables in index space.

        Parents are always on the prior row (tri_points-1):
          - parent L: (tri_points-1, l,   r+1)
          - parent R: (tri_points-1, l+1, r)

        Sibling is the elder sibling toward the spine on the same row:
          - sibling:  (tri_points,   l+1, r-1)

        All are stored as 1D vertex indices, or -1 if not valid.
        """
        p = self.i2t[:, 0]
        l = self.i2t[:, 1]
        r = self.i2t[:, 2]

        pp = p - 1
        par_l = np.full(self.vert_num, -1, dtype=np.int32)
        par_r = np.full(self.vert_num, -1, dtype=np.int32)

        m = pp >= 0
        # parent R: (tri_points-1, l+1, r)
        mr = m & ((l + 1) < self.dim_size)
        par_r[mr] = self.t2i[pp[mr], l[mr] + 1, r[mr]]

        # parent L: (tri_points-1, l, r+1)
        ml = m & ((r + 1) < self.dim_size)
        par_l[ml] = self.t2i[pp[ml], l[ml], r[ml] + 1]

        self.par[:, 0] = par_l
        self.par[:, 1] = par_r

        sib = np.full(self.vert_num, -1, dtype=np.int32)
        ms = ((l + 1) < self.dim_size) & ((r - 1) >= 0)
        sib[ms] = self.t2i[p[ms], l[ms] + 1, r[ms] - 1]
        self.sib[:] = sib

    def solve_seamq(self, idx):
        """Solve an equator point X (lat==0) marching from spine -> seam.

        For equator points we must constrain the *down/roof* triangle that uses the
        two row-above parents (p_l, p_r):

            Area(p_l, p_r, X_eq) == target_area

        This directly constrains the family of triangles that otherwise remain
        unconstrained by INNER solves and accumulate error near the equator.

        We solve 1D along the equator segment from a seamward already-solved neighbour (if any, otherwise seam corner)
        to the already-solved spineward sibling (elder sibling, `sb`).
        """
        if self.ll[idx][0] != -1:  # solved already.
            return

        tm = self.dim_size - 1
        p_idx, _l_idx, _r_idx = self.i2t[idx]
        if p_idx != tm:
            raise RuntimeError(f"SEAMQ called for non-equator idx={idx} plr={self.i2t[idx].tolist()}")

        # Octant longitudes.
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = float(self.ll[lo_lt][1])
        lon_r = float(self.ll[lo_rt][1])

        # Dependencies.
        p_l, p_r = self.par_ll(idx)
        sb = self.sib_ll(idx)
        if p_l is None or p_r is None or sb is None:
            raise RuntimeError(f"SEAMQ {idx} missing deps: par={self.par[idx]} sib={self.sib[idx]}")

        # Sibling is the already-solved spineward neighbour on the equator.
        if sb.lat != 0.0:
            sb = LatLon(0.0, float(sb.lon))

        # The constrained triangle is the down/roof triangle using both parents.
        pa_l = p_l
        pa_r = p_r

        # We must march spine -> seam, so the equator point must lie between:
        #   - a seamward already-solved neighbour (if any), otherwise the seam corner
        #   - the spineward already-solved elder sibling `sb`
        # This prevents collapsing multiple SEAMQ points onto the same longitude.

        # seamward neighbour on the equator: (tri_points, l-1, r+1)
        p, l, r = self.i2t[idx]
        seamward = None
        if l - 1 >= 0 and r + 1 < self.dim_size:
            sw_idx = self.t2i[p, l - 1, r + 1]
            if sw_idx >= 0 and self.ll[sw_idx][0] != -1.0:
                seamward = LatLon(*self.ll[sw_idx])

        # Default seamward bound is the seam corner at lon_l.
        if seamward is None:
            seamward = LatLon(0.0, lon_l)
        else:
            seamward = LatLon(0.0, float(seamward.lon))

        spineward = LatLon(0.0, float(sb.lon))

        # Ensure ordering by longitude (seamward.lon <= spineward.lon)
        if seamward.lon > spineward.lon:
            seamward, spineward = spineward, seamward

        seg0 = seamward
        seg1 = spineward

        inv = self.oracle.geod.Inverse(seg0.lat, seg0.lon, seg1.lat, seg1.lon)
        s_total = float(inv["s12"])

        def area_at_s(s: float) -> float:
            x = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, s)
            return self.oracle.tri_area(pa_l, pa_r, x)

        lo = 0.0
        hi = s_total
        a_lo = area_at_s(lo)  # area at seamward bound
        a_hi = area_at_s(hi)  # area at spineward bound

        a_min = min(a_lo, a_hi)
        a_max = max(a_lo, a_hi)
        if not (a_min <= self.target_area <= a_max):
            print(
                f"ERROR: Target area not bracketed on equator SEAMQ (roof tri): max={a_max} vs target={self.target_area} "
                f"for idx={idx} plr={self.i2t[idx].tolist()} p_l=({pa_l.lat},{pa_l.lon}) p_r=({pa_r.lat},{pa_r.lon}) "
                f"seg_lon=[{seg0.lon}->{seg1.lon}]"
            )
            # Hard failure: do not collapse onto an endpoint. Leave unsolved so upstream errors are visible.
            return

        result = None
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = area_at_s(mid)
            if abs(a_mid - self.target_area) <= self.rel_area_tol * max(1.0, self.target_area):
                result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, mid)
                break
            if (hi - lo) <= self.abs_s_tol_m:
                result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, mid)
                break

            # Keep a bracket without assuming monotonic direction.
            if (a_lo <= self.target_area <= a_mid) or (a_mid <= self.target_area <= a_lo):
                hi, a_hi = mid, a_mid
            else:
                lo, a_lo = mid, a_mid

        if result is None:
            result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, 0.5 * (lo + hi))

        # Diagnostics.
        a1 = self.oracle.tri_area(pa_l, pa_r, result)
        self.c_area[idx, 0] = a1
        self.c_rel[idx, 0] = (a1 - self.target_area) / self.target_area

        # Set canonical and alias.
        _p, _l, _r = self.i2t[idx]
        alias = self.t2i[_p, _r, _l]
        self.c_area[alias, 0] = a1
        self.c_rel[alias, 0] = self.c_rel[idx, 0]

        self.ll[idx] = [0.0, result.lon]
        self.ll[alias] = [0.0, _mirror_lon(result.lon, lon_mirror=lon_r)]

    def par_ll(self, idx):
        pl, pr = self.par[idx]
        pll = LatLon(*self.ll[pl]) if pl >= 0 else None
        plr = LatLon(*self.ll[pr]) if pr >= 0 else None
        return pll, plr

    def sib_ll(self, idx):
        sib = self.sib[idx]
        ll = LatLon(*self.ll[sib]) if sib >= 0 else None
        return ll

    def solve_seamm(self, idx):
        """
        Solve PL1 on lon_left meridian and PR1 on lon_right meridian, both at equal meridian distance s
        from the North Pole, such that Area(P,PL1,PR1)=target.
        Returns (pl1, pr1, s_from_pole_m).
        """
        if self.ll[idx][0] != -1:  # solved already.
            return
        tm = self.dim_size - 1
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_left = self.ll[lo_lt][1]
        lon_right = self.ll[lo_rt][1]
        # Row-1 mirrored-seam points may have no explicit parents populated yet.
        # The pole is always fixed at vertex 0.
        pole = self.pole
        target_area_m2 = self.target_area

        def area_at_s(s: float) -> float:
            pl1 = meridian_point_from_pole(self.oracle.geod, lon_left, s)
            pr1 = meridian_point_from_pole(self.oracle.geod, lon_right, s)
            return self.oracle.tri_area(pole, pl1, pr1)

        s0 = max(1e-6, np.sqrt(4.0 * target_area_m2 / np.pi))
        lo = 0.0
        hi = s0
        a_hi = area_at_s(hi)
        expand = 0
        while a_hi < target_area_m2:
            hi *= 2.0
            a_hi = area_at_s(hi)
            expand += 1
            if expand > 80:
                raise RuntimeError("Failed to bracket polar cap area.")
        result = LatLon(0, 0)
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = area_at_s(mid)
            if abs(a_mid - target_area_m2) <= self.rel_area_tol * max(1.0, target_area_m2):
                s = mid
                result = meridian_point_from_pole(self.oracle.geod, lon_left, s)
                break
            if (hi - lo) <= self.abs_s_tol_m:
                s = mid
                result = meridian_point_from_pole(self.oracle.geod, lon_left, s)
                break
            if a_mid >= target_area_m2:
                hi = mid
            else:
                lo = mid
        if result.lat == 0:
            result = meridian_point_from_pole(self.oracle.geod, lon_left, 0.5 * (lo + hi))
        _p, _l, _r = self.i2t[idx]
        alias = self.t2i[_p, _r, _l]  # assumes rt_side = alias
        # Diagnostics: achieved constraint area.
        pl1 = LatLon(result.lat, lon_left)
        pr1 = LatLon(result.lat, lon_right)
        a1 = self.oracle.tri_area(pole, pl1, pr1)
        self.c_area[idx, 0] = a1
        self.c_rel[idx, 0] = (a1 - self.target_area) / self.target_area
        self.c_area[alias, 0] = a1
        self.c_rel[alias, 0] = self.c_rel[idx, 0]
        self.ll[idx] = [result.lat, lon_left]
        self.ll[alias] = [result.lat, lon_right]

    def solve_seamg(self, idx):
        """
        Solve X on the geodesic segment seg0->seg1 such that Area(U,V,X)=target.
        Returns (x, s_from_seg0_m).
        Handles only left boundary seam (r==tm).
        """
        tm = self.dim_size - 1
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = float(self.ll[lo_lt][1])
        lon_r = float(self.ll[lo_rt][1])
        spine_lon = 0.5 * (lon_l + lon_r)

        left_corner = LatLon(0.0, lon_l)
        far_left = left_corner
        far_spine_eq = LatLon(0.0, spine_lon)

        if self.ll[idx][0] != -1:  # solved already.
            return

        p_l, p_r = self.par_ll(idx)
        sb = self.sib_ll(idx)
        if sb is None:
            raise RuntimeError(f"SEAMG {idx} missing sibling: sib={self.sib[idx]}")

        # Left meridian seam: constrain X to the meridian segment from the sole parent down to (0,0).
        if p_l is None and p_r is None:
            raise RuntimeError(f"SEAMG {idx} missing parents: par={self.par[idx]}")
        pa = p_r if p_l is None else p_l
        seg0 = pa
        seg1 = far_left
        inv = self.oracle.geod.Inverse(seg0.lat, seg0.lon, seg1.lat, seg1.lon)
        s_total = float(inv["s12"])

        def area_at_s(s: float) -> float:
            x = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, s)
            return self.oracle.tri_area(pa, sb, x)

        lo = 0.00000
        hi = s_total
        a_lo = area_at_s(lo)
        a_hi = area_at_s(hi)

        # Expect monotone; require target within [min,max]
        a_min = min(a_lo, a_hi)
        a_max = max(a_lo, a_hi)
        if not (a_min <= self.target_area <= a_max):
            print(
                f"ERROR: Target area not bracketed on seam segment: [min={a_min}, max={a_max}] vs target={self.target_area} "
                f"for idx={idx} plr={self.i2t[idx].tolist()} pa=({pa.lat},{pa.lon}) sb=({sb.lat},{sb.lon}) far=({seg1.lat},{seg1.lon})"
            )

        result = None
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = area_at_s(mid)
            if abs(a_mid - self.target_area) <= self.rel_area_tol * max(1.0, self.target_area):
                result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, mid)
                break
            if (hi - lo) <= self.abs_s_tol_m:
                result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, mid)
                break
            # keep bracket without assuming increasing/decreasing
            if (a_lo <= self.target_area <= a_mid) or (a_mid <= self.target_area <= a_lo):
                hi, a_hi = mid, a_mid
            else:
                lo, a_lo = mid, a_mid

        if result is None:
            result = point_on_geodesic_from_to(self.oracle.geod, seg0, seg1, 0.5 * (lo + hi))
        _p, _l, _r = self.i2t[idx]
        alias = self.t2i[_p, _r, _l]  # assumes rt_side = alias
        ap_l, ap_r = self.par_ll(alias)
        ap = ap_r if ap_l is None else ap_l
        # Diagnostics: achieved constraint area.
        a1 = self.oracle.tri_area(pa, sb, result)
        self.c_area[idx, 0] = a1
        self.c_rel[idx, 0] = (a1 - self.target_area) / self.target_area
        self.c_area[alias, 0] = a1
        self.c_rel[alias, 0] = self.c_rel[idx, 0]
        self.ll[idx] = [result.lat, result.lon]
        self.ll[alias] = [result.lat, ap.lon]

    def solve_spine(self, idx):
        """
        Solve C on the spine meridian (lon=45) at distance s from the pole such that
        Area(PL1, PR1, C) = target.  Choose the *lower* root (not the pole).
        """
        if self.ll[idx][0] != -1:
            return

        tm = self.dim_size - 1
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = self.ll[lo_lt][1]
        lon_r = self.ll[lo_rt][1]
        spine_lon = 0.5 * (lon_l + lon_r)

        pl1, pr1 = self.par_ll(idx)
        if pl1 is None or pr1 is None:
            raise RuntimeError(f"Spine point {idx} missing parents: {self.par[idx]}")

        def area_at_s(s: float) -> float:
            c = meridian_point_from_pole(self.oracle.geod, spine_lon, s)
            return self.oracle.tri_area(pl1, pr1, c)

        # ---- robust bracket search for a spine root ----
        # area_at_s(s) is not guaranteed monotone for arbitrary (pl1, pr1) at deeper layers,
        # so scan along the spine meridian and pick the most southward root (last sign change).

        pole_ll = LatLon(90.0, spine_lon)

        inv_equ = self.oracle.geod.Inverse(pole_ll.lat, pole_ll.lon, 0.0, spine_lon)
        s_equ = float(inv_equ["s12"])  # pole -> equator along the spine meridian

        inv_pl = self.oracle.geod.Inverse(pole_ll.lat, pole_ll.lon, pl1.lat, pl1.lon)
        inv_pr = self.oracle.geod.Inverse(pole_ll.lat, pole_ll.lon, pr1.lat, pr1.lon)
        s_min = max(float(inv_pl["s12"]), float(inv_pr["s12"]))

        lo_scan = max(s_min * 1.000000001, s_min + 1e-6)  # just beyond parents
        hi_scan = s_equ

        def f(s: float) -> float:
            return area_at_s(s) - self.target_area

        n_scan = 512
        ss = np.linspace(lo_scan, hi_scan, n_scan, dtype=np.float64)
        fs = np.empty_like(ss)
        for i in range(n_scan):
            fs[i] = f(float(ss[i]))

        br_lo = None
        br_hi = None
        for i in range(n_scan - 1):
            f0 = float(fs[i])
            f1 = float(fs[i + 1])
            if not np.isfinite(f0) or not np.isfinite(f1):
                continue
            if f0 == 0.0:
                br_lo = float(ss[i])
                br_hi = float(ss[i])
            elif f0 * f1 < 0.0:
                br_lo = float(ss[i])
                br_hi = float(ss[i + 1])

        if br_lo is None or br_hi is None:
            raise RuntimeError(
                f"Failed to bracket spine triangle area on scan: idx={idx} plr={self.i2t[idx].tolist()} "
                f"lo={lo_scan} hi={hi_scan} f0={float(fs[0])} f1={float(fs[-1])}"
            )

        if br_lo == br_hi:
            result = meridian_point_from_pole(self.oracle.geod, spine_lon, br_lo)
        else:
            lo, hi = br_lo, br_hi
            result = None

        if result is None:
            f_lo = f(float(lo))
            f_hi = f(float(hi))
            if not np.isfinite(f_lo) or not np.isfinite(f_hi):
                raise RuntimeError(f"Non-finite spine bracket: idx={idx} lo={lo} hi={hi} f_lo={f_lo} f_hi={f_hi}")
            if f_lo == 0.0:
                result = meridian_point_from_pole(self.oracle.geod, spine_lon, float(lo))
            elif f_hi == 0.0:
                result = meridian_point_from_pole(self.oracle.geod, spine_lon, float(hi))
            else:
                for _ in range(self.max_iter):
                    mid = 0.5 * (lo + hi)
                    f_mid = f(float(mid))

                    a_scale = max(1.0, float(self.target_area))
                    if np.isfinite(f_mid) and abs(float(f_mid)) <= self.rel_area_tol * a_scale:
                        result = meridian_point_from_pole(self.oracle.geod, spine_lon, float(mid))
                        break

                    if (hi - lo) <= self.abs_s_tol_m:
                        result = meridian_point_from_pole(self.oracle.geod, spine_lon, float(mid))
                        break

                    # Maintain a sign-change bracket without assuming monotonicity of area.
                    if not np.isfinite(f_mid):
                        # Shrink conservatively if we hit a NaN.
                        lo = 0.5 * (lo + mid)
                        hi = 0.5 * (hi + mid)
                        f_lo = f(float(lo))
                        f_hi = f(float(hi))
                        continue

                    if f_lo * f_mid <= 0.0:
                        hi = mid
                        f_hi = f_mid
                    else:
                        lo = mid
                        f_lo = f_mid

        if result is None:
            result = meridian_point_from_pole(self.oracle.geod, spine_lon, 0.5 * (lo + hi))

        # Diagnostics: achieved constraint area.
        a1 = self.oracle.tri_area(pl1, pr1, result)
        self.c_area[idx, 0] = a1
        self.c_rel[idx, 0] = (a1 - self.target_area) / self.target_area
        self.ll[idx] = [result.lat, result.lon]

    def solve_twins(self, idx):
        """Solve the first mirrored 2DOF pair (X, X') under the spine.

        This implements the L2 template discussed:
          f1(lat, lon) = Area(A, B, X)  - a = 0
          f2(lat, lon) = Area(X, B, X') - a = 0

        where:
          - A is the already-known "shoulder" vertex above-left (e.g. 279)
          - B is the already-known spine vertex above (e.g. 288)
          - X' is the mirror of X across the spine: lon' = lon_mirror - lon

        We solve in the (lat, lon) parameterisation using a damped Newton method
        with finite-difference Jacobian. The only geometry oracle used is Karney
        PolygonArea on WGS84.

        Returns:
          (x, x_m) where x_m is the mirrored partner.

        Notes:
          - This is intentionally conservative and bracket-aware; it clamps lat/lon
            to provided bounds each iteration.
          - For robust convergence, pass an initial guess by setting lat_bounds and
            lon_bounds tight around the expected location (e.g. between the known
            row above and the next seam row below).
        """
        if self.ll[idx][0] != -1:  # solved already.
            return
        oracle = self.oracle
        tm = self.dim_size - 1
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = self.ll[lo_lt][1]
        lon_r = self.ll[lo_rt][1]
        spine_lon = (lon_l + lon_r) / 2.0
        pl, pr = self.par_ll(idx)  # should have both, but no sibling...
        a_shoulder = pl
        b_spine = pr
        lon_bounds = (lon_l, spine_lon)
        lat_hi = min(a_shoulder.lat, b_spine.lat) - 1e-9
        if lat_hi <= 0.0:
            lat_hi = min(a_shoulder.lat, b_spine.lat)
        lat_bounds = (0.0, float(lat_hi))
        lon_mirror = lon_r
        a = self.target_area

        def residuals(lat_v: float, lon_v: float) -> Tuple[float, float]:
            x = LatLon(lat_v, lon_v)
            x_m = LatLon(lat_v, _mirror_lon(lon_v, lon_mirror=lon_mirror))
            return (
                oracle.tri_area(a_shoulder, b_spine, x) - a,
                oracle.tri_area(x, b_spine, x_m) - a,
            )

        x = solve_latlon_2d(
            residuals_fn=residuals,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            max_iter=self.max_iter,
            f_scale=a,
            f_tol_rel=self.rel_area_tol,  # consider loosening this if needs be.
            step_damp=self.step_damp,
        )

        _p, _l, _r = self.i2t[idx]
        alias = self.t2i[_p, _r, _l]  # assumes rt_side = alias

        x_m = LatLon(x.lat, _mirror_lon(x.lon, lon_mirror=lon_mirror))
        # Diagnostics: achieved constraint areas.
        a1 = oracle.tri_area(a_shoulder, b_spine, x)
        a2 = oracle.tri_area(x, b_spine, x_m)
        self.c_area[idx, 0] = a1
        self.c_area[idx, 1] = a2
        self.c_rel[idx, 0] = (a1 - a) / a
        self.c_rel[idx, 1] = (a2 - a) / a
        self.c_area[alias, 0] = a1
        self.c_area[alias, 1] = a2
        self.c_rel[alias, 0] = self.c_rel[idx, 0]
        self.c_rel[alias, 1] = self.c_rel[idx, 1]
        self.ll[idx] = [x.lat, x.lon]
        self.ll[alias] = [x_m.lat, x_m.lon]

    def _bisect_lon(
            self,
            fn,
            lo: float,
            hi: float,
            *,
            max_iter: int | None = None,
            tol_rel: float | None = None,
    ) -> float:
        """Bisection on longitude for fn(lon)==0. Assumes lo/hi bracket a sign change."""
        max_iter = self.max_iter if max_iter is None else max_iter
        tol_rel = self.rel_area_tol if tol_rel is None else tol_rel
        f_lo = float(fn(lo))
        f_hi = float(fn(hi))
        if not np.isfinite(f_lo) or not np.isfinite(f_hi):
            raise ValueError("Non-finite bracket values in _bisect_lon")
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if f_lo * f_hi > 0.0:
            raise ValueError("No sign-change bracket in _bisect_lon")

        a_scale = max(1.0, float(self.target_area))
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = float(fn(mid))
            if not np.isfinite(f_mid):
                lo = 0.5 * (lo + mid)
                hi = 0.5 * (hi + mid)
                continue
            if abs(f_mid) <= tol_rel * a_scale:
                return mid
            if (f_lo <= 0.0 <= f_mid) or (f_mid <= 0.0 <= f_lo):
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    def solve_qtwin(self, idx):
        """Solve the equator-adjacent-to-spine point X (and its mirror) with a 1D equator solve.

        where:
          - A is the already-known "shoulder" vertex above-left (e.g. 279)
          - B is the already-known spine vertex above (e.g. 288)
          - X' is the mirror of X across the spine: lon' = lon_mirror - lon

        We solve in the (lat, lon) parameterisation using a robust 1D solve on the equator.
        """
        if self.ll[idx][0] != -1:  # solved already.
            return
        oracle = self.oracle
        tm = self.dim_size - 1
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = self.ll[lo_lt][1]
        lon_r = self.ll[lo_rt][1]
        spine_lon = (lon_l + lon_r) / 2.0
        _, pr = self.par_ll(idx)  # should have both, but no sibling...
        b_spine = pr   # We do not use shoulder here.
        lon_bounds = (lon_l, spine_lon)
        lon_mirror = lon_r
        a = self.target_area

        # ---- 1D solve on the equator (lat == 0) ----
        def f1(lon_v: float) -> float:
            """ [0, 45 - x], [0, 45 + x] [ spine_parent.lat,  45] """
            xt1 = LatLon(0.0, spine_lon - lon_v)
            xt2 = LatLon(0.0, spine_lon + lon_v)
            return oracle.tri_area(xt1, xt2, b_spine) - a

        # Find a bracket for f1 by scanning (robust to local non-monotone behaviour).
        lo, hi = float(lon_bounds[0]), float(lon_bounds[1])
        n_scan = 512
        lons = np.linspace(lo, hi, n_scan)
        f1v = np.array([f1(float(x)) for x in lons], dtype=np.float64)

        def _best_bracket(vals: np.ndarray) -> tuple[float, float] | None:
            s = np.sign(vals)
            best = None
            for i in range(len(s) - 1):
                if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
                    continue
                if s[i] == 0.0:
                    best = (float(lons[i]), float(lons[i]))
                elif s[i] * s[i + 1] < 0.0:
                    best = (float(lons[i]), float(lons[i + 1]))
            return best

        br = _best_bracket(f1v)
        if br is None:
            # Fallback: pick lon with smallest |f1|.
            j = int(np.nanargmin(np.abs(f1v)))
            lon_best = float(lons[j])
        else:
            if br[0] == br[1]:
                lon_best = float(br[0])
            else:
                lon_best = float(self._bisect_lon(f1, br[0], br[1]))

        x = LatLon(0.0, spine_lon - lon_best)

        _p, _l, _r = self.i2t[idx]
        alias = self.t2i[_p, _r, _l]  # assumes rt_side = alias

        x_m = LatLon(0.0, _mirror_lon(x.lon, lon_mirror=lon_mirror))

        # Diagnostics: achieved constraint area (only constraint-0 is enforced for QTWIN).
        a1 = oracle.tri_area(b_spine, x, x_m)
        self.c_area[idx, 0] = a1
        self.c_rel[idx, 0] = (a1 - a) / a
        self.c_area[alias, 0] = a1
        self.c_rel[alias, 0] = self.c_rel[idx, 0]
        # constraint-1 is not applicable for QTWIN (triangle spans across spine, not a microtriangle)
        self.c_area[idx, 1] = np.nan
        self.c_rel[idx, 1] = np.nan
        self.c_area[alias, 1] = np.nan
        self.c_rel[alias, 1] = np.nan

        self.ll[idx] = [x.lat, x.lon]
        self.ll[alias] = [x_m.lat, x_m.lon]

    def solve_inner(self, idx):
        """Solve for an interior vertex X using two equal-area constraints.

        Key rule: pick constraints from the *actual* mesh triangles (self.tri_grid).
        For an INNER vertex, we choose two triangles containing `idx` such that
        `idx` is the last unknown vertex (the other two vertices already have ll).

        This avoids the 'greedy' branch where one of the adjacent (typically down/roof)
        triangles is not used to constrain X and later accumulates error near the equator.
        """
        if self.ll[idx][0] != -1:  # solved already.
            return

        oracle = self.oracle
        a = self.target_area
        tm = self.dim_size - 1

        # Octant longitudes; canonical side is lon in [lon_l, spine_lon].
        lo_lt, lo_rt = self.t2i[tm, 0, tm], self.t2i[tm, tm, 0]
        lon_l = float(self.ll[lo_lt][1])
        lon_r = float(self.ll[lo_rt][1])
        spine_lon = 0.5 * (lon_l + lon_r)

        lon_bounds = (lon_l, spine_lon)

        # Conservative latitude upper bound: must be south of any already-known neighbours.
        # We compute this from solved vertices in triangles that touch idx.
        tri = self.tri_grid
        hits = np.any(tri == idx, axis=1)
        tri_idx = np.flatnonzero(hits)

        # Collect triangles where the other two vertices are already solved.
        # Each element is (u_idx, v_idx) for the known edge opposite idx.
        known_edges: list[tuple[int, int]] = []
        lat_cap = 90.0
        for ti in tri_idx:
            a_i, b_i, c_i = tri[ti]
            if a_i == idx:
                u, v = int(b_i), int(c_i)
            elif b_i == idx:
                u, v = int(a_i), int(c_i)
            else:
                u, v = int(a_i), int(b_i)
            if self.ll[u][0] != -1.0 and self.ll[v][0] != -1.0:
                known_edges.append((u, v))
                lat_cap = min(lat_cap, float(self.ll[u][0]), float(self.ll[v][0]))

        if len(known_edges) < 2:
            # Fallback to the original dependency-based approach (parents + sibling)
            pl, pr = self.par_ll(idx)
            sb = self.sib_ll(idx)
            if pl is None or pr is None or sb is None:
                raise RuntimeError(f"INNER {idx} missing deps: par={self.par[idx]} sib={self.sib[idx]}")
            lat_cap = min(float(pl.lat), float(pr.lat), float(sb.lat))
            known_edges = []
            # Prefer the two 'up' triangles first.
            # (pl, sb, X) and (sb, pr, X)
            # Use indices, but we can just compute with LatLon objects.
            def residuals(lat: float, lon: float):
                x = LatLon(lat, lon)
                return (
                    oracle.tri_area(pl, sb, x) - a,
                    oracle.tri_area(sb, pr, x) - a,
                )

            lat_hi = max(0.0, float(lat_cap) - 1e-9)
            if lat_hi <= 0.0:
                lat_hi = float(lat_cap)
            lat_bounds = (0.0, lat_hi)

            x = solve_latlon_2d(
                residuals_fn=residuals,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                max_iter=self.max_iter,
                f_scale=a,
                f_tol_rel=self.rel_area_tol,
                step_damp=self.step_damp,
            )

            p, l, r = self.i2t[idx]
            alias = self.t2i[p, r, l]
            x_m = LatLon(x.lat, _mirror_lon(x.lon, lon_mirror=lon_r))

            a1 = oracle.tri_area(pl, sb, x)
            a2 = oracle.tri_area(sb, pr, x)
            self.c_area[idx, 0] = a1
            self.c_area[idx, 1] = a2
            self.c_rel[idx, 0] = (a1 - a) / a
            self.c_rel[idx, 1] = (a2 - a) / a
            self.c_area[alias, 0] = a1
            self.c_area[alias, 1] = a2
            self.c_rel[alias, 0] = self.c_rel[idx, 0]
            self.c_rel[alias, 1] = self.c_rel[idx, 1]

            self.ll[idx] = [x.lat, x.lon]
            self.ll[alias] = [x_m.lat, x_m.lon]
            return

        lat_hi = max(0.0, float(lat_cap) - 1e-9)
        if lat_hi <= 0.0:
            lat_hi = float(lat_cap)
        lat_bounds = (0.0, lat_hi)

        # Initial guess: midpoint of search box.
        lat0 = 0.5 * (lat_bounds[0] + lat_bounds[1])
        lon0 = 0.5 * (lon_bounds[0] + lon_bounds[1])

        # Try all pairs of usable triangles; pick the candidate that minimises
        # the max relative error over *all* usable adjacent triangles.
        def solve_for_edge_pair(e1: tuple[int, int], e2: tuple[int, int]) -> LatLon:
            u1, v1 = e1
            u2, v2 = e2
            uu1 = LatLon(*self.ll[u1])
            vv1 = LatLon(*self.ll[v1])
            uu2 = LatLon(*self.ll[u2])
            vv2 = LatLon(*self.ll[v2])

            def residuals(lat: float, lon: float):
                x = LatLon(lat, lon)
                return (
                    oracle.tri_area(uu1, vv1, x) - a,
                    oracle.tri_area(uu2, vv2, x) - a,
                )

            return solve_latlon_2d(
                residuals_fn=residuals,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                lat0=lat0,
                lon0=lon0,
                max_iter=self.max_iter,
                f_scale=a,
                f_tol_rel=self.rel_area_tol,
                step_damp=self.step_damp,
            )

        # Precompute LatLon for edges to score quickly.
        edge_ll: list[tuple[LatLon, LatLon]] = []
        for u, v in known_edges:
            edge_ll.append((LatLon(*self.ll[u]), LatLon(*self.ll[v])))

        def score(x: LatLon) -> float:
            # Max absolute relative deviation over all usable adjacent triangles.
            worst = 0.0
            for uu, vv in edge_ll:
                e = (oracle.tri_area(uu, vv, x) - a) / a
                ae = abs(float(e))
                if ae > worst:
                    worst = ae
            return float(worst)

        best_x = None
        best_pair = None
        best_score = float("inf")
        n = len(known_edges)
        for i in range(n - 1):
            for j in range(i + 1, n):
                x_try = solve_for_edge_pair(known_edges[i], known_edges[j])
                s_try = score(x_try)
                if s_try < best_score:
                    best_score = s_try
                    best_x = x_try
                    best_pair = (known_edges[i], known_edges[j])

        if best_x is None or best_pair is None:
            raise RuntimeError(f"INNER {idx} failed to produce a candidate")

        p, l, r = self.i2t[idx]
        alias = self.t2i[p, r, l]
        x_m = LatLon(best_x.lat, _mirror_lon(best_x.lon, lon_mirror=lon_r))

        # Store diagnostics for the two constraints used.
        (u1, v1), (u2, v2) = best_pair
        uu1 = LatLon(*self.ll[u1])
        vv1 = LatLon(*self.ll[v1])
        uu2 = LatLon(*self.ll[u2])
        vv2 = LatLon(*self.ll[v2])
        a1 = oracle.tri_area(uu1, vv1, best_x)
        a2 = oracle.tri_area(uu2, vv2, best_x)

        self.c_area[idx, 0] = a1
        self.c_area[idx, 1] = a2
        self.c_rel[idx, 0] = (a1 - a) / a
        self.c_rel[idx, 1] = (a2 - a) / a
        self.c_area[alias, 0] = a1
        self.c_area[alias, 1] = a2
        self.c_rel[alias, 0] = self.c_rel[idx, 0]
        self.c_rel[alias, 1] = self.c_rel[idx, 1]

        self.ll[idx] = [best_x.lat, best_x.lon]
        self.ll[alias] = [x_m.lat, x_m.lon]

    def solve_fixed(self, idx):
        pass

    def solve_alias(self, idx):
        # Populate alias points by mirroring their canonical partner (swap l<->r).
        if self.ll[idx][0] != -1:
            return
        tm = self.dim_size - 1
        lo_rt = self.t2i[tm, tm, 0]
        lon_r = float(self.ll[lo_rt][1])

        p, l, r = self.i2t[idx]
        src = self.t2i[p, r, l]
        if src < 0 or self.ll[src][0] == -1:
            return

        lat_s, lon_s = float(self.ll[src][0]), float(self.ll[src][1])
        self.ll[idx] = [lat_s, _mirror_lon(lon_s, lon_mirror=lon_r)]

        # Mirror diagnostic areas/residuals too.
        self.c_area[idx, :] = self.c_area[src, :]
        self.c_rel[idx, :] = self.c_rel[src, :]

    def compute(self) -> None:
        """
        Solve the latitude/longitudes of all points of an authalic triangular grid
        on an OCTANT of WGS84 with minimal solves per cell, following the criteria:
        seams fixed (meridians / equator edges),
        spine fixed (45° meridian)
        edges are geodesic segments
        per-triangle area target fixed
        symmetry enforced
        """
        f_msk = self.af != AF.ALIAS
        f_lut = {
            AF.FIXED: self.solve_fixed,
            AF.SPINE: self.solve_spine,
            AF.SEAMM: self.solve_seamm,
            AF.ALIAS: self.solve_alias,
            AF.INNER: self.solve_inner,
            AF.SEAMG: self.solve_seamg,
            AF.TWINS: self.solve_twins,
            AF.QTWIN: self.solve_qtwin,
            AF.SEAMQ: self.solve_seamq,
        }
        for row in range(1, self.dim_size):
            r_msk = self.i2t[:, 0] == row
            rxx = np.flatnonzero(r_msk & f_msk)
            tm = self.dim_size - 1
            # Default: process right-to-left by index.
            order = rxx[::-1]

            # Special-case equator: solve from spine -> seam so each SEAMQ has a solved spineward neighbour.
            if row == tm:
                # Canonical side only (exclude aliases) and sort by l descending (closest to spine first).
                plr_row = self.i2t[order]
                l_row = plr_row[:, 1]
                r_row = plr_row[:, 2]
                canon = r_row >= l_row
                order = order[canon]
                # Descending l
                order = order[np.argsort(-l_row[canon])]

            for idx in order:
                plr = self.i2t[idx]
                fn = AF(self.af[idx])
                f_lut[fn](idx)
                ll = self.ll[idx]
                crel = self.c_rel[idx]
                if np.isfinite(crel[1]):
                    err_txt = f" c_rel=({crel[0]:+.3e},{crel[1]:+.3e})"
                elif np.isfinite(crel[0]):
                    err_txt = f" c_rel=({crel[0]:+.3e})"
                else:
                    err_txt = ""
                print(f'row {row}, id:{idx}; plr:{plr} solve:{repr(fn)}, ll:{repr(ll)}{err_txt}')
        # Finally, fill all alias points for output/visualisation.
        alias_idx = np.flatnonzero(self.af == AF.ALIAS)
        for idx in alias_idx:
            self.solve_alias(int(idx))
        self.compute_triangle_areas()

    def compute_triangle_areas(self):
        for i, (a, b, c) in enumerate(self.tri_grid):
            va = LatLon(*self.ll[a])
            vb = LatLon(*self.ll[b])
            vc = LatLon(*self.ll[c])
            area = self.oracle.tri_area(va, vb, vc)
            self.tri_area[i] = area
            self.tri_rel[i] = (area - self.target_area) / self.target_area

    def report_grid(self):
        """Human-readable triangle report.

        NOTE: `tri_area`/`tri_rel` are per-triangle arrays (len == 9**hex_layer), not per-vertex.
        This report enumerates triangles from `self.tri_grid` and prints their vertex indices,
        lattice coordinates, area, and relative error.
        """
        print("\n[Triangles]")
        for i, (ia, ib, ic) in enumerate(self.tri_grid):
            a = float(self.tri_area[i])
            d = float(self.tri_rel[i])
            ta = self.i2t[ia].tolist()
            tb = self.i2t[ib].tolist()
            tc = self.i2t[ic].tolist()
            print(
                f"{i:02d}: v=({ia:02d},{ib:02d},{ic:02d}) "
                f"plr=({ta},{tb},{tc}) area:{a:.6f} d:{d:+.6e}"
            )

    def report_area_error(self) -> dict:
        """Compute per-triangle area stats and total closure error vs octant area."""
        # tri = self.tri_grid
        # areas = np.empty(tri.shape[0], dtype=np.float64)
        # for i, (a, b, c) in enumerate(tri):
        #     ta = LatLon(*self.ll[a])
        #     tb = LatLon(*self.ll[b])
        #     tc = LatLon(*self.ll[c])
        #     areas[i] = self.oracle.tri_area(ta, tb, tc)

        total = float(self.tri_area.sum())
        closure = total - float(self.oa)
        rel_closure = closure / float(self.oa) if self.oa else float('nan')

        stats = {
            "tri_count": int(self.tri_area.size),
            "target_area": float(self.target_area),
            "area_min": float(self.tri_area.min()),
            "area_max": float(self.tri_area.max()),
            "area_mean": float(self.tri_area.mean()),
            "area_std": float(self.tri_area.std()),
            "total_calculated_area": total,
            "octant_area": float(self.oa),
            "closure_error_m2": float(closure),
            "closure_error_rel": float(rel_closure),
            "max_rel_dev": float(np.max(np.abs(self.tri_rel))),
        }
        print("[Lattice] area stats:")
        for k in (
            "tri_count",
            "target_area",
            "area_min",
            "area_max",
            "area_mean",
            "area_std",
            "total_calculated_area",
            "octant_area",
            "closure_error_m2",
            "closure_error_rel",
            "max_rel_dev",
        ):
            print(f"  {k}: {stats[k]}")
        return stats

    def _tri_grid(self):
        """Generate indices of grid"""
        tm = self.dim_size - 1
        def _vx(i, j):
            p, r, l = i, tm - j, tm - (i - j)
            return self.t2i[p, l, r]
        tri = []
        for i in range(tm):
            for j in range(i + 1):
                tri.append((_vx(i, j), _vx(i + 1, j), _vx(i + 1, j + 1)))
        for i in range(1, tm):
            for j in range(i):
                tri.append((_vx(i, j), _vx(i, j + 1), _vx(i + 1, j + 1)))
        return np.array(tri, dtype=int)

    def save(self, f_name):
        """Save lattice grid"""
        layer = self.layer
        fn_npz = Path(f"output/{f_name}_l{layer}.npz")
        np.savez(
            fn_npz,
            i2t=self.i2t,
            t2i=self.t2i,
            af=self.af,
            par=self.par,
            sib=self.sib,
            grid=self.tri_grid,
            ll=self.ll,
            tri_area=self.tri_area,
            tri_rel=self.tri_rel,
            meta=dict(layer=layer, timestamp=float(time.time()))
        )
        print(f"[Lattice] wrote {fn_npz.name}: for hex_layer {layer}")


if __name__ == "__main__":
    ora = AreaOracle()
    for layer in [2]:
        la = Lattice(layer, ora)
        la.compute()
        la.report_grid()
        la.report_area_error()

        la.save("authalic")
