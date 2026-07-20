# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Registrar is the central registry of the Hex9 coordinate domains, point-formats, and projections.
It exposes a uniform API for discovering domains, instantiating them on demand,
 and composing projection chains between them. It acts as dependency-resolver and orchestration hex_layer.
"""
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from geographiclib.geodesic import Geodesic
from .domain import Domain
from .point_format import PointFormat
from .points import Points
from .composite import ComponentDomain, CompositeDomain
from .projection import Projection


# Maps frozenset({domain_a, domain_b}) -> projection name for inter-domain projections
# that neither domain self-registers on init.
_PAIR_TO_PROJ: dict[frozenset, str] = {
    frozenset(['p_pix', 'g_gcd']): 'pix_gcd',
    frozenset(['r_gcd', 'g_gcd']): 'rxd_gcd',
    frozenset(['c_ell', 'g_gcd']): 'ell_gcd',
    frozenset(['g_gcd', 'b_raw']): 'gcd_brw',
    frozenset(['b_raw', 'b_oct']): 'brw_bct',
    frozenset(['c_oct', 'c_ell']): 'oct_ell',
    frozenset(['c_sph', 'g_gcd']): 'aut_gcd',
    frozenset(['c_oct', 'c_sph']): 'oct_sph',
}


class Registrar:
    """
    Registrar manages the registers of
    • coordinate sets (as classes) (with their addresses formats)
    • projections (as classes)
    """

    def __init__(self):
        self.ellipsoid = Geodesic.WGS84
        self.ellipsoid_name = 'WGS84'
        # via_sphere: route g_gcd↔b_raw through the unit authalic sphere
        # (c_sph) — exact authalic-latitude datum reduction at the boundary,
        # purely spherical octahedral engine (AK a=b=1, Sphere-trained warp)
        # inside. c_ell keeps the true source ellipsoid either way.
        # DEFAULT since 2026-07-18: the via-sphere chain (equal-area ~20×
        # tighter, round-trip parity — see L6_MOBIUS_WARP.md §3.6). The
        # classic chain (ellipsoid AK + per-ellipsoid trained warp) remains
        # available via set_ellipsoid(..., via_sphere=False); note the two
        # chains yield different b_oct coordinates (and hence addresses at
        # depth), and the libhex9 accelerator currently implements only the
        # classic WGS84 chain. Set BEFORE the first domain() call.
        self.via_sphere = True
        self._ellipsoid_area = None
        self._domains = {}
        self._projections = {}
        self._domain_projections = {}
        self._formats = {}
        self._bridges = {}

    @property
    def ellipsoid_area(self) -> float:
        """Surface area of the reference ellipsoid in m², computed once and cached."""
        if self._ellipsoid_area is None:
            if self.ellipsoid.f == 0.0:
                # sphere: the oblate-spheroid formula is singular at e=0
                self._ellipsoid_area = 4.0 * np.pi * self.ellipsoid.a ** 2
            else:
                try:
                    from hhg9.algorithms.distance import calculate_ellipsoid_area
                    self._ellipsoid_area = float(calculate_ellipsoid_area(
                        self.ellipsoid.a, f'1/{self.ellipsoid.f}'
                    ))
                except Exception:
                    self._ellipsoid_area = 510065621724088.50944  # WGS84 fallback
        return self._ellipsoid_area

    def set_ellipsoid(self, *, a, f=None, inv_f=None, name='Unknown Ellipsoid',
                      via_sphere=None):
        """Set the reference ellipsoid. Pass either f (flattening) or inv_f
        (inverse flattening). via_sphere routes the g_gcd↔b_raw chain
        through the unit authalic sphere (exact geodetic↔authalic latitude
        at the boundary; spherical AK core; Sphere-trained warp) — c_ell
        and geodesic calculations keep the true ellipsoid set here.
        None (default) preserves the registrar's current setting; pass
        False explicitly for the classic per-ellipsoid chain."""
        if f is None and inv_f is not None:
            f = 1.0 / inv_f
        elif f is None:
            raise ValueError("set_ellipsoid requires either f or inv_f")
        self.ellipsoid = Geodesic(a, f)
        self._ellipsoid_area = None
        self.ellipsoid_name = name
        if via_sphere is not None:
            self.via_sphere = bool(via_sphere)

    def register_bridge(self, _chain: list):
        """Register a projection chain."""
        chain = [self._dom(a) for a in _chain]
        # chain = self._check_chain(chain)
        ch = [chain[0].name, chain[-1].name]  # these are the endpoints.
        if tuple(ch) in self._bridges or ch[0] == ch[-1]:
            return
        self._bridges[tuple(ch)] = chain
        self._bridges[tuple(ch[::-1])] = chain[::-1]

    def register_projection(self, obj, chain: NDArray):
        """Register a projection. Normally managed in the Projection init class."""
        if isinstance(obj, Projection):
            if obj.name not in self._projections:
                self._projections[obj.name] = obj
            fwd, bak = tuple(chain), tuple(reversed(chain))
            if fwd not in self._domain_projections:
                self._domain_projections[fwd] = {}
            self._domain_projections[fwd][obj.name] = obj.forward
            if bak not in self._domain_projections:
                self._domain_projections[bak] = {}
            self._domain_projections[bak][obj.name] = obj.backward

    def register_format(self, fmt: PointFormat):
        """Register set_format."""
        self._formats[fmt.name] = fmt

    def register_domain(self, dom: Domain):
        """Register set_domain."""
        self._domains[dom.name] = dom

    def domain(self, full_key):
        """
        return domain by key
        domain variations are declared with a ':'
        """
        if full_key not in self._domains:
            variation = full_key.split(':') if ':' in full_key else [full_key]
            key = variation[0]
            match key:
                case 'p_pix':
                    from hhg9.domains import PlatePixel
                    _ = PlatePixel(self)
                case 'g_gcd':
                    from hhg9.domains import GeneralGCD
                    _ = GeneralGCD(self)
                case 'r_gcd':
                    from hhg9.domains import RadiansGCD
                    _ = RadiansGCD(self)
                case 'c_ell':
                    from hhg9.domains import EllipsoidCartesian
                    _ = EllipsoidCartesian(self)
                case 'c_oct':
                    from hhg9.domains import OctahedralCartesian
                    _ = OctahedralCartesian(self)
                case 'w_oct':
                    # Same 3D octahedral container as c_oct, but reached warp-free
                    # from b_oct (post-warp coords lifted straight to 3D). No path
                    # to c_ell is wired, so AK can never be applied to it.
                    from hhg9.domains import OctahedralCartesian
                    _ = OctahedralCartesian(self, 'w_oct', link_boct=True)
                case 'b_oct':
                    from hhg9.domains import OctahedralBarycentric
                    _ = OctahedralBarycentric(self)
                case 'b_raw':
                    from hhg9.domains import OctahedralBaryRaw
                    _ = OctahedralBaryRaw(self)
                case 's_oct':
                    from hhg9.domains import OctahedralSimplex
                    _ = OctahedralSimplex(self)
                case 'c_sph':
                    from hhg9.domains import SphericalCartesian
                    _ = SphericalCartesian(self)
                case 'n_oct':
                    from hhg9.domains import OctahedralNet
                    layout = 'mortar'
                    theta = None
                    match len(variation):
                        case 1:
                            pass
                        case 2:
                            layout = variation[1]
                        case _:
                            layout = variation[1]
                            v2 = variation[2]
                            if len(v2) == 4:
                                theta = f'{int(variation[2]):04d}'
                    _ = OctahedralNet(self, layout=layout, theta=theta)
                case 'n_pix':
                    from hhg9.domains import NetPixel
                    _ = NetPixel(self)
                case _:
                    raise KeyError(key)
        return self._domains[full_key]

    def projection(self, full_key):
        """return projection by key"""
        key, variation = full_key.split(':') if ':' in full_key else (full_key, None)
        if key not in self._projections:
            if key in self._bridges:
                chain = self._bridges[key]
                return chain
            match key:
                case 'pix_gcd':
                    from hhg9.projections import PlatePixelGCD
                    _ = PlatePixelGCD(self)
                case 'plt_net':
                    from hhg9.projections import PlatePixelNet
                    doms = list(self._domains.keys())
                    nets = [k for k in doms if k[:6] == 'n_oct:' and k[6:].find(':') == -1]
                    for net in nets:
                        _ = PlatePixelNet(self, net)
                case 'oct_ell':
                    from hhg9.projections import AKOctahedralEllipsoid
                    _ = AKOctahedralEllipsoid(self)
                case 'oct_sph':
                    # AK on the unit authalic sphere (a=b=1) — the datum-free
                    # engine of the via-sphere chain.
                    from hhg9.projections import AKOctahedralEllipsoid
                    _ = AKOctahedralEllipsoid(self, name='oct_sph',
                                              a=1.0, b=1.0, cart='c_sph')
                case 'aut_gcd':
                    from hhg9.projections import AuthalicGCD
                    _ = AuthalicGCD(self)
                case 'ell_gcd':
                    from hhg9.projections import EllipsoidGCD
                    _ = EllipsoidGCD(self)
                case 'ell_gcr':
                    from hhg9.projections import EllipsoidGCDRad
                    _ = EllipsoidGCDRad(self)
                case 'rxd_gcd':
                    from hhg9.projections import RGCD_GCD
                    _ = RGCD_GCD(self)
                case 'gcd_bry':
                    from hhg9.projections import GCDBary
                    _ = GCDBary(self)
                case 'gcd_brw':
                    from hhg9.projections import GCDBraw
                    _ = GCDBraw(self)
                case 'brw_bct':
                    from hhg9.projections import BrawBoct
                    _ = BrawBoct(self)
                case _:
                    raise KeyError(key)
        return self._projections[key]

    def format(self, key):
        """
        return domain by key
        domain variations are declared with a ':'
        """
        if key not in self._formats:
            match key:
                case 'dec':
                    from hhg9.formats import DecimalCartesian
                    _ = DecimalCartesian(self)
                case 'deg':
                    from hhg9.formats import DecimalDegrees
                    _ = DecimalDegrees(self)
                case 'dms':
                    from hhg9.formats import DMS
                    _ = DMS(self)
                case 'h9':
                    from hhg9.formats import OctahedralH9
                    _ = OctahedralH9(self)
                case _:
                    raise KeyError(key)
        return self._formats[key]

    def _cmp_key(self, a, b, a_bins, b_bins):
        key = None
        if a_bins and isinstance(b, ComponentDomain):
            sign = b.sig()
            if sign not in a_bins:
                raise ValueError(f'b {sign} not in a bins')
            else:
                key = a_bins[sign], b.name
                if key not in self._domain_projections:
                    raise ValueError(f'chain {key} not found')
        elif b_bins and isinstance(a, ComponentDomain):
            sign = a.sig()
            if sign not in b_bins:
                raise ValueError(f'a {sign} not in b bins')
            else:
                key = a.name, b_bins[sign]
                if key not in self._domain_projections:
                    raise ValueError(f'chain {key} not found')
        return key

    def _dom(self, dom):
        if isinstance(dom, str):
            if dom not in self._domains:
                return self.domain(dom)
            else:
                return self._domains[dom]
        else:
            if dom.name not in self._domains:
                self.register_domain(dom)
            return dom

    def _check_chain(self, chain_):
        chain = []
        if not chain_:
            return chain

        # Extract the actual domain objects first to simplify logic
        doms = [self._dom(p) for p in chain_]

        for i, (a, b) in enumerate(pairwise(doms)):
            key = a.name, b.name
            pair = frozenset(key)

            # via_sphere: the c_oct↔c_ell hop must route through the authalic
            # sphere (c_oct↔c_sph↔g_gcd↔c_ell). The classic oct_ell map is a
            # different projection and yields different b_oct coordinates, so
            # mixing it into a via-sphere chain shifts positions by kilometres.
            if self.via_sphere and pair == frozenset(('c_oct', 'c_ell')):
                names = ('c_oct', 'c_sph', 'g_gcd', 'c_ell')
                if key[0] != 'c_oct':
                    names = names[::-1]
                links = [self._dom(n) for n in names]
                if not chain or chain[-1] != links[0]:
                    chain.append(links[0])
                chain.extend(links[1:])
                continue

            # Handle direct projections
            if key in self._domain_projections:
                # Only add 'a' if it's the start or isn't a duplicate of the last entry
                if i == 0:
                    chain.append(a)
                chain.append(b)
            elif pair in _PAIR_TO_PROJ:
                if i == 0:
                    chain.append(a)
                chain.append(b)

            elif key in self._bridges:
                links = self._bridges[key]
                for j, (c, d) in enumerate(pairwise(links)):
                    # Avoid adding 'c' if it's already the end of the chain
                    if not chain or chain[-1] != c:
                        chain.append(c)
                    chain.append(d)

            elif isinstance(a, CompositeDomain) and isinstance(b, CompositeDomain):
                # Composite-composite: validated and dispatched per-component in project()
                shared = a.sides.keys() & b.sides.keys()
                if not (shared and len(shared) == len(a.sides)):
                    raise ValueError(f'A projection {key} is not registered.')
                if i == 0:
                    chain.append(a)
                chain.append(b)

            else:
                raise ValueError(f'A projection {key} is not registered.')

        return chain

    def _project_composites(self, pts: Points, a, a2b):
        if pts.oid is None:
            a.binning(pts)
        uvw = pts.oid
        res = None
        for sig, cmp in a.sides.items():
            key = (cmp.name, a2b[cmp].name)
            facilitator = next(iter(self._domain_projections[key]))
            mask = uvw == cmp.oid
            crds = pts.coords[mask]
            if crds.size > 0:
                rex = self._domain_projections[key][facilitator](crds)
                if res is None:
                    res = np.zeros([pts.coords.shape[0], rex.shape[-1]])
                res[mask] = rex
        if res is None:
            res = np.zeros_like(pts.coords)
        return Points(res, samples=pts.samples, oid=pts.oid)

    def project(self, coords: Points, chain: NDArray | list) -> Points:
        """
        Transform coordinates through a chain of domains.
        """
        chain = self._check_chain(chain)
        coords.domain = self._dom(coords.domain)
        if coords.domain != chain[0]:  # allow for implicit 'from'.
            chain = [coords.domain] + chain
        for (a, b) in pairwise(chain):
            key = a.name, b.name
            if key in self._domain_projections:
                alts = self._domain_projections[key]
                coords = alts[next(iter(alts))](coords)
                continue
            # Check _PAIR_TO_PROJ before composite dispatch (works for any pair type).
            k0 = key[0].split(':')[0] if ':' in key[0] else key[0]
            lookup_key = (k0, key[1])
            pair = frozenset(lookup_key)
            if pair in _PAIR_TO_PROJ:
                self.projection(_PAIR_TO_PROJ[pair])
                alts = self._domain_projections.get(key) or self._domain_projections[lookup_key]
                coords = alts[next(iter(alts))](coords)
                continue
            # Composite fallback: project component-wise without a pairwise registration.
            if isinstance(a, CompositeDomain) and isinstance(b, CompositeDomain):
                a_components = a.sides
                b_components = b.sides
                shared = a_components.keys() & b_components.keys()
                if shared and len(shared) == len(a_components):
                    a2b = {a_components[k]: b_components[k] for k in shared}
                    coords = self._project_composites(coords, a, a2b)
                    coords.domain = b
                    continue
            raise ValueError(f'A projection {key} is not registered.')
        return coords
