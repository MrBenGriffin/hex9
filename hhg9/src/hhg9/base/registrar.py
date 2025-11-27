# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Registrar is the central registry of the Hex9 coordinate domains, point-formats, and projections.
It exposes a uniform API for discovering domains, instantiating them on demand,
 and composing projection chains between them. It acts as dependency-resolver and orchestration layer.
"""
from functools import cache
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from .domain import Domain
from .point_format import PointFormat
from .points import Points
from .composite import ComponentDomain, CompositeDomain
from .projection import Projection


class Registrar:
    """
    Registrar manages the registers of
    • coordinate sets (as classes) (with their addresses formats)
    • projections (as classes)
    """

    @cache
    def __init__(self):
        self._domains = {}
        self._projections = {}
        self._domain_projections = {}
        self._formats = {}

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
        elif isinstance(obj, str) and obj == 'chain':
            # Ensure all domains in the chain are registered
            for csys in chain:
                if isinstance(csys, Domain):
                    if csys.name not in self._domains:
                        self.register_domain(csys)
                elif isinstance(csys, str):
                    dom = self.domain(csys)
                    # if csys not in self._domains:
                    #     raise ValueError(f"{csys} Unregistered domain")
            # Materialise names for the chain
            ch = [val if isinstance(val, str) else val.name for val in chain]

            # Register the full chain from first→last and its reverse
            fwd, bak = (ch[0], ch[-1]), (ch[-1], ch[0])
            self._domain_projections.setdefault(fwd, {})['chain'] = ch
            self._domain_projections.setdefault(bak, {})['chain'] = list(reversed(ch))

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
            key, variation = full_key.split(':') if ':' in full_key else (full_key, None)
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
                case 'b_oct':
                    from hhg9.domains import OctahedralBarycentric
                    _ = OctahedralBarycentric(self)
                case 's_oct':
                    from hhg9.domains import OctahedralSimplex
                    _ = OctahedralSimplex(self)
                case 'c_sph':
                    from hhg9.domains import SphericalCartesian
                    _ = SphericalCartesian(self)
                case 'n_oct':
                    c_oct = self.domain('c_oct')
                    b_oct = self.domain('b_oct')
                    from hhg9.domains import OctahedralNet
                    _ = OctahedralNet(self, layout=variation)
                case 'n_pix':
                    from hhg9.domains import NetPixel
                    _ = NetPixel(self)
                case _:
                    raise KeyError(key)
        return self._domains[full_key]

    def projection(self, full_key):
        """return projection by key"""
        if full_key not in self._domains:
            key, variation = full_key.split(':') if ':' in full_key else (full_key, None)
        # if key not in self._projections:
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
                case 'ell_gcd':
                    # EllipsoidGCD(registrar, 'ell_gcd', 'c_ell', 'g_gcd')
                    from hhg9.projections import EllipsoidGCD
                    _ = EllipsoidGCD(self)
                case 'ell_gcr':
                    from hhg9.projections import EllipsoidGCDRad
                    _ = EllipsoidGCDRad(self)
                case 'gcd_bry':
                    from hhg9.projections import GCDBary
                    _ = GCDBary(self)
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
        for dom in chain_:
            chain.append(self._dom(dom))
        return chain

    def _project_composites(self, pts: Points, a, a2b):
        if pts.components is None:
            pts = a.binning(pts)
        # pts.coords = np.atleast_2d(pts.coords)
        res = np.zeros_like(pts.coords)
        uvw = (pts.components >= 0) @ (4, 2, 1)
        for sig, cmp in a.components.items():
            key = (cmp.name, a2b[cmp].name)
            facilitator = next(iter(self._domain_projections[key]))
            side = np.asarray(sig, dtype='b')
            ref = (side >= 0) @ (4, 2, 1)
            crds = pts.coords[uvw == ref]  # these are the coordinates for this projection.
            if crds.size > 0:
                rex = self._domain_projections[key][facilitator](crds)
                if rex.shape[-1] != res.shape[-1]:
                    res = np.zeros([pts.coords.shape[0], rex.shape[-1]])
                res[uvw == ref] = rex
        return Points(res, samples=pts.samples, components=pts.components)

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
            if key not in self._domain_projections:
                # Composite fallback: if both domains are composite and have matching components,
                # project component-wise without requiring a pairwise registration.
                if isinstance(a, CompositeDomain) and isinstance(b, CompositeDomain):
                    a_components = a.components
                    b_components = b.components
                    shared = a_components.keys() & b_components.keys()
                    # Require a one-to-one mapping for all a's components
                    if shared and len(shared) == len(a_components):
                        a2b = {a_components[k]: b_components[k] for k in shared}
                        coords = self._project_composites(coords, a, a2b)
                        coords.domain = b
                        continue
                else:
                    if ':' in key[0]:
                        k0, variation = key[0].split(':')
                        key = k0, key[1]
                    match key:
                        case ('p_pix', 'g_gcd') | ('g_gcd', 'p_pix'):
                            from hhg9.projections import PlatePixelGCD
                            prj = PlatePixelGCD(self)
                        case ('r_gcd', 'g_gcd') | ('g_gcd', 'r_gcd'):
                            from hhg9.projections import RGCD_GCD
                            prj = RGCD_GCD(self)
                        case ('c_ell', 'g_gcd') | ('g_gcd', 'c_ell'):
                            from hhg9.projections import EllipsoidGCD
                            prj = EllipsoidGCD(self)
                        case ('g_gcd', 'b_oct') | ('b_oct', 'g_gcd'):
                            from hhg9.projections import GCDBary
                            prj = GCDBary(self)
                        case ('c_oct', 'c_ell') | ('c_ell', 'c_oct'):
                            from hhg9.projections import AKOctahedralEllipsoid
                            prj = AKOctahedralEllipsoid(self)
                        case _:
                            raise ValueError(f'A projection {key} is not registered.')
                    coords = self._domain_projections[key][prj.name](coords)
                    continue
            else:
                # we could have alternatives, if there was a way of passing a key for it.
                alts = self._domain_projections[key]
                name = next(iter(alts))  # *currently* grab the first projection.
                if name == 'chain':
                    if (
                        isinstance(a, CompositeDomain) and isinstance(b, CompositeDomain)
                        and alts['chain'] == [a.name, b.name]
                    ):
                        a_components = a.components
                        b_components = b.components
                        ab = a_components.keys() & b_components.keys()
                        if len(ab) == len(a_components):
                            a2b = {a_components[k]: b_components[k] for k in ab}
                            coords = self._project_composites(coords, a, a2b)
                            coords.domain = b
                            continue
                        else:
                            raise ValueError(f"A projection {(a.name, b.name)} is not registered (component mismatch).")
                    else:  # non-composite..
                        sub_chain = self._domain_projections[key][name]
                        try:
                            sub_ch = self._check_chain(sub_chain)
                        except ValueError as err:
                            raise ValueError(
                                f"Degenerate chain registered for ({a.name}, {b.name}) at {str(err)} with no concrete projection."
                            )
                        coords = self.project(coords, sub_ch)
                else:
                    coords = self._domain_projections[key][name](coords)
        return coords
