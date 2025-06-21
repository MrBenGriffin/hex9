"""
Part of the H9 project
"""
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from .domain import Domain
from .points import Points
from .composite import ComponentDomain, CompositeDomain
from .projection import Projection


class Registrar:
    """
    Registrar manages the registers of
    • coordinate sets (as classes) (with their address formats)
    • projections (as classes)
    """

    def __init__(self):
        self._domains = {}
        self._projections = {}

    def register_projection(self, obj, chain: NDArray):
        """Register a projection. Normally managed in the Projection init class."""
        if isinstance(obj, Projection):
            fwd, bak = tuple(chain), tuple(reversed(chain))
            if fwd not in self._projections:
                self._projections[fwd] = {}
            self._projections[fwd][obj.name] = obj.forward
            if bak not in self._projections:
                self._projections[bak] = {}
            self._projections[bak][obj.name] = obj.backward
        elif isinstance(obj, str) and obj == 'chain':
            for csys in chain:
                if isinstance(csys, Domain):
                    if csys.name not in self._domains:
                        self.register_domain(csys)
                elif isinstance(csys, str):
                    if csys not in self._domains:
                        raise ValueError(f"{csys} Unregistered coordinate set")
            ch = [val if isinstance(val, str) else val.name for val in chain]
            fwd, bak = tuple([ch[0], ch[-1]]), tuple([ch[-1], ch[0]])
            if fwd not in self._projections:
                self._projections[fwd] = {}
            self._projections[fwd]['chain'] = ch
            if bak not in self._projections:
                self._projections[bak] = {}
            self._projections[bak]['chain'] = list(reversed(ch))

    def register_domain(self, dom: Domain):
        """Register set_domain."""
        self._domains[dom.name] = dom

    def domain(self, key):
        """return domain by key"""
        if key not in self._domains:
            raise KeyError(key)
        return self._domains[key]

    def register_domain_alias(self, key: str, alias: str):
        """Register an alias for a given Domain."""
        self._domains[alias] = self._domains[key]

    def _cmp_key(self, a, b, a_bins, b_bins):
        key = None
        if a_bins and isinstance(b, ComponentDomain):
            sign = b.sig()
            if sign not in a_bins:
                raise ValueError(f'b {sign} not in a bins')
            else:
                key = a_bins[sign], b.name
                if key not in self._projections:
                    raise ValueError(f'chain {key} not found')
        elif b_bins and isinstance(a, ComponentDomain):
            sign = a.sig()
            if sign not in b_bins:
                raise ValueError(f'a {sign} not in b bins')
            else:
                key = a.name, b_bins[sign]
                if key not in self._projections:
                    raise ValueError(f'chain {key} not found')
        return key

    def _check_chain(self, chain):
        for dom in chain:
            if isinstance(dom, Domain):
                if dom.name not in self._domains:
                    self.register_domain(dom)
            else:
                raise ValueError(f'chain {dom} Unregistered Domain')

    def _project_composites(self, pts: Points, a, a2b):
        if pts.components is None:
            pts = a.binning(pts)
        # pts.coords = np.atleast_2d(pts.coords)
        res = np.zeros_like(pts.coords)
        uvw = (pts.components >= 0) @ (4, 2, 1)
        for sig, cmp in a.components.items():
            key = (cmp.name, a2b[cmp].name)
            facilitator = next(iter(self._projections[key]))
            side = np.asarray(sig, dtype='b')
            ref = (side >= 0) @ (4, 2, 1)
            crds = pts.coords[uvw == ref]  # these are the coordinates for this projection.
            if crds.size > 0:
                rex = self._projections[key][facilitator](crds)
                if rex.shape[-1] != res.shape[-1]:
                    res = np.zeros([pts.coords.shape[0], rex.shape[-1]])
                res[uvw == ref] = rex
        return Points(res, samples=pts.samples, components=pts.components)

    def project(self, coords: Points, chain: NDArray) -> Points:
        """Transform coordinates from one set to another."""
        self._check_chain(chain)
        for (a, b) in pairwise(chain):
            key = a.name, b.name
            if key not in self._projections:
                a_components = a.components if isinstance(a, CompositeDomain) else None
                b_components = b.components if isinstance(b, CompositeDomain) else None
                if not (a_components and b_components):
                    key = self._cmp_key(a, b, a_components, b_components)
                    if key is None:
                        raise ValueError(f'The projection for ({a} {b}) is not registered.')
                else:
                    ab = a_components.keys() & b_components.keys()
                    if len(ab) == len(a_components):
                        a2b = {a_components[k]: b_components[k] for k in ab}
                        coords = self._project_composites(coords, a, a2b)
                        coords.domain = b
                    else:
                        raise ValueError(f'A projection {key} is not registered.')
            else:
                # we could have alternatives, if there was a way of passing a key for it.
                alts = self._projections[key]
                name = next(iter(alts))  # *currently*  grab the first projection.
                if name == 'chain':
                    sub_ch = [self._domains[k] for k in self._projections[key][name]]
                    coords = self.project(coords, sub_ch)
                else:
                    coords = self._projections[key][name](coords)
        return coords
