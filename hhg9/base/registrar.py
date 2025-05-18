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

    def _project_composites(self, pts, a, ab, a2b):
        bundle = []
        if isinstance(pts, tuple):
            for group in pts:
                bundle.append(self._project_composites(group, a, ab, a2b))
            return tuple(bundle)
        if isinstance(pts, Points) and pts.dom.name in a2b.keys():
            key = (pts.dom.name, a2b[pts.dom.name])
            name = next(iter(self._projections[key]))
            return self._projections[key][name](pts)
        repo = a.binning(pts)  # generate the tuple.
        for bin_pts in repo:
            inc = bin_pts.domain()
            key = (inc, a2b[inc])
            alts = self._projections[key]
            name = next(iter(alts))  # *currently*  grab the first projection.
            if name == 'chain':
                bundle.extend([self._domains[k] for k in self._projections[key][name]])
            else:
                bundle.extend([self._projections[key][name](bin_pts)])
        return tuple(bundle)

    def project(self, coords: Points, chain: NDArray) -> Points:
        """Transform coordinates from one set to another."""
        self._check_chain(chain)
        for (a, b) in pairwise(chain):
            key = a.name, b.name
            if key not in self._projections:
                a_bins = a.bins() if isinstance(a, CompositeDomain) else None
                b_bins = b.bins() if isinstance(b, CompositeDomain) else None
                if not (a_bins and b_bins):
                    key = self._cmp_key(a, b, a_bins, b_bins)
                else:
                    ab = a_bins.keys() & b_bins.keys()
                    if len(ab) == len(a_bins):
                        a2b = {a_bins[k]: b_bins[k] for k in ab}
                        coords = self._project_composites(coords, a, ab, a2b)
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
        if isinstance(coords, tuple) and len(coords) == 1:
            return coords[0]
        return coords
