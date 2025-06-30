"""
Part of the H9 project
"""
from typing import Sequence
import numpy as np

# from . import ComponentDomain
# from .domain import Domain


class Points:
    """
    A domain-aware collection of coordinate positions.
    Each coordinate has a domain, and associated sample data.
    Each 'point' represents a location that may be approximate,
    depending on its Domain and formatting resolution.
    """
    def __init__(self, coords: np.ndarray, domain=None, components=None, samples=None):
        self.coords = coords
        self.domain = domain  # This is the composite domain, if the
        self.components = components
        self.samples = samples

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            raise TypeError(
                f"2D indexing like Points[{idx}] is not supported.\n"
                "→ Use eg `pts.coords[...]` instead if you need NumPy-style slicing."
            )
        coords = self.coords[idx]
        domain = self.domain
        components = self.components[idx] if self.components is not None and idx < len(self.components) else None
        samples = self.samples[idx] if self.samples is not None and idx < len(self.samples) else None
        return Points(coords, domain, components, samples)

    def __len__(self):
        return len(self.coords)

    def __format__(self, format_spec):
        """Allow f-string formatting."""
        if self.coords is None or len(self.coords) == 0:
            return ''

        if self.domain is None:
            return self.coords.__format__(format_spec)
        # Identify the format and subtype or length.
        main_sub = format_spec.split('.')
        name = main_sub[0]
        sub = main_sub[1] if len(main_sub) > 1 else ''
        # Handle formatting a single row or multiple
        is_scalar = self.coords.ndim == 1 or self.coords.shape[0] == 1
        if is_scalar:
            pt = self.coords[0] if self.coords is not None and self.coords.shape[0] == 1 else self.coords
            cp = self.components[0] if self.components is not None and self.components.shape[0] == 1 else self.components
            dom = self.domain if self.components is None else self.domain.components[tuple(cp)]
            if name not in dom.address_formats:
                return self.coords.__format__(format_spec)
            formatter = dom.address_formats[name]
            return formatter.format(pt, dom, sub)
        else:
            out = []
            for i, coord in enumerate(self.coords):
                dom = self.domain
                if self.components is not None:
                    dom = self.domain.components[tuple(self.components[i])]
                if name not in dom.address_formats:
                    out.append(coord.__format__(format_spec))
                else:
                    formatter = dom.address_formats[name]
                    out.append(formatter.format(coord, dom, sub))
            if len(out) == 1:
                return out[0]
            return '\n'.join(out)

    def __repr__(self):
        keys = ', '.join(self.samples.keys())
        return f"Points(coords={self.coords.shape}, samples=[{keys}])"

    def copy(self):
        """Copy points"""
        return Points(
            coords=self.coords.copy(),  # Defensive deep copy
            domain=self.domain,  # Immutable or shared as needed
            components=self.components.copy() if self.components is not None else None,  # Safe if immutable or reference-shared
            samples=self.samples.copy() if self.samples is not None else None
        )

    @classmethod
    def concat(cls, points_list):
        """Concatenate multiple Points instances into one."""
        if not points_list:
            raise ValueError('No points provided')

        # Check all are Points
        for p in points_list:
            if not isinstance(p, cls):
                raise TypeError(f"Expected Points, got {type(p)}")

        # Check all share the same domain
        domains = {id(p.domain) for p in points_list}
        if len(domains) > 1:
            raise ValueError("Cannot concatenate Points with different domains")

        domain = points_list[0].domain

        # Concatenate coords
        coords = np.concatenate([p.coords for p in points_list], axis=0)

        # Concatenate components if present
        has_components = any(p.components is not None for p in points_list)
        if has_components:
            components = np.concatenate([
                p.components if p.components is not None else np.zeros(len(p.coords), dtype=int)
                for p in points_list
            ])
        else:
            components = None

        has_samples = any(p.samples is not None for p in points_list)
        if has_samples:
            samples = np.concatenate([
                p.samples if p.samples is not None else np.zeros(len(p.coords), dtype=int)
                for p in points_list
            ])
        else:
            samples = None

        return cls(coords, domain=domain, components=components, samples=samples)

    def image(self, dim, flip=True):
        """
        return the image that these points represent.
        """
        xs, ys = self.coords[:, 0], self.coords[:, 1]
        w, h = dim
        x0 = np.min(xs)
        y0 = np.min(ys)
        y_adj = (h-1e-6)/(np.max(ys)-y0)
        x_adj = (w-1e-6)/(np.max(xs)-x0)
        yy = np.floor(y_adj*(ys-y0)).astype(np.uint64)
        xx = np.floor(x_adj*(xs-x0)).astype(np.uint64)
        ch = self.samples
        if flip:
            y = (h - 1) - yy.astype(np.uint64)  # still in cartesian (ie, 0 is bottom left).
        else:
            y = yy.astype(np.uint64)
        x = xx.astype(np.uint64)
        channels = 1 if ch.ndim == 1 else ch.shape[1]
        ch = ch.reshape(-1, channels)
        img = np.ones((h, w, channels), dtype=ch.dtype)
        img[y, x] = ch
        return img
