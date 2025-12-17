# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Points are used as a generalised means of handling coordinates in a projection chain.
They are domain-associated. For example, while GCD (latitude/longitude) and Simplex(uv) coordinates
both are identical in the sense that they are both 2D, they are different in the sense that
GCD coordinates are associated with WGS84, whereas Simplex coordinates are associated
with a domain that is a triangle.
They have a `component` array which indicates which octant each point is in, for those domains that use them.
The components are passed as metadata for those domains that don't need them - but can be then used by later projections
as a reliable octant indicator.
"""
import numpy as np


class Points:
    """
    A domain-aware collection of coordinate positions.
    Each coordinate has a domain, and associated sample data.
    Each 'point' represents a location that may be approximate,
    depending on its Domain and formatting resolution.
    component may be passed as a template for all the coords.
    """
    def __init__(self, coords: np.ndarray, domain=None, components=None, samples=None):
        self.coords = coords
        self.domain = domain  # This is the composite domain, if the
        if components is not None:
            components = np.asarray(components)
            if components.ndim == 1:
                if components.shape[0] != 3:
                    components = self.invert_octant_ids(components)
                # Broadcast single 3-element tuple to match coords
                else:
                    components = np.broadcast_to(components, (self.coords.shape[0], 3))
            elif components.shape != (self.coords.shape[0], 3):
                raise ValueError(
                    f"Invalid component shape: expected {(self.coords.shape[0], 3)} or (3,), got {components.shape}")
        self.components = components
        if self.components is None and self.coords.shape[1] == 3:
            self.binning()
        self.samples = samples

    def binning(self, sig: tuple = None):
        """Return points with domain set by composite set_domain"""
        # caller('Points: binning')
        if self.components is None:
            c = self.coords.copy()  # don't mess with the coords.
            c[c == 0] = np.finfo(self.coords.dtype).tiny
            self.components = np.atleast_2d(np.sign(c).astype(np.int8))
        return self

    @classmethod
    def calc_octant_ids(cls, components):
        """Definitive utility to calculate octant IDs from sign components."""
        return ((components[:, 2] < 0) << 2) | \
               ((components[:, 1] < 0) << 1) | \
               ((components[:, 0] < 0) << 0).astype(np.uint8)

    @classmethod
    def invert_octant_ids(cls, octants_):
        """
        Given octant IDs (0..7), return array of components
        """
        octants = np.asarray(octants_, dtype=np.uint8)
        return 1 - 2 * np.stack([
            (octants >> 0) & 1,  # X
            (octants >> 1) & 1,  # Y
            (octants >> 2) & 1   # Z
        ], axis=-1).astype(np.int8)
        # return signs.astype(np.int8)  # Extract bits and map {0,1} → {1,-1} using 1 - 2*b

    # @classmethod
    # def mode(self, cmp):
    #     """
    #     :param cmp:
    #     :return mode based on component.
    #     """
    #     side = self.calc_octant_ids(cmp).astype(np.uint8)
    #     bits = np.bitwise_count(side)
    #     return np.array(bits % 2, dtype=np.uint8)
    @classmethod
    def class_mode(cls, cmp):
        """Given array of components, return octant IDs and mode."""
        cmp = np.atleast_2d(cmp)
        side = cls.calc_octant_ids(cmp).astype(np.uint8)
        bits = np.bitwise_count(side)
        mode = np.ascontiguousarray(bits % 2, dtype=np.uint8)
        side = np.ascontiguousarray(side, dtype=np.uint8)
        return side, mode

    def cm(self):
        """
            Shortened variation of component with mode.
        """
        if self.components is not None:
            return self.class_mode(self.components)
        return None, None

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            raise TypeError(
                f"2D indexing like Points[{idx}] is not supported.\n"
                "→ Use eg `pts.coords[...]` instead if you need NumPy-style slicing."
            )
        coords = np.array([self.coords[idx]])
        domain = self.domain
        components = np.array([self.components[idx]]) if self.components is not None and idx < len(self.components) else None
        samples = np.array([self.samples[idx]]) if self.samples is not None and idx < len(self.samples) else None
        return Points(coords, domain, components, samples)

    def __len__(self):
        return len(self.coords)

    # def _scalar(self, name, format_spec, idx=0):
    #     pt = np.atleast_2d(self.coords)[idx]
    #     cp = np.atleast_2d(self.components)[idx]
    #     dom = self.domain
    #     composite = dom.components[tuple(cp)] if dom is not None and cp is not None else None
    #     if name not in dom.address_formats:
    #         return self.coords.__format__(format_spec)
    #     formatter = dom.address_formats[name]
    #     return formatter.format(pt, composite, format_spec)

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
        if name not in self.domain.address_formats:
            formatter = self.domain.__getattribute__(name)
            if formatter is None:
                return self.coords[0].__format__(format_spec)
            else:
                return formatter.format(self, None, sub)
        formatter = self.domain.address_formats[name]
        return formatter.format(self, None, sub)

    def __repr__(self):
        cmp_str = f'{self.components.shape}' if self.components is not None else 'None'
        smp_str = f'{self.samples.shape}' if self.samples is not None else 'None'
        pts_str = f'{self.coords.shape}' if self.coords is not None else 'None'
        return f"Points(coords={pts_str}, domain={self.domain}, components={cmp_str}, samples={smp_str})"

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

    def select(self, good: np.ndarray):
        """Return only those for which good is true"""
        if len(self.coords) != len(good):
            raise ValueError('Number of coordinates does not match boolean array')
        sel_c = self.components[good] if self.components is not None else None
        smp_c = self.samples[good] if self.samples is not None else None
        return Points(self.coords[good], domain=self.domain, components=sel_c, samples=smp_c)
