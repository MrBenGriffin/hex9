# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Points are used as a generalised means of handling coordinates in a projection chain.
They are domain-associated. For example, while GCD (latitude/longitude) and Simplex(uv) coordinates
both are identical in the sense that they are both 2D, they are different in the sense that
GCD coordinates are associated with WGS84, whereas Simplex coordinates are associated
with a domain that is a triangle.

Each point carries an `oid` (octant id, uint8, 0-7) that identifies which octant it belongs to
for domains that use the octahedral decomposition.  255 is the sentinel for "unset / invalid".
`oid` is passed as metadata through domains that don't need it so later projections can
use it as a reliable octant indicator without re-binning.
"""
import numpy as np

OID_INVALID: np.uint8 = np.uint8(255)


class Points:
    """
    A domain-aware collection of coordinate positions.
    Each coordinate has a domain, and associated sample data.
    Each 'point' represents a location that may be approximate,
    depending on its Domain and formatting resolution.
    oid may be passed as a scalar uint8 (broadcast to all points) or
    as a (N,) uint8 array.
    """
    def __init__(self, coords: np.ndarray, domain=None, oid=None, samples=None):
        self.coords = coords
        self.domain = domain
        if oid is not None:
            oid = np.asarray(oid, dtype=np.uint8)
            if oid.ndim == 0:
                # scalar — broadcast to every point
                oid = np.full(len(self.coords), int(oid), dtype=np.uint8)
            elif oid.ndim == 1 and oid.shape[0] == len(self.coords):
                oid = np.ascontiguousarray(oid, dtype=np.uint8)
            else:
                raise ValueError(
                    f"Invalid oid shape: expected ({len(self.coords)},) or scalar, got {oid.shape}")
        self.oid = oid
        from hhg9.base.composite import CompositeDomain
        if self.oid is None and self.domain is not None and isinstance(self.domain, CompositeDomain):
            self.domain.binning(self)
        self.samples = samples

    def binning(self, sig: tuple = None):
        """Return points with oid set by composite domain binning."""
        if self.oid is None and self.domain is not None:
            from hhg9.base.composite import CompositeDomain
            if isinstance(self.domain, CompositeDomain):
                self.domain.binning(self)
        return self

    # ------------------------------------------------------------------
    # Octant id utilities (static — useful for working with H9O tables)
    # ------------------------------------------------------------------

    @staticmethod
    def calc_octant_ids(components) -> np.ndarray:
        """
        Convert (N, 3) int8 sign-component array to (N,) uint8 octant ids.
        Kept as a utility for code that works with H9O sign tables.
        """
        c = np.asarray(components)
        return (
            ((c[:, 2] < 0).astype(np.uint8) << 2) |
            ((c[:, 1] < 0).astype(np.uint8) << 1) |
            (c[:, 0] < 0).astype(np.uint8)
        )

    @staticmethod
    def invert_octant_ids(octants_) -> np.ndarray:
        """
        Convert (N,) uint8 octant ids back to (N, 3) int8 sign-component array.
        Kept as a utility for code that works with H9O sign tables.
        """
        octants = np.asarray(octants_, dtype=np.uint8)
        return 1 - 2 * np.stack([
            (octants >> 0) & 1,  # X
            (octants >> 1) & 1,  # Y
            (octants >> 2) & 1   # Z
        ], axis=-1).astype(np.int8)

    def oids(self) -> np.ndarray:
        """Return the octant id array (alias for self.oid)."""
        return self.oid

    @classmethod
    def class_mode(cls, oid):
        """Given (N,) uint8 oid array, return (oid, net_mode) where net_mode = popcount(oid) % 2."""
        oid = np.asarray(oid, dtype=np.uint8)
        bits = np.bitwise_count(oid)
        mode = np.ascontiguousarray(bits % 2, dtype=np.uint8)
        return np.ascontiguousarray(oid, dtype=np.uint8), mode

    def cm(self):
        """Return (oid_array, mode_array), or (None, None) if oid is not set."""
        if self.oid is not None:
            return self.class_mode(self.oid)
        return None, None

    def bbox(self, *, return_diff=False, trbl=False):
        """Return bounding box of points"""
        if self.coords is not None:
            if self.domain.name in ['g_gcd']:
                (y_min, x_min), (y_max, x_max) = np.min(self.coords, axis=0), np.max(self.coords, axis=0)
            elif self.domain.name in ['c_ell', 'c_oct']:
                p_min, p_max = np.min(self.coords, axis=0), np.max(self.coords, axis=0)
                return p_min, p_max
            else:
                (x_min, y_min), (x_max, y_max) = np.min(self.coords, axis=0), np.max(self.coords, axis=0)
            bbox = (x_min, y_min, x_max, y_max) if not trbl else (y_max, x_max, y_min, x_min)
            if not return_diff:
                return bbox
            else:
                return bbox, (x_max - x_min, y_max - y_min)
        return None

    def __getitem__(self, idx):
        """Constructs new Points instance from single indexed element"""
        if isinstance(idx, tuple):
            raise TypeError(
                f"2D indexing like Points[{idx}] is not supported.\n"
                "→ Use eg `pts.coords[...]` instead if you need NumPy-style slicing."
            )
        coords = np.array([self.coords[idx]])
        domain = self.domain
        oid = np.array([self.oid[idx]]) if self.oid is not None and idx < len(self.oid) else None
        samples = np.array([self.samples[idx]]) if self.samples is not None and idx < len(self.samples) else None
        return Points(coords, domain, oid, samples)

    def __len__(self):
        return len(self.coords)

    def __format__(self, format_spec):
        """Allow f-string formatting."""
        if self.coords is None or len(self.coords) == 0:
            return ''
        if self.domain is None:
            return self.coords.__format__(format_spec)
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
        oid_str = f'{self.oid.shape}' if self.oid is not None else 'None'
        smp_str = f'{self.samples.shape}' if self.samples is not None else 'None'
        pts_str = f'{self.coords.shape}' if self.coords is not None else 'None'
        return f"Points(coords={pts_str}, domain={self.domain}, oid={oid_str}, samples={smp_str})"

    def copy(self):
        """Copy points"""
        return Points(
            coords=self.coords.copy(),
            domain=self.domain,
            oid=self.oid.copy() if self.oid is not None else None,
            samples=self.samples.copy() if self.samples is not None else None
        )

    @classmethod
    def concat(cls, points_list):
        """Concatenate multiple Points instances into one."""
        if not points_list:
            raise ValueError('No points provided')

        for p in points_list:
            if not isinstance(p, cls):
                raise TypeError(f"Expected Points, got {type(p)}")

        domains = {id(p.domain) for p in points_list}
        if len(domains) > 1:
            raise ValueError("Cannot concatenate Points with different domains")

        domain = points_list[0].domain
        coords = np.concatenate([p.coords for p in points_list], axis=0)

        has_oid = any(p.oid is not None for p in points_list)
        if has_oid:
            oid = np.concatenate([
                p.oid if p.oid is not None else np.full(len(p.coords), OID_INVALID, dtype=np.uint8)
                for p in points_list
            ])
        else:
            oid = None

        has_samples = any(p.samples is not None for p in points_list)
        if has_samples:
            samples = np.concatenate([
                p.samples if p.samples is not None else np.zeros(len(p.coords), dtype=int)
                for p in points_list
            ])
        else:
            samples = None

        return cls(coords, domain=domain, oid=oid, samples=samples)

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
            y = (h - 1) - yy.astype(np.uint64)
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
        sel_oid = self.oid[good] if self.oid is not None else None
        smp_c = self.samples[good] if self.samples is not None else None
        return Points(self.coords[good], domain=self.domain, oid=sel_oid, samples=smp_c)
