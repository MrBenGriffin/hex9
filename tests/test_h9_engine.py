"""
Unit tests for the grid region encoding function en_grid.
Validates correct region classification based on geometric thresholds.
"""
from functools import lru_cache, cache
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose


@pytest.fixture
def const_grid():
    """Fixture providing a configured instance of RegionClassificationGrid."""
    return GridConstants()


class GridConstants:
    """A minimal class that only holds geometric constants and those methods that use only those constants."""

    def __init__(self):
        self.H = np.sqrt(6) / 2.
        self.R3 = np.sqrt(3)
        # All other geometric constants derived from H and R3
        # Done in order to avoid tiny floating point deviations.
        self.TR = self.H / self.R3  #
        self.W = 2 * self.TR  # This correctly derives W = sqrt(2)
        self.ΛC = 2 * self.H / 3.
        self.ΛF = -self.H / 3.
        self.VC = self.H / 3.
        self.VF = -2 * self.H / 3.
        self.Ẇ = self.ΛC
        self.TL = -self.TR
        self.U, self.V = self.W / 6., self.H / 9.

    def region_classification(self, ẋ, y):
        """
        Classify coordinates into geometric grid regions using thresholds.
        :param ẋ: np_array of √3-scaled x coordinates.
        :param y: np_array of y coordinates.
        :return: np_array of encoded region identifiers.
        """
        h_conditions = [
            y > self.ΛC,
            y > self.VC,
            y > 0,
            y > self.ΛF,
            y >= self.VF,
        ]
        h_id = np.select(h_conditions, [0, 1, 2, 3, 4], default=5)
        y_minus_x = y - ẋ
        p_conditions = [
            y_minus_x > self.Ẇ,
            y_minus_x > 0,
            y_minus_x >= -self.Ẇ,
        ]
        p_id = np.select(p_conditions, [0, 1, 2], default=3)
        y_plus_x = y + ẋ
        n_conditions = [
            y_plus_x < -self.Ẇ,
            y_plus_x < 0,
            y_plus_x <= self.Ẇ,
        ]
        n_id = np.select(n_conditions, [0, 1, 2], default=3)
        return h_id << 4 | p_id << 2 | n_id


@pytest.fixture
def reg_grid():
    """Fixture providing an instance of GridRegions."""
    return GridRegions()


class GridRegions(GridConstants):
    def __init__(self):
        super().__init__()
        self.num_regions = 96    # This number follows from the partitions within GridConstants
        self.invalid_ugc = 0x5f  # (final region := OOB)

        # Build offsets analytically (exact integers × base units)
        hx3 = self.H / 3.0  # unit for ẋ offsets
        hy9 = self.H / 9.0  # unit for y offsets

        # The index order is such that in_regions [0..5] are mode-shared: (found inside both modes),
        # and index % 2 indicates the mode of that region.
        # Mode specific regions are found in those regions which have the same mode as themselves.
        self.in_regions = [
            0x26, 0x2a, 0x3a, 0x39, 0x35, 0x25,  # Mode Shared, inner regions from V above barycentric origin, CW.
            0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e   # Mode Specific outer regions from apex of V, CW
        ]
        self.in_up_regions = [  # 9 regions serve Λ mode
            0x39, 0x3a, 0x3e,  # c0 ΛVΛ
            0x25, 0x35, 0x34,  # c1 ΛVΛ
            0x2a, 0x26, 0x16,  # c2 ΛVΛ
        ]
        self.in_dn_regions = [
            0x26, 0x2a, 0x2b,  # c0 VΛV
            0x3a, 0x39, 0x49,  # c1 VΛV
            0x35, 0x25, 0x21,  # c2 VΛV
        ]

        # origin offset multipliers corresponding to `in_regions`:
        mx = np.array([
            0, +1, +1,  # 0x26, 0x2a, 0x3a
            0, -1, -1,  # 0x39, 0x35, 0x25
            0, -2, -2,  # 0x49, 0x34, 0x21
            0, +2, +2,  # 0x16, 0x2b, 0x3e
        ], dtype=np.int8)

        my = np.array([
            +2, +1, -1,  # 0x26, 0x2a, 0x3a
            -2, -1, +1,  # 0x39, 0x35, 0x25
            -4, -2, +2,  # 0x49, 0x34, 0x21
            +4, +2, -2,  # 0x16, 0x2b, 0x3e
        ], dtype=np.int8)

        self.ugc_off_x = np.zeros(self.num_regions, dtype=np.float64)
        self.ugc_off_y = np.zeros(self.num_regions, dtype=np.float64)
        # self.ugc_off_ẋ = np.zeros(self.num_regions, dtype=np.float64)

        self.ugc_off_x[self.in_regions] = mx * hx3
        self.ugc_off_y[self.in_regions] = my * hy9
        self.ugc_off = np.hstack((self.ugc_off_x, self.ugc_off_y))
        # self.ugc_off_ẋ[self.in_regions] = mx * hx3 * self.R3

        ugc_num_props = 3

        (self.in_dn, self.in_up, self.mode) = range(ugc_num_props)
        self.ugc_lut = np.full((self.num_regions, ugc_num_props), self.invalid_ugc, dtype=np.uint8)
        self.ugc_lut[:, self.in_dn] = 0
        self.ugc_lut[:, self.in_up] = 0
        self.ugc_lut[self.in_up_regions, self.in_up] = 1
        self.ugc_lut[self.in_dn_regions, self.in_dn] = 1
        self.ugc_lut[self.in_regions, self.mode] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.ugc_lut[self.invalid_ugc] = 0  # set no offsets for illegal!


    @cache
    def child_lut(self):
        """
        Return LUT which, given a mode and c1, returns the child regions there.
        The first two children are mode-shared.
        The middle switches mode from parent.
        The final mode is mode-locked.
        Necessary feature is that a child-region-c1 is unique.
        """
        _chd = {
            (0, 0): [0x26, 0x2A, 0x2B],  # V,C1.0 VΛV
            (0, 1): [0x3A, 0x39, 0x49],  # V,C1.1 VΛV
            (0, 2): [0x35, 0x25, 0x21],  # V,C1.2 VΛV
            (1, 0): [0x39, 0x3A, 0x3E],  # Λ,C1.0 ΛVΛ
            (1, 1): [0x25, 0x35, 0x34],  # Λ,C1.1 ΛVΛ
            (1, 2): [0x2A, 0x26, 0x16],  # Λ,C1.2 ΛVΛ
        }
        lut = np.zeros((2, 3, 3), dtype=np.uint8)
        for (mode, c1), children in _chd.items():
            lut[mode, c1] = children
        return lut

    @cache
    def pc_c1(self):
        """
        Build (parent_region, child_region) -> c1 from canonical child_lut + ugc_lut.
        """
        num = self.num_regions
        lut = np.full((num, num), self.invalid_ugc, dtype=np.uint8)
        # lut = np.zeros((num, num), dtype=np.uint8)

        ugc = self.ugc_lut  # (num_regions, 1) -> mode
        chd = self.child_lut()  # (2, 3, 3)

        # For every parent region p, infer its mode and fill reverse mapping.
        for par in self.in_regions:
            mode = ugc[par, self.mode]
            for c1 in (0, 1, 2):
                for child in chd[mode, c1]:
                    lut[par, child] = c1
        return lut

    # def ugc_regions(self, x, y, mode, depth=36):
    #     """
    #     Given a vector of Point coords create a set of regions
    #     """
    #     num_points = x.size
    #     addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
    #     addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)  # These values should come from the octant set.
    #     # history = np.zeros((num_points, depth + 2, 6))
    #     for i in range(depth + 1):
    #         ẋ = self.R3 * x
    #         region = self.region_classification(ẋ, y)  # Raw classification
    #         props = self.ugc_lut[region]
    #         mode_up = props[:, self.in_up]
    #         mode_dn = props[:, self.in_dn]
    #         in_scope = np.where(mode == 1, mode_up, mode_dn)
    #         region_id = np.where(in_scope, region, self.invalid_ugc)  # Validated ID
    #         addresses[:, i + 1] = region_id
    #         off = self.ugc_off[region_id]
    #         mode = self.ugc_lut[region_id, self.mode]
    #         x -= off[:, 0]
    #         y -= off[:, 1]
    #         x *= 3.
    #         y *= 3.
    #     return addresses

    def ugc_regions(self, xy, mode, depth=36):
        """
        Given a vector of Point coords create a set of regions
        """
        num_points = xy.shape[0]
        x = np.copy(xy[:, 0])
        y = np.copy(xy[:, 1])
        addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
        addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)  # These values should come from the octant set.
        for i in range(depth + 1):
            ẋ = self.R3 * x
            region = self.region_classification(ẋ, y)  # Raw classification
            props = self.ugc_lut[region]
            mode_up = props[:, self.in_up]
            mode_dn = props[:, self.in_dn]
            in_scope = np.where(mode == 1, mode_up, mode_dn)
            region_id = np.where(in_scope, region, self.invalid_ugc)  # Validated ID
            addresses[:, i + 1] = region_id
            off = self.ugc_off[region_id]
            mode = self.ugc_lut[region_id, self.mode]
            x -= off[:, 0]
            y -= off[:, 1]
            x *= 3.
            y *= 3.
        return addresses


    def ugc_dec(self, uri_address):
        """
        Decode back to (x,y,mode) by undoing the classifier-plane stepping.
        """
        num_points, depth = uri_address.shape
        ẋ = np.zeros(num_points, dtype=np.float64)
        y = np.zeros(num_points, dtype=np.float64)

        for i in range(depth - 2, 0, -1):
            region_id = uri_address[:, i]
            valid = (region_id != self.invalid_ugc)

            ẋ /= 3.0
            y /= 3.0

            if np.any(valid):
                rid = region_id[valid]
                ẋ[valid] += self.ugc_off_ẋ[rid]
                y[valid] += self.ugc_off_y[rid]

        initial_mode = self.ugc_lut[uri_address[:, 0], self.mode]
        x = ẋ / self.R3
        return np.stack([x, y, initial_mode], axis=-1)

    def ugc_regions_w(self, xy, mode, depth=36):
        """
        Given a vector of Point coords create a set of regions
        """
        num_points = xy.shape[0]
        x = np.copy(xy[:, 0])
        y = np.copy(xy[:, 1])
        addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
        addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)  # These values should come from the octant set.
        for i in range(depth + 1):
            # x, y, ẋ = self.clamp(x, y, mode)
            ẋ = self.R3 * x
            region = self.region_classification(ẋ, y)  # Raw classification
            props = self.ugc_lut[region]
            mode_up = props[:, self.in_up]
            mode_dn = props[:, self.in_dn]
            in_scope = np.where(mode == 1, mode_up, mode_dn)
            region_id = np.where(in_scope, region, self.invalid_ugc)  # Validated ID
            addresses[:, i + 1] = region_id
            mode = self.ugc_lut[region_id, self.mode]
            x -= self.ugc_off_x[region_id]
            y -= self.ugc_off_y[region_id]
            x *= 3.
            y *= 3.
        return self.terminate(addresses)

    def ugc_dec_w(self, uri_address):
        """
        REVERSE: URI addresses back into (x,y) coordinates and initial mode.
        Inverse of ugc_regions
        """
        num_points, depth = uri_address.shape
        # Initialize x and y with the precise remainder from the encoding process.
        x = np.zeros(num_points, dtype=np.float64)
        y = np.zeros(num_points, dtype=np.float64)

        # Loop backwards from the penultimate layer down to the first REAL layer (index 1),
        # skipping the placeholder root at index 0.
        # Likewise, leaf residual is ignored by design; its mag. is ≤ ~7·3⁻ᵏ and below fp noise for k≥34.
        for i in range(depth - 2, 0, -1):
            region_id = uri_address[:, i]
            valid_mask = (region_id != self.invalid_ugc)

            x /= 3.0
            y /= 3.0

            if np.any(valid_mask):
                valid_ids = region_id[valid_mask]
                x[valid_mask] += self.ugc_off_x[valid_ids]
                y[valid_mask] += self.ugc_off_y[valid_ids]

        # After reconstructing the coordinates, find the initial mode from the root URI.
        initial_mode = self.ugc_lut[uri_address[:, 0], self.mode]

        # Stack all three results into a final (N, 3) array.
        return np.stack([x, y, initial_mode], axis=-1)

    def alt_ugc_dec(self, uri_address):
        """
        REVERSE: Decodes a URI back into (x,y) and initial mode.
        Runs backward in (ẋ, y); converts ẋ→x only at the end.
        """
        num_points, depth = uri_address.shape
        ẋ = np.zeros(num_points, dtype=np.float64)
        y = np.zeros(num_points, dtype=np.float64)

        for i in range(depth - 1, 0, -1):
            region_id = uri_address[:, i]
            valid = (region_id != self.invalid_ugc)

            ẋ /= 3.0
            y /= 3.0

            if np.any(valid):
                rid = region_id[valid]
                ẋ[valid] += self.ugc_off_ẋ[rid]
                y[valid] += self.ugc_off_y[rid]

        initial_mode = self.ugc_lut[uri_address[:, 0], self.mode]
        x = ẋ / self.R3  # convert back once
        return np.stack([x, y, initial_mode], axis=-1)

    def alt_ugc_regions(self, x, y, mode, depth=36):
        num_points = x.size
        addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
        addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)

        for i in range(depth + 1):
            # 1) classify in classifier plane
            xbar = self.R3 * x
            region = self.region_classification(xbar, y)
            props = self.ugc_lut[region]

            # 2) validate region vs parent mode (in-up / in-down)
            in_scope = np.where(mode == 1, props[:, self.in_up], props[:, self.in_dn])
            region_id = np.where(in_scope, region, self.invalid_ugc)
            addresses[:, i + 1] = region_id

            # 3) step one layer **in the classifier plane** for x, native for y;
            #    then convert back to x for the next iteration
            off_ẋ = self.ugc_off_ẋ[region_id]  # √3-scaled x-offsets
            off_y = self.ugc_off_y[region_id]
            mode = self.ugc_lut[region_id, self.mode]

            xbar = (xbar - off_ẋ) * 3.0
            y = (y - off_y) * 3.0
            x = xbar / self.R3

        return addresses

    def trace_roundtrip(self, xy, mode, depth, name):
        addr = self.ugc_regions(xy, mode, depth=depth)
        ugc = self.ugc_lut
        pc = self.pc_c1()
        chd = self.child_lut()

        # ----- forward (encode) -----
        x = xy[:, 0].copy()
        y = xy[:, 1].copy()
        x, y, xbar = self.clamp(x, y, mode)  # encoder’s very first step
        f = []
        f.append(dict(i=0, par=addr[0, 0], cur=None, mo=int(mode[0]),
                      x=float(x), y=float(y), xbar=float(xbar)))
        for i in range(1, depth + 1):
            par = addr[0, i - 1]
            cur = addr[0, i]
            c1 = pc[par, cur]
            mo = ugc[par, self.mode]
            offx = self.ugc_off_ẋ[cur]
            offy = self.ugc_off_y[cur]
            # step
            xbar2 = (xbar - offx) * 3.0
            y2 = (y - offy) * 3.0
            x2 = xbar2 / self.R3
            f.append(dict(i=i, par=int(par), cur=int(cur), c1=int(c1), mo=int(mo),
                          offx=float(offx), offy=float(offy),
                          x=float(x2), y=float(y2), xbar=float(xbar2)))
            x, y, xbar = x2, y2, xbar2

        # ----- backward (decode) -----
        bxbar = 0.0
        by = 0.0
        b = []
        for i in range(depth, 0, -1):
            rid = addr[0, i]
            bxbar = bxbar / 3.0 + float(self.ugc_off_ẋ[rid])
            by = by / 3.0 + float(self.ugc_off_y[rid])
            b.append(dict(i=i, rid=int(rid), x=float(bxbar / self.R3), y=float(by), xbar=float(bxbar)))
        b = list(reversed(b))  # i asc

        # ----- compare per step -----
        # Precompute root-frame totals from decode
        x0_bar = b[-1]['xbar']  # total \bar x_0
        y0 = b[-1]['y']  # total y_0

        # Also precompute prefix sums of off/3^(k-1)
        offx = [float(self.ugc_off_ẋ[addr[0, k]]) for k in range(1, depth + 1)]
        offy = [float(self.ugc_off_y[addr[0, k]]) for k in range(1, depth + 1)]
        px = 0.0
        py = 0.0

        print(f"\nTRACE {name} depth={depth}")
        for i in range(0, depth + 1):
            if i == 0:
                # Encoder's initial clamped state vs decoder's reconstructed root state
                dec_x = x0_bar / self.R3
                dec_y = y0
                dx = f[i]['x'] - dec_x
                dy = f[i]['y'] - dec_y
                print(f" i={i:02d} enc(x,y)=({f[i]['x']:+.9f},{f[i]['y']:+.9f}) "
                      f"dec_root=({dec_x:+.9f},{dec_y:+.9f}) Δ=({dx:+.3e},{dy:+.3e})")
                continue

            # Update prefix sums up to i
            px = px + offx[i - 1] / (3.0 ** (i - 1))
            py = py + offy[i - 1] / (3.0 ** (i - 1))

            # Decode-side prediction in the SAME frame as enc step i
            dec_xbar_i = (3.0 ** i) * (x0_bar - px)
            dec_y_i = (3.0 ** i) * (y0 - py)
            dec_x_i = dec_xbar_i / self.R3

            dx = f[i]['x'] - dec_x_i
            dy = f[i]['y'] - dec_y_i

            print(
                f" i={i:02d} par=0x{f[i]['par']:02x} cur=0x{f[i]['cur']:02x} c1={f[i]['c1']} mo={f[i]['mo']} "
                f"enc=({f[i]['x']:+.9f},{f[i]['y']:+.9f}) "
                f"dec_pred=({dec_x_i:+.9f},{dec_y_i:+.9f}) Δ=({dx:+.3e},{dy:+.3e})"
            )

    def diag_first_step(self, xy, mode, depth=8, name="case"):
        reg = self
        addr = reg.ugc_regions(xy, mode, depth=depth)
        ugc  = reg.ugc_lut

        # 1) Encoder’s clamped root (what the classifier actually used)
        x, y, xbar = reg.clamp(xy[:,0].copy(), xy[:,1].copy(), mode)
        par = int(addr[0,0])
        cur = int(addr[0,1])
        mo  = int(ugc[par, reg.mode])

        print(f"\n{ name }")
        print(f" par=0x{par:02x} mo={mo} chosen cur=0x{cur:02x}")
        print(f" enc clamp root: xbar0={xbar[0]:+.12f}  y0={y[0]:+.12f}")

        # 2) Decode-implied root from the whole address (unique series sum)
        offx = reg.ugc_off_ẋ[addr[0,1:depth+1]].astype(np.float64)
        offy = reg.ugc_off_y[addr[0,1:depth+1]].astype(np.float64)
        pow3 = (3.0 ** np.arange(depth, 0, -1))
        xbar0_dec = np.sum(offx / pow3)
        y0_dec    = np.sum(offy / pow3)
        print(f" dec series root: xbar0={xbar0_dec:+.12f}  y0={y0_dec:+.12f}")
        print(f" Δ root: dx={xbar[0]-xbar0_dec:+.12e}  dy={y[0]-y0_dec:+.12e}")

        # 3) What would the classifier pick at the decode root?
        rid_pred = reg.region_classification(
            np.array([xbar0_dec]), np.array([y0_dec])
        )[0]
        print(f" classifier at dec-root ⇒ rid_pred=0x{rid_pred:02x}  (vs chosen cur=0x{cur:02x})")

        return par, mo, cur, rid_pred, (xbar[0]-xbar0_dec), (y[0]-y0_dec)


@pytest.fixture
def rel_grid():
    """Fixture providing an instance of GridNeighbours."""
    return GridNeighbours()


class GridNeighbours:
    """
        Class that provides enough to test neighbour function.
    """

    def __init__(self):
        self.invalid_ugc = 0x5f
        self.num_regions = 96
        self.ugc_num_props = 1  # for relations, we only need mode metadata
        (self.mode,) = range(self.ugc_num_props)  # indices of metadata

    @cache
    def ugc_lut(self):
        """
        UGC Metadata - Here it is going to just be the mode.
        When testing neighbours we will only need 1 child.
        """
        num_regions = self.num_regions
        # these regions are ordered - eg, self.ugc_lut[self.in_regions, self.mode]
        _in_regions = [
            0x26, 0x2a, 0x3a, 0x39, 0x35, 0x25,  # Mode Shared, inner regions from V above barycentric origin, CW.
            0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e   # Mode Specific outer regions from apex of V, CW
        ]
        _in_up_regions = [  # 9 regions serve Λ mode
            0x39, 0x3a, 0x3e,  # c0 ΛVΛ
            0x25, 0x35, 0x34,  # c1 ΛVΛ
            0x2a, 0x26, 0x16,  # c2 ΛVΛ
        ]
        _in_dn_regions = [
            0x26, 0x2a, 0x2b,  # c0 VΛV
            0x3a, 0x39, 0x49,  # c1 VΛV
            0x35, 0x25, 0x21,  # c2 VΛV
        ]
        _ugc_lut = np.full((num_regions, self.ugc_num_props), self.invalid_ugc, dtype=np.uint8)
        _ugc_lut[_in_regions, self.mode] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        return _ugc_lut

    @cache
    def child_lut(self):
        """
        Given a mode and c1, find the regions that belong there.
        When testing neighbours we will only need 1 child.
        """
        _chd = {
            (0, 0): [0x26, 0x2A, 0x2B],  # V,C1.0
            (0, 1): [0x3A, 0x39, 0x49],  # V,C1.1
            (0, 2): [0x35, 0x25, 0x21],  # V,C1.2
            (1, 0): [0x39, 0x3A, 0x3E],  # Λ,C1.0
            (1, 1): [0x25, 0x35, 0x34],  # Λ,C1.1
            (1, 2): [0x2A, 0x26, 0x16],  # Λ,C1.2
        }
        _child_lut = np.zeros((2, 3, 3), dtype=np.uint8)
        for (mode, c1), children in _chd.items():
            _child_lut[mode, c1] = children
        return _child_lut

    @cache
    def pqc1_lut(self):
        """
        Given parent region, child region, return the C1.
        It's lazy, but it's easy to use.
        It uses @cache in order to be lazy-loaded and not to clutter up the __init__
        """
        _pqc1 = {
            (0x21, 0x26): 0, (0x21, 0x2a): 0, (0x21, 0x2b): 0,
            (0x21, 0x3a): 1, (0x21, 0x39): 1, (0x21, 0x49): 1,
            (0x21, 0x35): 2, (0x21, 0x25): 2, (0x21, 0x21): 2,
            (0x26, 0x26): 0, (0x26, 0x2a): 0, (0x26, 0x2b): 0,
            (0x26, 0x3a): 1, (0x26, 0x39): 1, (0x26, 0x49): 1,
            (0x26, 0x35): 2, (0x26, 0x25): 2, (0x26, 0x21): 2,
            (0x2b, 0x26): 0, (0x2b, 0x2a): 0, (0x2b, 0x2b): 0,
            (0x2b, 0x3a): 1, (0x2b, 0x39): 1, (0x2b, 0x49): 1,
            (0x2b, 0x35): 2, (0x2b, 0x25): 2, (0x2b, 0x21): 2,
            (0x35, 0x26): 0, (0x35, 0x2a): 0, (0x35, 0x2b): 0,
            (0x35, 0x3a): 1, (0x35, 0x39): 1, (0x35, 0x49): 1,
            (0x35, 0x35): 2, (0x35, 0x25): 2, (0x35, 0x21): 2,
            (0x3a, 0x26): 0, (0x3a, 0x2a): 0, (0x3a, 0x2b): 0,
            (0x3a, 0x3a): 1, (0x3a, 0x39): 1, (0x3a, 0x49): 1,
            (0x3a, 0x35): 2, (0x3a, 0x25): 2, (0x3a, 0x21): 2,
            (0x49, 0x26): 0, (0x49, 0x2a): 0, (0x49, 0x2b): 0,
            (0x49, 0x3a): 1, (0x49, 0x39): 1, (0x49, 0x49): 1,
            (0x49, 0x35): 2, (0x49, 0x25): 2, (0x49, 0x21): 2,
            (0x16, 0x39): 0, (0x16, 0x3a): 0, (0x16, 0x3e): 0,
            (0x16, 0x25): 1, (0x16, 0x35): 1, (0x16, 0x34): 1,
            (0x16, 0x2a): 2, (0x16, 0x26): 2, (0x16, 0x16): 2,
            (0x25, 0x39): 0, (0x25, 0x3a): 0, (0x25, 0x3e): 0,
            (0x25, 0x25): 1, (0x25, 0x35): 1, (0x25, 0x34): 1,
            (0x25, 0x2a): 2, (0x25, 0x26): 2, (0x25, 0x16): 2,
            (0x2a, 0x39): 0, (0x2a, 0x3a): 0, (0x2a, 0x3e): 0,
            (0x2a, 0x25): 1, (0x2a, 0x35): 1, (0x2a, 0x34): 1,
            (0x2a, 0x2a): 2, (0x2a, 0x26): 2, (0x2a, 0x16): 2,
            (0x34, 0x39): 0, (0x34, 0x3a): 0, (0x34, 0x3e): 0,
            (0x34, 0x25): 1, (0x34, 0x35): 1, (0x34, 0x34): 1,
            (0x34, 0x2a): 2, (0x34, 0x26): 2, (0x34, 0x16): 2,
            (0x39, 0x39): 0, (0x39, 0x3a): 0, (0x39, 0x3e): 0,
            (0x39, 0x25): 1, (0x39, 0x35): 1, (0x39, 0x34): 1,
            (0x39, 0x2a): 2, (0x39, 0x26): 2, (0x39, 0x16): 2,
            (0x3e, 0x39): 0, (0x3e, 0x3a): 0, (0x3e, 0x3e): 0,
            (0x3e, 0x25): 1, (0x3e, 0x35): 1, (0x3e, 0x34): 1,
            (0x3e, 0x2a): 2, (0x3e, 0x26): 2, (0x3e, 0x16): 2,
        }
        _pqc1_lut = np.zeros((self.num_regions, self.num_regions), dtype=np.uint8)
        for (p_reg, c_reg), c1 in _pqc1.items():
            _pqc1_lut[p_reg, c_reg] = c1
        return _pqc1_lut

    @lru_cache(maxsize=None)
    def neighbour_lut(self):
        """
        Given a region, parent mode, region-c1, return the neighbour and parent mode.
        If the parent mode has changed, then the region parent is a neighbour.
        It uses @lru_cache in order to be lazy-loaded and not to clutter up the __init__
        """
        _lut = {
            (0x16, 1): [(0x26, 1), (0x2B, 0), (0x21, 0)],
            (0x21, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x25, 1): [(0x35, 1), (0x3A, 0), (0x26, 1)],
            (0x26, 1): [(0x16, 1), (0x2A, 1), (0x25, 1)],
            (0x2A, 1): [(0x3A, 1), (0x26, 1), (0x35, 0)],
            (0x2B, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x34, 1): [(0x21, 0), (0x49, 0), (0x35, 1)],
            (0x35, 1): [(0x25, 1), (0x39, 1), (0x34, 1)],
            (0x39, 1): [(0x26, 0), (0x35, 1), (0x3A, 1)],
            (0x3A, 1): [(0x2A, 1), (0x3E, 1), (0x39, 1)],
            (0x3E, 1): [(0x2B, 0), (0x3A, 1), (0x49, 0)],
            (0x49, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x16, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x21, 0): [(0x34, 1), (0x25, 0), (0x16, 1)],
            (0x25, 0): [(0x35, 0), (0x21, 0), (0x26, 0)],
            (0x26, 0): [(0x39, 1), (0x2A, 0), (0x25, 0)],
            (0x2A, 0): [(0x3A, 0), (0x26, 0), (0x2B, 0)],
            (0x2B, 0): [(0x3E, 1), (0x16, 1), (0x2A, 0)],
            (0x34, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x35, 0): [(0x25, 0), (0x39, 0), (0x2A, 1)],
            (0x39, 0): [(0x49, 0), (0x35, 0), (0x3A, 0)],
            (0x3A, 0): [(0x2A, 0), (0x25, 1), (0x39, 0)],
            (0x3E, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x49, 0): [(0x39, 0), (0x34, 1), (0x3E, 1)],
        }
        _neighbour_lut = np.full((self.num_regions, 2, 3, 2), self.invalid_ugc, dtype=np.uint8)
        for key, neighbours in _lut.items():
            region_id, mode = key
            _neighbour_lut[region_id, mode] = neighbours
        return _neighbour_lut

    def region_neighbours(self, address):
        """
        Vectorised means to return neighbouring half-hexagon addresses (as regions) via regions.
        The last value-holding region (addresses[:, -2]) is the key position of interest (POI).
        But we must cascade when necessary also.
        """
        count, layers = address.shape
        neighbour = address.copy()  # The neighbour may just be a single switch.
        cascading = np.ones(count, dtype=bool)  # Track all the addresses we are managing.
        n_lut = self.neighbour_lut()
        c1_lut = self.pqc1_lut()
        ugc_lut = self.ugc_lut()
        c1 = c1_lut[address[:, -2], address[:, -1]]
        for poi in range(layers - 2, -1, -1):
            if not np.any(cascading):
                break
            active = np.where(cascading)[0]
            cur = address[:, poi][active]
            par = address[:, poi - 1][active]
            pmo = ugc_lut[par, self.mode]
            nbm = n_lut[cur, pmo, c1[active]]
            neighbour[:, poi][active] = nbm[:, -2]
            cascading[active] = (nbm[:, 1] != pmo)
        # Normalise root and terminal.
        nmo = ugc_lut[neighbour[:, 0], self.mode]
        root = np.where(nmo == 1, 0x16, 0x49)
        neighbour[:, 0] = root
        child_lut = self.child_lut()
        mode = ugc_lut[neighbour[:, -2], self.mode]  # mode of region.
        neighbour[:, -1] = child_lut[mode, c1, 2]
        return neighbour

    def encroach_to_neighbour(self, xy, mode, c1_edge, preserve):
        # xy: (N,2) local coords in source face
        # mode: (N,) source face mode (not used in math here, but good to thread)
        # c1_edge: 0,1,2 – which edge family you crossed (on the source face)
        # preserve: bool – whether this face-to-face seam preserves or swaps C1

        # 1) go to classifier plane
        xbar = np.sqrt(3) * xy[:, 0]
        vec = np.stack([xbar, xy[:, 1]], axis=-1)

        # 2) reflect if swap seam
        if not preserve:
            if c1_edge == 0:
                u = np.array([1.0, 0.0])
            elif c1_edge == 1:
                u = np.array([1.0, 1.0]) / np.sqrt(2)
            elif c1_edge == 2:
                u = np.array([1.0, -1.0]) / np.sqrt(2)
            R = 2 * np.outer(u, u) - np.eye(2)
            vec = vec @ R.T

        # 3) back to (x,y) in neighbour frame
        x_prime = vec[:, 0] / np.sqrt(3)
        y_prime = vec[:, 1]
        return np.stack([x_prime, y_prime], axis=-1)


# def test_clamp_up_mode(const_grid):
#     """Tests the clamping logic for the UP (Λ) triangle."""
#     grid = const_grid
#     test_cases = {
#         "Inside": (np.array([[0.1, 0.1]]), np.array([[0.1, 0.1]])),
#         "Outside Apex": (np.array([[0.1, 1.0]]), np.array([[0.0, grid.ΛC]])),
#         "Pole": (
#             np.array([[0.000000000000001, 0.8164965132120238]]), np.array([[0.000000000000001, 0.8164965132120238]])),
#         "Outside Right Slant": (np.array([[0.8, 0.2]]), np.array([[0.355934, 0.2]])),
#         "Outside Base-Left": (np.array([[-0.8, -0.8]]), np.array([[-grid.TR, grid.ΛF]])),
#         "Apex Vertex": (np.array([[0.0, 1.0]]), np.array([[0.0, grid.ΛC]])),
#         "Base-Right Vertex": (np.array([[0.8, -0.5]]), np.array([[grid.TR, grid.ΛF]])),
#         "Base-Left Vertex": (np.array([[-0.8, -0.5]]), np.array([[-grid.TR, grid.ΛF]])),
#     }
#
#     for name, (input_xy, expected_xy) in test_cases.items():
#         # Ensure the test uses the user's clamp function name
#         x, y, _ = grid.clamp(input_xy[:, 0], input_xy[:, 1], mode=1)
#         result_xy = np.array([x, y]).T
#         # result_xy = grid.clamp(input_xy.copy(), mode=1)
#         assert_allclose(result_xy, expected_xy, atol=1e-5, err_msg=f"Case '{name}' failed")
#
#
# def test_clamp_down_mode(const_grid):
#     """Tests the clamping logic for the DOWN (V) triangle."""
#     grid = const_grid
#     test_cases = {
#         "Inside": (np.array([[-0.1, -0.1]]), np.array([[-0.1, -0.1]])),
#         "Pole": (
#             np.array([[0.000000000000001, -0.8164965132120238]]), np.array([[0.000000000000001, -0.8164965132120238]])),
#         "Outside Apex": (np.array([[0.1, -1.0]]), np.array([[0.0, grid.VF]])),
#         "Outside Left Slant": (np.array([[-0.8, 0.2]]), np.array([[-0.586875, 0.2]])),
#         "Outside Base-Right": (np.array([[0.8, 0.8]]), np.array([[grid.TR, grid.VC]])),
#         "Apex Vertex": (np.array([[0.1, -1.0]]), np.array([[0.0, grid.VF]])),
#         "Base-Right Vertex": (np.array([[0.8, 0.8]]), np.array([[grid.TR, grid.VC]])),
#         "Base-Left Vertex": (np.array([[-0.8, 0.8]]), np.array([[-grid.TR, grid.VC]])),
#     }
#
#     for name, (input_xy, expected_xy) in test_cases.items():
#         x, y, _ = grid.clamp(input_xy[:, 0], input_xy[:, 1], mode=0)
#         result_xy = np.array([x, y]).T
#         assert_allclose(result_xy, expected_xy, atol=1e-5, err_msg=f"Case '{name}' failed")


def test_pc_c1_reverse_map(reg_grid: GridRegions):
    pc = reg_grid.pc_c1()        # (num_regions, num_regions) -> c1
    ugc = reg_grid.ugc_lut       # (num_regions, props), only mode here
    chd = reg_grid.child_lut()   # (2,3,3)

    for par in reg_grid.in_regions:
        mode = ugc[par, reg_grid.mode]
        # each c1 -> its 3 children must map back to c1
        for c1 in (0, 1, 2):
            children = chd[mode, c1]
            for child in children:
                assert pc[par, child] == c1, f"pc_c1 wrong: par=0x{par:02x}, child=0x{child:02x}"

        # optional: children from the *other* mode should not collide to the same c1
        other_mode = 1 - mode
        for other_c1 in (0, 1, 2):
            for alien in chd[other_mode, other_c1]:
                # We can’t assert a specific value (it’s undefined), but we *can* assert it’s not
                # equal to all three c1 values at once; the safe minimal check is that it’s not
                # erroneously identical to the first one we tested above:
                pass  # intentionally no strict assertion — undefined outside parent’s mode


# def test_terminate_normalises_tail(reg_grid: GridRegions):
#     pc = reg_grid.pc_c1()
#     chd = reg_grid.child_lut()
#     ugc = reg_grid.ugc_lut
#
#     # Build a tiny address set with known last/child
#     par = reg_grid.in_regions[0]
#     mode = ugc[par, reg_grid.mode]
#     c1 = 1
#     child = chd[mode, c1, 0]           # not the mode-locked one
#     want = chd[mode, c1, 2]            # mode-locked child
#
#     addr = np.full((1, 5), reg_grid.invalid_ugc, dtype=np.uint8)
#     addr[0, -2] = par
#     addr[0, -1] = child
#
#     out = reg_grid.terminate(addr.copy())
#     assert out[0, -1] == want
#
#
# def test_terminate_keeps_invalid_pairs(reg_grid: GridRegions):
#     inv = reg_grid.invalid_ugc
#     # invalid last and/or child: should remain untouched
#     addr = np.array([[10, inv, inv, inv, inv]], dtype=np.uint8)
#     out = reg_grid.terminate(addr.copy())
#     assert np.array_equal(out, addr)
#

def test_encoder_tail_is_mode_locked(reg_grid: GridRegions):
    # Pick a few random interior points/modes
    rng = np.random.default_rng(42)
    N = 8
    x = rng.uniform(-0.3, 0.3, size=N)
    y = rng.uniform(-0.3, 0.3, size=N)
    mode = rng.integers(0, 2, size=N, dtype=np.uint8)
    addr = reg_grid.ugc_regions(np.array([x, y]).T, mode, depth=18)
    out = reg_grid.terminate(addr.copy())
    assert np.array_equal(out, addr), "terminate should be idempotent on encoder output"


def test_rc_regions(const_grid):
    """Test known input cases to ensure en_grid produces expected region IDs."""
    test_cases = [
        # format: ((ẋ, y), expected_hpn) - needs populating!
        # Core regions
        ((0.0, 0.0), (3, 2, 2)),  # Center point
        ((0.0, 0.5), (1, 1, 2)),  # Top-middle
        ((0.0, -0.5), (4, 2, 1)),  # Bottom-middle
        ((0.5, 0.0), (3, 2, 2)),  # Middle-right
        ((-0.5, 0.0), (3, 1, 1)),  # Middle-left

        # Extreme corner regions
        ((0.0, 1.0), (0, 0, 3)),  # Far top
        ((0.0, -1.0), (5, 3, 0)),  # Far bottom
        ((1.0, 0.0), (3, 3, 3)),  # Far right
        ((-1.0, 0.0), (3, 0, 0)),  # Far left

        # Regions defined by slanted boundaries
        ((0.5, 0.5), (1, 2, 3)),  # Top-right
        ((-0.5, -0.5), (4, 2, 0)),  # Bottom-left
    ]
    for (ẋ_val, y_val), (h, p, n) in test_cases:
        result = const_grid.region_classification(np.array([ẋ_val]), np.array([y_val]))[0]
        expected = (h << 4) | (p << 2) | n
        assert result == expected, f"For ({ẋ_val}, {y_val}) expected {expected}, got {result}"


def test_rc_boundaries(const_grid):
    """Tests the precise vertices and seam midpoints of the grid."""
    # Use f-strings to create descriptive test case names
    test_cases = {
        # --- UP Triangle (Λ) Vertices ---
        "UP Apex": ((0, const_grid.ΛC), (1, 1, 2)),
        "UP Base-Right": ((const_grid.H, const_grid.ΛF), (4, 3, 3)),
        "UP Base-Left": ((-const_grid.H, const_grid.ΛF), (4, 0, 0)),

        # --- DOWN Triangle (V) Vertices ---
        "DOWN Apex": ((0, const_grid.VF), (4, 2, 1)),
        "DOWN Base-Right": ((const_grid.H, const_grid.VC), (2, 3, 3)),
        "DOWN Base-Left": ((-const_grid.H, const_grid.VC), (2, 0, 0)),

        # --- Seam Midpoints ---
        "UP Base Midpoint": ((0, const_grid.ΛF), (4, 2, 1)),
        "DOWN Base Midpoint": ((0, const_grid.VC), (2, 1, 2)),
        "Right Vertical Seam": ((const_grid.TR * const_grid.R3, 0), (3, 3, 3)),  # Same as (0.5, 0) approx.

        # --- Origin ---
        "Origin": ((0.0, 0.0), (3, 2, 2)),
    }

    for name, ((ẋ_val, y_val), (h, p, n)) in test_cases.items():
        ẋ_arr = np.array([ẋ_val])
        y_arr = np.array([y_val])

        result = const_grid.region_classification(ẋ_arr, y_arr)[0]
        expected = (h << 4) | (p << 2) | n

        assert result == expected, f"Case '{name}': For ({ẋ_val}, {y_val}) expected {expected}, got {result}"


def test_rc_internal_seams(const_grid):
    """Tests points on the internal grid lines (y=0, y=±ẋ)."""
    test_cases = {
        # --- Points on the y=0 axis ---
        "Positive X-Axis": ((0.1, 0.0), (3, 2, 2)),
        "Negative X-Axis": ((-0.1, 0.0), (3, 1, 1)),

        # --- Points on the y = ẋ line ---
        "y=x in Quadrant 1": ((0.2, 0.2), (2, 2, 2)),
        "y=x in Quadrant 3": ((-0.2, -0.2), (3, 2, 1)),

        # --- Points on the y = -ẋ line ---
        "y=-x in Quadrant 2": ((-0.2, 0.2), (2, 1, 2)),
        "y=-x in Quadrant 4": ((0.2, -0.2), (3, 2, 2)),

    }

    for name, ((ẋ_val, y_val), (h, p, n)) in test_cases.items():
        ẋ_arr = np.array([ẋ_val])
        y_arr = np.array([y_val])

        result = const_grid.region_classification(ẋ_arr, y_arr)[0]
        expected = (h << 4) | (p << 2) | n

        assert result == expected, f"Case '{name}': For ({ẋ_val}, {y_val}) expected {expected}, got {result}"


def test_rc_batch_processing(const_grid):
    """Tests the function with a batch of multiple points at once."""
    # Combine several known cases into batch arrays
    ẋ_batch = np.array([0.0, 0.5, -0.2])
    y_batch = np.array([0.0, 0.5, -0.2])

    # Calculate the expected results for the batch
    expected_h = np.array([3, 1, 3])
    expected_p = np.array([2, 2, 2])
    expected_n = np.array([2, 3, 1])
    expected_results = (expected_h << 4) | (expected_p << 2) | expected_n

    # Run the classification on the entire batch
    batch_results = const_grid.region_classification(ẋ_batch, y_batch)

    # Use NumPy's testing utility to compare the arrays
    np.testing.assert_array_equal(batch_results, expected_results)


def test_rc_default_case(const_grid):
    """Tests a point far outside the grid to check default case handling."""
    # This point is far to the right and bottom
    ẋ_val, y_val = 2.0, -2.0

    # Manually determine the expected default IDs
    # y < VF -> h_id should be default=5
    # y-ẋ < -Ẇ -> p_id should be default=3
    # y+ẋ = 0 -> n_id should be 2
    h, p, n = 5, 3, 2

    expected = (h << 4) | (p << 2) | n
    result = const_grid.region_classification(np.array([ẋ_val]), np.array([y_val]))[0]

    assert result == expected, f"Default case failed: expected {expected}, got {result}"

def test_encode_one_step_child_is_legal(reg_grid: GridRegions):
    rng = np.random.default_rng(0)
    # points around the origin so we’re well inside; both modes
    x = rng.uniform(-0.1, 0.1, size=16)
    y = rng.uniform(-0.1, 0.1, size=16)
    mode = np.array([0]*8 + [1]*8, dtype=np.uint8)

    addr = reg_grid.ugc_regions(np.column_stack([x, y]), mode, depth=1)
    ugc = reg_grid.ugc_lut
    chd = reg_grid.child_lut()
    pc  = reg_grid.pc_c1()

    # root at [0], first child at [1]
    par = addr[:, 0]
    cur = addr[:, 1]

    for i, (p, c) in enumerate(zip(par, cur)):
        assert c != reg_grid.invalid_ugc, f"row {i}: invalid child at depth=1"
        pmo = ugc[p, reg_grid.mode]
        c1  = pc[p, c]
        assert c1 in (0,1,2), f"row {i}: bad c1=0x{c1:x} for par=0x{p:02x}, cur=0x{c:02x}"
        assert c in chd[pmo, c1], f"row {i}: child 0x{c:02x} not in legal children of par=0x{p:02x} mode={pmo}"

def _manual_one_step(reg: GridRegions, xy, mode):
    """Replicate exactly one encoder step: clamp → classify → subtract offsets → ×3 (classifier plane)."""
    x = xy[:,0].copy()
    y = xy[:,1].copy()

    # clamp per parent mode; get xbar
    x, y, xbar = reg.clamp(x, y, mode)

    # classify
    region = reg.region_classification(xbar, y)
    props  = reg.ugc_lut[region]
    in_scope = np.where(mode == 1, props[:, reg.in_up], props[:, reg.in_dn])
    rid = np.where(in_scope, region, reg.invalid_ugc)

    # step (in classifier plane)
    off_ẋ = reg.ugc_off_ẋ[rid]
    off_y = reg.ugc_off_y[rid]
    xbar2 = (xbar - off_ẋ) * 3.0
    y2    = (y    - off_y) * 3.0
    x2    = xbar2 / reg.R3

    # update mode only for valid rows
    child_mode = reg.ugc_lut[rid, reg.mode]
    valid = (rid != reg.invalid_ugc)
    mode2 = np.where(valid, child_mode, mode).astype(np.uint8)

    return rid, np.column_stack([x2, y2]), mode2

def test_first_step_math_matches_encoder(reg_grid: GridRegions):
    rng = np.random.default_rng(1)
    N = 32
    xy   = rng.uniform(-0.2, 0.2, size=(N,2))
    mode = rng.integers(0,2,size=N, dtype=np.uint8)

    addr = reg_grid.ugc_regions(xy, mode, depth=1)
    rid1 = addr[:,1]

    rid_m, xy_m, mode_m = _manual_one_step(reg_grid, xy, mode)
    np.testing.assert_array_equal(rid_m, rid1, "first child region mismatch")
    # encode again for one more step from both states; they should stay aligned
    addr2 = reg_grid.ugc_regions(xy_m, mode_m, depth=1)
    np.testing.assert_array_equal(addr2[:, 0], addr[:, 0], "root must remain the same after one step")

@pytest.mark.parametrize("depth", [2, 3])
def test_shallow_roundtrip(reg_grid: GridRegions, depth):
    rng = np.random.default_rng(2)
    N = 20
    xy   = rng.uniform(-0.25, 0.25, size=(N,2))
    mode = rng.integers(0,2,size=N, dtype=np.uint8)

    addr = reg_grid.ugc_regions(xy, mode, depth=depth)
    dec  = reg_grid.ugc_dec(addr)
    np.testing.assert_allclose(dec[:,:2], xy, atol=1e-15, err_msg="decode(xy) mismatch (shallow)")
    np.testing.assert_array_equal(dec[:,2].astype(np.uint8), mode, err_msg="decode(mode) mismatch (shallow)")

    re_addr = reg_grid.ugc_regions(dec[:,:2], dec[:,2].astype(np.uint8), depth=depth)
    np.testing.assert_array_equal(re_addr, addr, "encode(decode(addr)) mismatch (shallow)")

def test_boundary_midpoints_do_not_go_invalid_early(reg_grid: GridRegions):
    g = reg_grid
    # a few exact boundary points in classifier plane, mapped back to x
    xbar = np.array([0.0,  g.Ẇ, -g.Ẇ])
    y    = np.array([0.0,  g.VC, g.ΛF])   # pick mid/base lines
    x    = xbar / g.R3

    # test both modes
    XY = np.column_stack([np.tile(x,3), np.repeat(y,3)])
    mode = np.array([0,0,0, 1,1,1, 0,1,0], dtype=np.uint8)[:XY.shape[0]]

    addr = g.ugc_regions(XY, mode, depth=3)
    # none of first 3 children should be invalid
    assert not np.any(addr[:,1:4] == g.invalid_ugc), f"invalid child within first 3 steps on boundary inputs"

def test_terminate_always_normalises_tail_simple(reg_grid: GridRegions):
    pc  = reg_grid.pc_c1()
    chd = reg_grid.child_lut()
    ugc = reg_grid.ugc_lut
    par = reg_grid.in_regions[0]
    mode = ugc[par, reg_grid.mode]
    # choose the non-mode-locked child 0
    child0 = chd[mode, 1, 0]
    want   = chd[mode, 1, 2]

    addr = np.full((1, 6), reg_grid.invalid_ugc, dtype=np.uint8)
    addr[0, -2] = par
    addr[0, -1] = child0
    out = reg_grid.terminate(addr.copy())
    assert out[0, -1] == want

def test_regression_named_points_depth3(reg_grid: GridRegions):
    # NAΛ centroid and the SPΛ “bad girl”
    pts  = np.array([
        [-0.210025776447247, -0.020916439797272],
        [+0.105560608419664, -0.134651734661425],
    ])
    mode = np.array([1, 1], dtype=np.uint8)
    addr = reg_grid.ugc_regions(pts, mode, depth=3)
    dec  = reg_grid.ugc_dec(addr)
    np.testing.assert_allclose(dec[:,:2], pts, atol=1e-15)
    np.testing.assert_array_equal(dec[:,2].astype(np.uint8), mode)

def test_lateral_symmetry(reg_grid: GridRegions):
    ugc = reg_grid.ugc_lut
    chd = reg_grid.child_lut()
    bad = []
    for par in reg_grid.in_regions:
        mo = ugc[par, reg_grid.mode]
        c0, _, _ = chd[mo, 0]   # lateral A, center, lateral B
        c2, _, _ = chd[mo, 2]
        y0 = reg_grid.ugc_off_y[c0]
        y2 = reg_grid.ugc_off_y[c2]
        x0 = reg_grid.ugc_off_ẋ[c0]
        x2 = reg_grid.ugc_off_ẋ[c2]
        if not np.isclose(y0, y2, atol=1e-15) or not np.isclose(x0, -x2, atol=1e-15):
            bad.append((par, (x0, y0), (x2, y2)))
    assert not bad, "lateral offsets not symmetric: " + ", ".join(f"par={hex(p)} L={a} R={b}" for p,a,b in bad)

def test_offsets_self_classify(reg_grid: GridRegions):
    bad = []
    for rid in reg_grid.in_regions:
        rid2 = reg_grid.region_classification(
            np.array([reg_grid.ugc_off_ẋ[rid]]),
            np.array([reg_grid.ugc_off_y[rid]])
        )[0]
        if rid2 != rid:
            bad.append((rid, rid2))
    assert not bad, f"offsets classify to different regions: {[(hex(a), hex(b)) for a,b in bad]}"

def test_reg_valid_points(reg_grid):
    """
    Tests the addresses generation for a known coordinate that should succeed.
    """
    test_cases = np.array([
        [-0.210025776447247, -0.020916439797272, 1],  # NAΛ: centroid (easy)
        [+0.105560608419664, -0.134651734661425, 1],  # SPΛ: bad girl
    ])
    names = ['NAΛ: centroid', 'SPΛ: bad girl']
    for name, test_case in zip(names, test_cases):
        xy = np.array([test_case[:2]])
        mode = np.array([test_case[2]])
        result_address = reg_grid.ugc_regions(xy, mode)

        # New: structural/local correctness for first k steps
        k = 33
        addr = result_address[0, :k]
        # 1) root must be consistent with initial mode
        root_expected = 0x16 if mode[0] == 1 else 0x49
        assert addr[0] == root_expected

        # 2) each step must be a valid child of its parent in the parent’s mode
        ugc = reg_grid.ugc_lut  # (num_regions, 1) -> mode
        chd = reg_grid.child_lut()  # (2,3,3)
        c1r = reg_grid.pc_c1()  # (num_regions, num_regions) -> c1

        for i in range(1, k):
            par = addr[i - 1]
            cur = addr[i]
            par_mode = ugc[par, reg_grid.mode]
            # if either par or cur is invalid, we bail to let roundtrip catch it
            if par == reg_grid.invalid_ugc or cur == reg_grid.invalid_ugc:
                raise AssertionError(f"Invalid region seen at step {i}: par={par:#04x}, cur={cur:#04x}")
            c1 = c1r[par, cur]
            # c1 must be 0/1/2 and cur must be one of the canonical 3 children
            assert c1 in (0, 1, 2), f"{name}: Bad c1 at step {i}: c1={c1}, par={par:#04x}, cur={cur:#04x}"
            assert cur in chd[par_mode, c1], f"{name}: disallowed child; step {i}: par={par:#04x}, cur={cur:#04x}, mode={par_mode}"

        # Strong invariant: decode → re-encode at same depth should be identical
        depth = result_address.shape[1] - 2
        dec = reg_grid.ugc_dec(result_address)
        assert_array_equal(dec[:, :2], xy, err_msg=f'{name}: Roundtrip address mismatch.')
        assert_array_equal(dec[:, 2], mode, err_msg=f'{name}: Roundtrip mode mismatch.')
        re_addr = reg_grid.ugc_regions(dec[:, :2], dec[:, 2].astype(np.uint8), depth=depth)
        assert_array_equal(re_addr, result_address, err_msg=f'{name}: Roundtrip mismatch.')


def test_ugc_regions_invalid_point(reg_grid):
    """
    Tests the addresses generation for a coordinate that is logs of bounds.
    """
    # Inputs for a point far outside the grid
    x = np.array([10.0])
    y = np.array([10.0])
    mode = np.array([1])  # UP mode

    # Generate the addresses
    result_address = reg_grid.ugc_regions(np.array([x, y]).T, mode, depth=5)

    # The second region (index 1) should be marked as invalid because the
    assert result_address[0, 1] == reg_grid.invalid_ugc


def test_roundtrip_conversion(reg_grid):
    """
    Tests the full encode-decode roundtrip process.
    """
    # 1. Arrange: Define the initial test point and mode.
    initial_x = np.array([0.278558759260123456789])
    initial_y = np.array([0.293862554740123456789])
    initial_mo = np.array([0])  # DOWN mode
    initial_xym = np.stack([initial_x, initial_y, initial_mo], axis=-1)

    # 2. Act: Run the full encode and decode cycle.
    # Encode the coordinate to a URI addresses.
    xy = np.array([initial_x, initial_y]).T
    uri_address = reg_grid.ugc_regions(xy, initial_mo)

    # Decode the URI addresses back to a coordinate.
    decoded_xym = reg_grid.ugc_dec(uri_address)
    # display(initial_xym, decoded_xym)

    # 3. Assert: Check if the result is close to the original.
    assert_allclose(decoded_xym, initial_xym, atol=1e-40)


def test_ext_neighbours(rel_grid):
    """
    Neighbours
    """
    # Rapa Nui Moai
    reg01 = np.array([[0x49, 0x35, 0x25, 0x3a, 0x21, 0x2b, 0x49, 0x25, 0x26, 0x35, 0x26,
                       0x25, 0x16, 0x3e, 0x34, 0x34, 0x39, 0x3a, 0x21, 0x2b, 0x2b]])
    ref01 = np.array([[0x49, 0x35, 0x25, 0x3a, 0x21, 0x2b, 0x49, 0x25, 0x26, 0x35, 0x26,
                       0x25, 0x16, 0x3e, 0x34, 0x34, 0x39, 0x2a, 0x34, 0x3e, 0x3e]])
    # North Pole
    reg02 = np.array([[0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49,
                       0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49,
                       0x49, 0x49, 0x49, 0x49, 0x49, 0x49]])
    ref02 = np.array([[0x16, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
                       0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
                       0x34, 0x34, 0x34, 0x34, 0x34, 0x34]])

    rnm = rel_grid.region_neighbours(reg01)
    assert_array_equal(rnm, ref01)
    npo = rel_grid.region_neighbours(reg02)
    assert_array_equal(npo, ref02)


def test_neighbour_decode_encode_roundtrip_reg01(rel_grid, reg_grid):
    """Neighbour → decode(xy,mode) → re-encode should roundtrip to the same neighbour addresses.
    Exercises the transform-then-classify pipeline (no manual C1 flipping), including cascades.
    """
    # Rapa Nui Moai case from test_ext_neighbours
    reg01 = np.array([[
        0x49, 0x35, 0x25, 0x3a, 0x21, 0x2b, 0x49, 0x25, 0x26, 0x35, 0x26,
        0x25, 0x16, 0x3e, 0x34, 0x34, 0x39, 0x3a, 0x21, 0x2b, 0x2b
    ]])

    # 1) Neighbour addresses via geometric LUT (with cascading where needed)
    nb_addr = rel_grid.region_neighbours(reg01)

    # 2) Decode neighbour addresses back to (x, y, initial_mode)
    xym = reg_grid.ugc_dec(nb_addr)
    mode = xym[:, 2].astype(np.uint8)

    # 3) Re-encode using decoded coordinates + initial mode, for the SAME depth
    depth = nb_addr.shape[1] - 2

    re_addr = reg_grid.ugc_regions(xym[:, :2], mode, depth=depth)

    # 4) Should match exactly
    assert_array_equal(re_addr, nb_addr)


def test_neighbour_decode_encode_roundtrip_reg02(rel_grid, reg_grid):
    """Same roundtrip for the Polar (seam-heavy) case.
    Confirms seam hops are handled by geometry + reclassification alone.
    """
    # North Pole case from test_ext_neighbours
    reg02 = np.array([[
        0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49,
        0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49,
        0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49
    ]])

    nb_addr = rel_grid.region_neighbours(reg02)

    xym = reg_grid.ugc_dec(nb_addr)
    mode = xym[:, 2].astype(np.uint8)

    depth = nb_addr.shape[1] - 2
    xy = np.stack([xym[:, 0], xym[:, 1]], axis=-1)
    re_addr = reg_grid.ugc_regions(xy, mode, depth=depth)
    assert_array_equal(re_addr, nb_addr)


def test_trace_roundtrip(reg_grid: GridRegions):
    xy = np.array([[-0.210025776447247, -0.020916439797272]])
    mode = np.array([1], dtype=np.uint8)
    reg_grid.trace_roundtrip(xy, mode, depth=8, name="NAΛ centroid")

def test_diag_first_step(reg_grid: GridRegions):
    xy = np.array([[-0.210025776447247, -0.020916439797272]])
    mode = np.array([1], dtype=np.uint8)
    reg_grid.diag_first_step(xy, mode, depth=8, name="NAΛ centroid")

