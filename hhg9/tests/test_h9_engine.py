"""
Unit tests for the grid region encoding function en_grid.
Validates correct region classification based on geometric thresholds.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
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

    def clamp(self, xy, mode):
        """
        Given an array of points, clamp them to be within the barycentric triangle.
        This should only be necessary when preparing points for projection to barycentre.
        """
        xx = xy[:, 0]
        yy = xy[:, 1]
        ẋ = self.R3 * xx
        eps = 1e-14  # A tolerance to detect if we're at a vertex

        if mode == 1:  # Clamping for the UP triangle
            invalid = (yy < self.ΛF) | (yy > (self.ΛC - np.abs(ẋ)))
            if np.any(invalid):
                yy = np.clip(yy, self.ΛF, self.ΛC)
                max_abs_ẋ = self.ΛC - yy
                at_apex = np.isclose(yy, self.ΛC, atol=eps)
                max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)
                ẋ_clamped = np.clip(ẋ, -max_abs_ẋ, max_abs_ẋ)
                at_base = np.isclose(yy, self.ΛF, atol=eps)
                xc = ẋ_clamped / self.R3
                xx_final = np.where(at_base, np.sign(xc) * self.TR, xc)
                yy_final = np.where(at_base, self.ΛF, yy)
                xy[:, 0] = xx_final
                xy[:, 1] = yy_final
        else:  # Clamping for the DOWN triangle
            invalid = (yy > self.VC) | (yy < (self.VF + np.abs(ẋ)))
            if np.any(invalid):
                yc = np.clip(yy, self.VF, self.VC)
                max_abs_ẋ = yc - self.VF
                at_apex = np.isclose(yy, self.VF, atol=eps)
                max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)
                # max_abs_ẋ = np.where(max_abs_ẋ < eps, 0.0, max_abs_ẋ)
                ẋ_clamped = np.clip(ẋ, -max_abs_ẋ, max_abs_ẋ)
                at_base = np.isclose(yc, self.VC, atol=eps)  # Use yc
                xc = ẋ_clamped / self.R3
                xx_final = np.where(at_base, np.sign(xc) * self.TR, xc)
                yy_final = np.where(at_base, self.VC, yc)  # Use yc
                xy[:, 0] = xx_final
                xy[:, 1] = yy_final
        return xy


@pytest.fixture
def reg_grid():
    """Fixture providing an instance of GridRegions."""
    return GridRegions()


class GridRegions(GridConstants):
    def __init__(self):
        super().__init__()
        self.POS = [
            # The co-ordinate to the centre of each sub-triangle
            # The order starts from '021' below the origin, and goes clockwise through the inner set, then the outer set.
            (0, -self.V * 2.), (-self.U, -self.V), (-self.U, self.V),
            (0, self.V * 2.), (self.U, self.V), (self.U, -self.V),
            (0, -self.V * 4.), (-self.U * 2., -self.V * 2.), (-self.U * 2., self.V * 2.),
            (0, self.V * 4.), (self.U * 2., self.V * 2.), (self.U * 2., -self.V * 2.)
        ]
        self.invalid_ugc = 0x5f
        self.num_regions = 96
        ugc_num_props = 11
        num_regions = self.num_regions
        # these regions are ordered - eg, self.ugc_lut[self.in_regions, self.mode]
        self.in_regions = [0x39, 0x35, 0x25, 0x26, 0x2a, 0x3a, 0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e]
        self.in_up_regions = [0x39, 0x3a, 0x3e, 0x25, 0x35, 0x34, 0x2a, 0x26, 0x16]
        self.in_dn_regions = [0x26, 0x2a, 0x2b, 0x3a, 0x39, 0x49, 0x35, 0x25, 0x21]

        self.ugc_off = np.full((num_regions, 2), 0., dtype=np.float64)
        self.ugc_off[self.in_regions] = self.POS
        (self.in_dn, self.in_up, self.mode, self.d_ci, self.u_ci,
         self.dc0, self.dc1, self.dc2, self.uc0, self.uc1, self.uc2) = range(ugc_num_props)
        self.ugc_lut = np.full((num_regions, ugc_num_props), self.invalid_ugc, dtype=np.uint8)
        self.ugc_lut[:, self.in_dn] = 0
        self.ugc_lut[:, self.in_up] = 0
        self.ugc_lut[self.in_regions, self.mode] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
        self.ugc_lut[self.invalid_ugc] = 0  # set no offsets for illegal!
        self.ugc_lut[self.in_up_regions, self.in_up] = 1
        self.ugc_lut[self.in_dn_regions, self.in_dn] = 1

    def ugc_regions(self, x, y, mode, depth=36):
        """
        Given a vector of Point coords create a set of regions
        """
        num_points = x.size
        addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
        addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)  # These values should come from the octant set.
        # history = np.zeros((num_points, depth + 2, 6))
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
        REVERSE: Decodes a URI address back into (x,y) coordinates and its
        initial mode.
        """
        num_points, depth = uri_address.shape
        # Initialize x and y with the precise remainder from the encoding process.
        x = np.zeros(num_points, dtype=np.float64)
        y = np.zeros(num_points, dtype=np.float64)

        # Loop backwards from the last layer down to the first REAL layer (index 1),
        # skipping the placeholder root at index 0.
        for i in range(depth - 1, 0, -1):
            region_id = uri_address[:, i]
            valid_mask = (region_id != self.invalid_ugc)

            x /= 3.0
            y /= 3.0

            if np.any(valid_mask):
                valid_ids = region_id[valid_mask]
                off = self.ugc_off[valid_ids]
                x[valid_mask] += off[:, 0]
                y[valid_mask] += off[:, 1]

        # After reconstructing the coordinates, find the initial mode from the root URI.
        initial_mode = self.ugc_lut[uri_address[:, 0], self.mode]

        # Stack all three results into a final (N, 3) array.
        return np.stack([x, y, initial_mode], axis=-1)


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

    @lru_cache(maxsize=None)
    def ugc_lut(self):
        """
        UGC Metadata - Here it is going to just be the mode.
        When testing neighbours we will only need 1 child.
        """
        num_regions = self.num_regions
        # these regions are ordered - eg, self.ugc_lut[self.in_regions, self.mode]
        _in_regions = [0x39, 0x35, 0x25, 0x26, 0x2a, 0x3a, 0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e]
        # _in_up_regions = [0x39, 0x3a, 0x3e, 0x25, 0x35, 0x34, 0x2a, 0x26, 0x16]
        # _in_dn_regions = [0x26, 0x2a, 0x2b, 0x3a, 0x39, 0x49, 0x35, 0x25, 0x21]
        _ugc_lut = np.full((num_regions, self.ugc_num_props), self.invalid_ugc, dtype=np.uint8)
        _ugc_lut[_in_regions, self.mode] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
        return _ugc_lut

    @lru_cache(maxsize=None)
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

    @lru_cache(maxsize=None)
    def pqc1_lut(self):
        """
        Given parent region, child region, return the C1.
        It's lazy, but it's easy to use.
        It uses @lru_cache in order to be lazy-loaded and not to clutter up the __init__
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

    def neighbours(self, address):
        """
        Vectorised means to return neighbouring half-hexagon addresses (as regions) via regions.
        The last value-holding region (address[:, -2]) is the key position of interest (POI).
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


def test_clamp_up_mode(const_grid):
    """Tests the clamping logic for the UP (Λ) triangle."""
    grid = const_grid
    test_cases = {
        "Inside": (np.array([[0.1, 0.1]]), np.array([[0.1, 0.1]])),
        "Outside Apex": (np.array([[0.1, 1.0]]), np.array([[0.0, grid.ΛC]])),
        "Pole": (
            np.array([[0.000000000000001, 0.8164965132120238]]), np.array([[0.000000000000001, 0.8164965132120238]])),
        "Outside Right Slant": (np.array([[0.8, 0.2]]), np.array([[0.355934, 0.2]])),
        "Outside Base-Left": (np.array([[-0.8, -0.8]]), np.array([[-grid.TR, grid.ΛF]])),
        "Apex Vertex": (np.array([[0.0, 1.0]]), np.array([[0.0, grid.ΛC]])),
        "Base-Right Vertex": (np.array([[0.8, -0.5]]), np.array([[grid.TR, grid.ΛF]])),
        "Base-Left Vertex": (np.array([[-0.8, -0.5]]), np.array([[-grid.TR, grid.ΛF]])),
    }

    for name, (input_xy, expected_xy) in test_cases.items():
        # Ensure the test uses the user's clamp function name
        result_xy = grid.clamp(input_xy.copy(), mode=1)
        assert_allclose(result_xy, expected_xy, atol=1e-5, err_msg=f"Case '{name}' failed")


def test_clamp_down_mode(const_grid):
    """Tests the clamping logic for the DOWN (V) triangle."""
    grid = const_grid
    test_cases = {
        "Inside": (np.array([[-0.1, -0.1]]), np.array([[-0.1, -0.1]])),
        "Pole": (
            np.array([[0.000000000000001, -0.8164965132120238]]), np.array([[0.000000000000001, -0.8164965132120238]])),
        "Outside Apex": (np.array([[0.1, -1.0]]), np.array([[0.0, grid.VF]])),
        "Outside Left Slant": (np.array([[-0.8, 0.2]]), np.array([[-0.586875, 0.2]])),
        "Outside Base-Right": (np.array([[0.8, 0.8]]), np.array([[grid.TR, grid.VC]])),
        "Apex Vertex": (np.array([[0.1, -1.0]]), np.array([[0.0, grid.VF]])),
        "Base-Right Vertex": (np.array([[0.8, 0.8]]), np.array([[grid.TR, grid.VC]])),
        "Base-Left Vertex": (np.array([[-0.8, 0.8]]), np.array([[-grid.TR, grid.VC]])),
    }

    for name, (input_xy, expected_xy) in test_cases.items():
        result_xy = grid.clamp(input_xy.copy(), mode=0)
        assert_allclose(result_xy, expected_xy, atol=1e-5, err_msg=f"Case '{name}' failed")


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


def test_reg_valid_point(reg_grid):
    """
    Tests the address generation for a known coordinate that should succeed.
    """
    # Inputs for a known address
    x = np.array([0.278558759260])
    y = np.array([0.29386255474])
    mode = np.array([0])  # DOWN mode
    expected_address = np.array([0x49, 42, 22, 62, 37, 38,
                                 73, 33, 37, 53, 37, 62, 58,
                                 43, 73, 42, 22, 58, 33, 53,
                                 38, 43, 53, 38, 53, 33, 57,
                                 52, 57, 42, 42, 62, 42, 37, 37, 0x2b])  # Includes reified root
    # Generate the address
    result_address = reg_grid.ugc_regions(x, y, mode)

    # Check that the first 8 regions of the generated address are correct
    assert_array_equal(result_address[0, :8], expected_address[:8])


def test_ugc_regions_invalid_point(reg_grid):
    """
    Tests the address generation for a coordinate that is out of bounds.
    """
    # Inputs for a point far outside the grid
    x = np.array([10.0])
    y = np.array([10.0])
    mode = np.array([1])  # UP mode

    # Generate the address
    result_address = reg_grid.ugc_regions(x, y, mode, depth=5)

    # The second region (index 1) should be marked as invalid because the
    assert result_address[0, 1] == reg_grid.invalid_ugc


def display(fwd_history, rev_history):
    """
    Displays the forward and reverse history side-by-side in a table.
    """
    # 1. Concatenate the two history arrays side-by-side
    combined_history = np.hstack((fwd_history, rev_history))

    # 2. Define clear column names for the table
    columns = [
        'dx', 'dy', 'dm', 'ix', 'iy', 'im'
        # 'FWD:', 'uri', 'x', 'y', 'off_x', 'off_y',
        # 'REV:', 'uri', 'x', 'y', 'off_x', 'off_y'
    ]

    # 3. Create a pandas DataFrame
    df = pd.DataFrame(combined_history, columns=columns)

    # Optional: Set display options for better viewing
    pd.set_option('display.precision', 25)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 5000)
    pd.set_option('display.width', None)
    # 4. Print the DataFrame
    print()
    print(df)


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
    # Encode the coordinate to a URI address.
    uri_address = reg_grid.ugc_regions(initial_x, initial_y, initial_mo)

    # Decode the URI address back to a coordinate.
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

    rnm = rel_grid.neighbours(reg01)
    assert_array_equal(rnm, ref01)
    npo = rel_grid.neighbours(reg02)
    assert_array_equal(npo, ref02)
