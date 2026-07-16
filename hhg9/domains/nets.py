# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Octahedral net layouts.
"""
net_layouts = {
    # the final digit is a rotation measured in 60º
    # of each face (each face in b_oct has the apex
    # at a pole, and flat side at the equator)
    # width/height are in Lattice U/V
    # U=6*W, V=9*H
    'mortar': {
        'width':  21,  # 3.5 full triangles across
        'height': 27,  # 3 full triangles up.
        'grid': {
            (+1, +1, +1): (3., 4., 3),  # NEA
            (-1, +1, +1): (4., 5., 5),  # NEP
            (+1, -1, +1): (2., 5., 1),  # NWA
            (-1, -1, +1): (2., 7., 5),  # NWP
            (+1, +1, -1): (3., 2., 3),  # SEA
            (-1, +1, -1): (5., 4., 5),  # SEP
            (+1, -1, -1): (1., 4., 1),  # SWA
            (-1, -1, -1): (6., 5., 3)   # SWP
        }
    },
    'rhombus': {
        'width': 19,     # should be 3 1/6 full triangles across 18+1
        'height': 15,    # should be 1 2/3 full triangles up. 9+6=15
        'grid': {
            (+1, +1, +1): (3., 2., 3),  # NEA
            (-1, +1, +1): (4., 3., 5),  # NEP
            (+1, -1, +1): (2., 3., 1),  # NWA
            (-1, -1, +1): [(2., 5., 5), (4., 0., 4), (2., 0, -4),  (0, 0, 0)],    # NWP North
            (+1, +1, -1): [(3., 0., 3), (0., 0., 0), (2., 0, -2), (-2., 0, -4)],  # SEA South
            (-1, +1, -1): (5., 2., 5),  # SEP
            (+1, -1, -1): (1., 2., 1),  # SWA
            (-1, -1, -1): [(6., 3., 3), (0., 0., 0), (-6., 0, 0), (0, 0, 0)],   # SWP
        }
    },
    'oct_rhombus': {
        'width': 30,  # number of full triangles across * 6
        'height':27,  # number of full triangles up * 9
        'grid': {
            (-1, -1, -1): (2., 4., 4),  # SWP
            (-1, -1, +1): (3., 5., 4),  # NWP
            (+1, -1, +1): (4., 4., 0),  # NWA
            (+1, +1, +1): (5., 5., 2),  # NEA
            (+1, +1, -1): (6., 4., 2),  # SEA
            (-1, +1, -1): (7., 5., 0),  # SEP

            (+1, -1, -1): (2., 2., 2),  # SWA
            (-1, +1, +1): (7., 7., 0),  # NEP

        }
    },
    'diamonds_native': {
        'width': 24,  # number of full triangles across * 6
        'height': 18,  # number of full triangles up * 9
        'grid': {
            (-1, -1, +1): (1., 2., 0),  # NWP
            (-1, -1, -1): (1., 4., 0),  # SWP

            (+1, -1, +1): (3., 4., 0),  # NWA
            (+1, -1, -1): (3., 2., 0),  # SWA

            (+1, +1, +1): (5., 2., 0),  # NEA
            (+1, +1, -1): (5., 4., 0),  # SEA

            (-1, +1, +1): (7., 4., 0),  # NEP
            (-1, +1, -1): (7., 2., 0),  # SEP
        }
    },
    'diamonds': {
        'width': 24,   # number of full triangles across * 6
        'height': 18,  # number of full triangles up * 9
        'grid': {
            (-1, -1, +1): (1., 4., 3),  # NWP; Rotation in odd = flip net_mode and rotate.
            (-1, -1, -1): (1., 2., 3),  # SWP
            (+1, -1, +1): (3., 4., 0),  # NWA
            (+1, -1, -1): (3., 2., 0),  # SWA
            (+1, +1, +1): (5., 4., 3),  # NEA
            (+1, +1, -1): (5., 2., 3),  # SEA
            (-1, +1, +1): (7., 4., 0),  # NEP
            (-1, +1, -1): (7., 2., 0),  # SEP
        }
    },
    'windmill_pacific': {
        'width': 21,  # number of full triangles across
        'height': 27,  # number of full triangles up.
        'grid': {
            (-1, -1, +1): (4., 5., 2),  # NWP 3
            (-1, -1, -1): (5., 4., 2),  # SWP 7
            (+1, -1, +1): (4., 7., 4),  # NWA
            (+1, -1, -1): (6., 5., 0),  # SWA 6
            (+1, +1, +1): (2., 5., 4),  # NEA 0
            (+1, +1, -1): (1., 4., 4),  # SEA 4
            (-1, +1, +1): (3., 4., 0),  # NEP 1
            (-1, +1, -1): (3., 2., 0),  # SEP 5
        }
    },
    'windmill': {
        'flipped': True,
        'width': 21,   # number of full triangles across
        'height': 27,  # number of full triangles up.
        'grid': {
            (-1, -1, +1): (3., 7., 5),  # NWP
            (-1, -1, -1): (1., 5., 3),  # SWP
            (+1, -1, +1): (3., 5., 1),  # NWA
            (+1, -1, -1): (2., 4., 1),  # SWA
            (+1, +1, +1): (4., 4., 3),  # NEA
            (+1, +1, -1): (4., 2., 3),  # SEA
            (-1, +1, +1): (5., 5., 5),  # NEP
            (-1, +1, -1): (6., 4., 5),  # SEP
        }
    },
    'turbine': {
        # Probably not the best tile. Sorry asia and S.Am!
        'flipped': False,
        'width': 21,   # number of full triangles across
        'height': 27,  # number of full triangles up.
        'grid': {
            (-1, -1, +1): (2., 5., 4),  # NWP
            (-1, -1, -1): (1., 4., 4),  # SWP
            (+1, -1, +1): (3., 4., 0),  # NWA
            (+1, -1, -1): (1., 2., 2),  # SWA
            (+1, +1, +1): (4., 5., 2),  # NEA
            (+1, +1, -1): (5., 4., 2),  # SEA
            (-1, +1, -1): (6., 5., 0),  # SEP
            (-1, +1, +1): (6., 7., 0),  # NEP

        }
    },
    'butterfly': {  # A suitable butterfly for no longitude shift.
        'flipped': False,
        'width': 18,   # number of full triangles across
        'height': 27,  # number of full triangles up.
        # 'theta': 2000, # global rotation in 15
        'grid': {
            (-1, -1, +1): (2., 5., 4),  # left wing
            (-1, -1, -1): (1., 4., 4),  #
            (+1, -1, +1): (3., 4., 0),  # left body
            (+1, -1, -1): (3., 2., 0),  #
            (+1, +1, +1): (4., 5., 2),  # right body
            (+1, +1, -1): (5., 4., 2),  #
            (-1, +1, +1): (4., 7., 4),  # right wing
            (-1, +1, -1): (5., 8., 4),  #
        }
    },
    'cylinder': {
        # Translation-tiling equatorial strip (the b_oct<->planar CRS net).
        # Every face is placed by a PURE translation (theta=0, matrix=I), so the
        # fold is just (X,Y) = (cx,cy) + t[oid].  Rows are by bary mode:
        # mode-0 faces (apex-down) on the lower row (gy=2), mode-1 (apex-up) on
        # the upper row (gy=4); each column is one equatorial-edge diamond.
        # Columns run in longitude order EA|EP|WP|WA (lon 0-90|90-180|180-270|
        # 270-360), so the left edge of EA and the right edge of WA are the same
        # lon-0 / A-axis meridian: X wraps modulo width, healing that one cut and
        # leaving all curvature at the 6 cone vertices (poles top/bottom edge, the
        # 4 equatorial vertices on the mid-line).
        'flipped': False,
        'width': 24,    # 4*sqrt2 in wi units; also the X-wrap modulus
        'height': 18,
        'wrap_x': 24,   # longitude modulus (== width); the A-axis meridian seam
        'grid': {
            (+1, +1, +1): (1., 2., 0),  # NEA  col EA (lon 0-90),   lower
            (+1, +1, -1): (1., 4., 0),  # SEA               upper
            (-1, +1, +1): (3., 4., 0),  # NEP  col EP (lon 90-180),  upper
            (-1, +1, -1): (3., 2., 0),  # SEP               lower
            (-1, -1, +1): (5., 2., 0),  # NWP  col WP (lon 180-270), lower
            (-1, -1, -1): (5., 4., 0),  # SWP               upper
            (+1, -1, +1): (7., 4., 0),  # NWA  col WA (lon 270-360), upper
            (+1, -1, -1): (7., 2., 0),  # SWA               lower
        }
    },
    'butterfly_cahill': {  # Cahill butterfly (without the necessary 15º shift).
        'flipped': False,
        'width': 18,   # number of full triangles across
        'height': 27,  # number of full triangles up.
        'grid': {
            (+1, +1, +1): (2., 5., 4),  # NEA LW
            (+1, +1, -1): (1., 4., 4),  # SEA LW
            (-1, +1, +1): (3., 4., 0),  # NEP LB
            (-1, +1, -1): (3., 2., 0),  # SEP LB
            (-1, -1, +1): (4., 5., 2),  # NWP RB
            (-1, -1, -1): (5., 4., 2),  # SWP RB
            (+1, -1, +1): (4., 7., 4),  # NWA RW
            (+1, -1, -1): (5., 8., 4),  # SWA RW
        }
    }
}
