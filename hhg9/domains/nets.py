# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Octahedral net layouts.
"""
net_layouts = {
    # the final digit is a rotation measured in 60º
    # of each face (each starts with the apex at a pole)
    # for flipped, this should be odd numbers
    # for not flipped this should be even numbers.
    'mortar': {
        'width':  3.5,  # number of full triangles across
        'height': 3,  # number of full triangles up.
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
    'diamonds': {
        'width': 4,   # number of full triangles across
        'height': 2,  # number of full triangles up.
        'grid': {
            (-1, -1, +1): (1., 4., 3),  # NWP; Rotation in odd = flip mode and rotate.
            (-1, -1, -1): (1., 2., 3),  # SWP
            (+1, -1, +1): (3., 4., 0),  # NWA
            (+1, -1, -1): (3., 2., 6),  # SWA
            (+1, +1, +1): (5., 4., 3),  # NEA
            (+1, +1, -1): (5., 2., 3),  # SEA
            (-1, +1, +1): (7., 4., 0),  # NEP
            (-1, +1, -1): (7., 2., 6),  # SEP
        }
    },
    'windmill': {
        'flipped': True,
        'width': 3.5,   # number of full triangles across
        'height': 3,  # number of full triangles up.
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
        'width': 3.5,   # number of full triangles across
        'height': 3,  # number of full triangles up.
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
        'width': 3,   # number of full triangles across
        'height': 3,  # number of full triangles up.
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
    'c_butterfly': {  # Cahill butterfly (without necessary the 15º shift).
        'flipped': False,
        'width': 3,   # number of full triangles across
        'height': 3,  # number of full triangles up.
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
