# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
The domains recognised in Hex9.
"""
from .general_gcd import GeneralGCD  # g_gcd
from .radians_gcd import RadiansGCD  # r_gcd
from .spherical_cartesian import SphericalCartesian  # c_sph
from .ellipsoid_cartesian import EllipsoidCartesian  # c_ell
from .octahedral_simplex import OctantSimplex, OctahedralSimplex  # s_oct
from .octahedral_cartesian import OctantCartesian, OctahedralCartesian  # c_oct
from .octahedral_barycentric import OctantBarycentric, OctahedralBarycentric  # b_oct
from .octahedral_net import OctantNet, OctahedralNet  # n_oct
from .plate_pixel import PlatePixel  # p_pix
from .net_pixel import NetPixel  # n_pix

__all__ = [
    "GeneralGCD",
    "RadiansGCD",
    "SphericalCartesian",
    "OctantCartesian", "OctahedralCartesian",
    "OctantBarycentric", "OctahedralBarycentric",
    "OctantSimplex", "OctahedralSimplex",
    "OctantNet", "OctahedralNet",
    "PlatePixel", "NetPixel",
    "EllipsoidCartesian"
]

