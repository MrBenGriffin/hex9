# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from hhg9.h9.grid import hex_props

if __name__ == '__main__':
    for level in range(32):
        #             area: area in m^2
        #             side: side length s (aka) circumradius: (vertex-centre)
        #             inradius: r (flat-centre)
        #             flat_diameter: 2*r aka 'height'
        #             point_diameter: 2*side
        area, side, inr, flat_d, pt_d = hex_props(level)
        print(f'level:{level} area: {area}m^2, side:{side}, inr:{inr}, flat_d:{flat_d}, pt_d:{pt_d}')
