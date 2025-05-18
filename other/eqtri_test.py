import numpy as np
# import math


def tx_valid(tx, tolerance=1e-6):
    v1, v2, v3 = tx
    # Calculate the dot products
    dot12 = np.dot(v1, v2)
    dot23 = np.dot(v2, v3)
    dot31 = np.dot(v3, v1)

    # Calculate the angular distances
    theta12 = np.arccos(dot12)
    theta23 = np.arccos(dot23)
    theta31 = np.arccos(dot31)

    # 72° = 1.256637061435917r

    # Check if all angular distances are equal within a tolerance
    return np.abs(theta12 - theta23) < tolerance and \
        np.abs(theta23 - theta31) < tolerance and \
        np.abs(theta31 - theta12) < tolerance


if __name__ == '__main__':
    vx = [
        [-0.41468222, 0.65596241,  0.63067581],
        [0.42015243, 0.07814525,  0.90408255],
        [0.51883673, 0.83542038,  0.18133184],
        [0.99500944, -0.09134780, 0.04014717],
        [0.35578140, -0.84358000, 0.40223423],
        [-0.51545596, -0.38171689, 0.76720099],
        [0.41468222, -0.65596241, -0.63067581],
        [-0.42015243, -0.07814525, -0.90408255],
        [-0.51883673, -0.83542038, -0.18133184],
        [-0.99500944, 0.09134780, -0.04014717],
        [-0.35578140, 0.84358000, -0.40223423],
        [0.51545596, 0.38171689, -0.76720099]
    ]
    sides = [
        [0, 2, 1], [2, 0, 10], [10, 9, 7], [9, 10, 0], [3, 1, 2], [1, 3, 4],
        [1, 5, 0], [5, 1, 4], [3, 6, 4], [4, 8, 5], [8, 4, 6], [6, 7, 8], [10, 11, 2], [7, 11, 10], [2, 11, 3],
        [6, 3, 11], [7, 6, 11], [9, 0, 5], [9, 5, 8], [8, 7, 9]]

    # Example usage
    for side in sides:
        trx = vx[side[0]], vx[side[1]], vx[side[2]]
        print(tx_valid(trx))  # Output: True or False

    nie = tuple([
        [0.0, -0.5257311121191336, 0.85065080835204],
        [0.0, 0.5257311121191336, 0.85065080835204],
        [0.85065080835204, 0.0, 0.5257311121191336]
    ])
    print(tx_valid(nie))
    #  Icosahedron drawn using golden ratio.
    # ±(1,±𝛾, 0),±(0, 1,±𝛾),±(±𝛾, 0, 1)
    print('now do icosphere')
    π = np.pi
    𝜑 = 2.0 * np.cos(π / 5.0)
    ico_vertices = np.array([
        (0, -1, -𝜑), (0, -1, +𝜑), (0, +1, -𝜑), (0, +1, +𝜑),
        (-1, -𝜑, 0), (-1, +𝜑, 0), (+1, -𝜑, 0), (+1, +𝜑, 0),
        (-𝜑, 0, -1), (-𝜑, 0, +1), (+𝜑, 0, -1), (+𝜑, 0, +1)
    ])
    ico_sides = [  # The order here is arbitrary and does not even represent cw/ccw triangle orientation.
        (2, 10, 7), (10, 11, 7), (11, 6, 1), (6, 10, 0), (0, 6, 4), (4, 6, 1), (6, 1, 11),
        (9, 4, 1), (9, 1, 3), (9, 3, 5), (5, 9, 8), (9, 8, 4), (8, 4, 0), (2, 0, 10),
        (8, 0, 2), (2, 8, 5), (5, 2, 7), (7, 5, 3), (3, 11, 7), (1, 3, 11)
    ]
    ico = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    for side in ico_sides:
        trx = ico[side[0]], ico[side[1]], ico[side[2]]
        print(tx_valid(trx))  # Output: True or False

    # google_ll = {(x, y, z): tuple([np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))), np.degrees(np.arctan2(y, x))]) for x, y, z in ico}
    # print(google_ll)
    # using this ll calculation, the ico_points can be mapped to

#     {
# 0   (0.0, -0.5257311121191336, -0.85065080835204): (-58.282525588538995, -90.0),    Tierra del Fuego
# 1   (0.0, -0.5257311121191336, 0.85065080835204): (58.282525588538995, -90.0),      Hudson Bay
# 2   (0.0, 0.5257311121191336, -0.85065080835204): (-58.282525588538995, 90.0),      McDonald Islands
# 3   (0.0, 0.5257311121191336, 0.85065080835204): (58.282525588538995, 90.0),        Russia
# 4   (-0.5257311121191336, -0.85065080835204, 0.0): (0.0, -121.717474411461),        West Galapagos
# 5   (-0.5257311121191336, 0.85065080835204, 0.0): (0.0, 121.717474411461),          Indonesia
# 6   (0.5257311121191336, -0.85065080835204, 0.0): (0.0, -58.282525588538995),       Amazon
# 7   (0.5257311121191336, 0.85065080835204, 0.0): (0.0, 58.282525588538995),         Seychelles/Arabian Sea
# 8   (-0.85065080835204, 0.0, -0.5257311121191336): (-31.717474411461005, 179.995),  New Zealand
# 9   (-0.85065080835204, 0.0, 0.5257311121191336): (31.717474411461005, 179.9995),   Midway
# 10  (0.85065080835204, 0.0, -0.5257311121191336): (-31.717474411461005, 0.0),       Cape Town
# 11  (0.85065080835204, 0.0, 0.5257311121191336): (31.717474411461005, 0.0)          Algeria
# }

# and the sides as follows:
# 2,10,7   : Madagascar
# 10,11,7  : Central Africa
# 11,6,10  : Liberia
# 6,10,0   : Paraguay
# 0,6,4    : Peru
# 4,6,1    : Cuba/USA
# 6,1,11   : North Atlantic
# 9,4,1    : San Francisco
# 9,1,3    : Bering Strait
# 9,3,5    : Japan
# 5,9,8    : PNG
# 9,8,4    : Kiribati
# 8,4,0    : SPO // French Polynesia
# 2,0,10   : Atlantic Antarctica
# 8,0,2    : Pacific Antarctica
# 2,8,5    : Australia
# 5,2,7    : Indian Ocean
# 7,5,3    : China
# 3,11,7   : Iran
# 1,3,11   : UK


