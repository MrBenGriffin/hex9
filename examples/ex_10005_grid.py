"""
Part of the H9 project -
Various geometric/coordinates tests
"""
import numpy as np


def _in_up(ẋ, y, λf, λc):
    """Return true where ẋ, y is in up supercell"""
    return (λf <= y) & (y <= λc - np.abs(ẋ))


def _in_dn(ẋ, y, vf, vc):
    """Return true where ẋ, y is in down supercell"""
    return (vf + np.abs(ẋ) <= y) & (y <= vc)


def uv(n, p, h, m):
    """Convert (n,p,h,m) to (u,v)"""
    return n + p - 3, 8 - (3 * h) - m


def nph(u, v, hn, m):
    """Convert (u,v, h_numerator, m) to (n,p,h)"""
    h = hn // 3
    n = (u - h - m + 6) >> 1
    p = (u + h + m) >> 1
    return n, p, h


if __name__ == '__main__':

    from hhg9.algorithms.id_packing import compose_luts
    encode, decode, *_ = compose_luts([6, 4, 4])  # id lut = 96 size.

    # Define the Cell/Supercell constants, modes, and membership LUTs
    # Decode has several non-geometrics (04,05,06, etc.) which we need to mask.
    # (1)√ identify the 42 valid geometric ids.
    # (2)√ to determine the mode of each geometric cell
    # Because the H-layer is downward, rather than traditional
    # upward, we invert the parity of each address accordingly.
    # (3)√ To identify the Lattice-related U,V indices of each
    # of the 42 geometrically valid cell that determine their triangle-centroids.
    # and generate luts for cell<->uv
    # (3) To identify supercell membership for each supercell mode -
    # (this may be done via the _in_up / _in_dn methods) for each geometric cell.
    # (4) To identify the in_scope membership - the logical OR of (3) across both modes.
    # (5) Maybe to identify the 3 groups of 3 C2 memberships for each supercell mode.

    #    \     \     \    /     /     /
    #  00 \  01 \  02 \03/ 07  / 0b  / 0f
    # _____\_____\ ____\/____ /____ /______
    #       \     \    /\    /     /
    #     10 \  11 \12/16\17/ 1b  / 1f
    # ________\_____\/____\/____ /_________
    #          \    /\    /\    /
    #        20 \21/25\26/2a\2b/ 2f
    # ___________\/____\/____\/____________
    #            /\    /\    /\
    #        30 /34\35/39\3a/3e\ 3f
    # _________/____\/____\/____\__________
    #         /     /\    /\     \
    #     40 /  44 /48\49/4d\ 4e  \ 4f
    # ______/____ /____\/____\ ____\________
    #      /     /     /\     \     \
    #  50 / 54  / 58  /5c\  5d \  5e \ 5f

    # Unpack classifier thresholds
    # h_idx, p_idx, n_idx = decode.astype(int).T  # Gather decode (96,3) into h,p,n arrays.
    # m_idx = ((h_idx + p_idx + n_idx) & 1) ^ 1  # 0=V, 1=Λ

    # from geometry.
    r3 = np.sqrt(3.0, dtype=np.float64)  # √3 Because equilateral triangles.
    th = np.sqrt(6.0, dtype=np.float64) / 2.0  # This one has an edge of √2, so it's height is √6/2
    tw = 2 * th / r3  # Edge length of the *full* barycentric triangle: √2
    fu, fv = tw / 6., th / 9.  # horizontal/vertical unit multipliers.
    n_levels = 3
    p_levels = 3

    data = {}
    # generate the full u,v grid co-ordinates.
    for u in range(-6, 6):
        for v in range(-9, 9):
            m = ((u % 2) + ((v // 3) % 2)) % 2
            hn = (8 - v - m)
            if hn % 3 == 0:   # This is on a barycentre.
                index = nph(u, v, hn, m)
                idx = np.array(index, dtype=np.int8)
                idx[idx < 0] = 0    # clip to fit encode/decode arrays.
                n, p, h = idx
                n = min(n, n_levels)
                p = min(p, p_levels)
                d = encode[h][p][n]
                if d not in data:
                    data[d] = [(u, v), (fu*u, fv*v)]
                else:
                    c = data[d][0]
                    if c != min(c, (u, v), key=lambda q: q[0] ** 2 + q[1] ** 2):
                        data[d] = [(u, v), (fu * u, fv * v)]
    reference_points = data
    assert len(reference_points) == 42, len(reference_points)
    # print(reference_points)

    # We now have u,v coordinates for each cell, as well as centroids for each.
    # We can identify those that are in_up and in_down supercells.
    # Our work is now done.

