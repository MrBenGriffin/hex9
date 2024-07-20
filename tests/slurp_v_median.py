import numpy as np
import math


def slerp(a, b, t):
    ab_dot = np.dot(a, b)
    th = math.acos(ab_dot)
    s = math.sin(th)
    j = math.sin((1. - t) * th)
    k = math.sin(t * th)
    return (j*a + k*b) / s


def p2_median(i, j):
    return np.array([
        # np.mean([i, j, k], axis=0),  # centre.
        np.mean([i, i, j], axis=0),  # iij.
        np.mean([i, j], axis=0),  # iij.
        np.mean([i, j, j], axis=0),  # ijj.
    ])


def p2_slurp(i, j):

    return np.array([
        slerp(i, j, 1./3.),
        slerp(i, j, 0.5),
        slerp(i, j, 2./3.),
    ])


def xyz_ll(xyz: tuple) -> tuple:
    x, y, z = xyz
    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))


def p2_diff(i, j):
    m = p2_slurp(i, j)
    s = p2_median(i, j)
    m0 = xyz_ll(i)
    m1 = xyz_ll(s[0])
    m2 = xyz_ll(s[1])
    m3 = xyz_ll(s[2])
    m4 = xyz_ll(j)
    m5 = xyz_ll(m[0])
    m6 = xyz_ll(m[1])
    m7 = xyz_ll(m[2])
    print(m0, m1, m3, m4)

    return tuple(s - m)


if __name__ == '__main__':
    # make two sets of 100 random unit vectors suitable for a sphere and normalise.
    n = 10
    u = np.random.normal(size=(n, 3))
    v = np.random.normal(size=(n, 3))

    for idx in range(n):
        a = u[idx] / np.linalg.norm(u[idx], axis=0)
        b = v[idx] / np.linalg.norm(v[idx], axis=0)
        p2_diff(a, b)
        # print(p2_diff(a, b))
#         1,031.34, 17,924.10 18,955.44 km
