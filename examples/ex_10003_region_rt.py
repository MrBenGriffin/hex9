"""
Part of the H9 project -
This round-trips region encode/decode
and shows the maximum difference after 50 cycles
"""
import numpy as np
from hhg9 import Points, Registrar
from hhg9.h9.region import xy_regions, regions_xy


def ulps(a, b):
    ai = a.view(np.int64)
    bi = b.view(np.int64)
    ai ^= (ai >> 63) & 0x7fffffffffffffff
    bi ^= (bi >> 63) & 0x7fffffffffffffff
    return np.abs(ai - bi)


if __name__ == '__main__':
    # seed = 1234512
    # samples = 10000
    # layers = 36  # 36 reaches mathematical limits on 64-bit floats.
    # np.random.seed(seed)

    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    samples = np.array([
        [0.707106781029412823, -0.408248290463863128],
        [0.707106781029412712, -0.408248290463863128],
        [-0.707106781029412934, 0.408248290463863073],
        [0.000000000007909877, -0.816496580914025771]
        ])
    cmp = np.array([(1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, 1, 1)])
    bpts = Points(samples, b_oct, components=cmp)
    _, mode = bpts.cm()
    xy = bpts.coords.copy()
    for _ in range(50):
        rg = xy_regions(xy, mode, 38)  # encode (layers depth chosen internally)
        xym = regions_xy(rg)  # decode -> (x, y, mode)
        xy = xym[:, :2]
        mode = xym[:, 2].astype(np.uint8)

    # Compare to original
    xy_ref = bpts.coords.astype(np.float64, copy=False)
    xy_err = np.abs(xy_ref - xy)
    m_err = (bpts.cm()[1].astype(np.uint8) != mode)

    # L∞ errors
    ex = np.max(xy_err[:, 0])
    ey = np.max(xy_err[:, 1])
    em = np.any(m_err)
    print(ex, ey, float(em))

    # Stronger: fail if anything drifts beyond a whisper above eps
    import numpy.testing as npt

    npt.assert_allclose(xy, xy_ref, rtol=0, atol=1e-15)
    assert not em, "mode mismatch in round-trip"

    x_u, y_u = ulps(xy_ref[:, 0], xy[:, 0]).max(), ulps(xy_ref[:, 1], xy[:, 1]).max()
    print(f'Maximum ULP deviation: x:{x_u}, y:{y_u}')
