"""
Compute authalic log-density ℓ = log(area_scale) for points in one octant.
"""
import numpy as np
from hhg9 import Registrar, Points


def get_density(reg: Registrar, pts: Points, octant_id: int = 0):
    """
    Compute authalic log-density ℓ = log(area_scale) for points in one octant.
    :param reg: H9 Registrar
    :param pts: Points object to compute density for.
    :param octant_id: octant index (0) for matrix application
    :return: authalic log-density ℓ
    """
    if not (0 <= octant_id < 8):
        raise ValueError(f"octant_id must be in [0, 7]")
    if not isinstance(pts, Points):
        raise ValueError("Points must be a Points object in c_oct, b_oct, or s_oct domain")
    ake = reg.projection('oct_ell')
    b_oct = reg.domain('b_oct')
    cmp = b_oct.signs_by_id[octant_id]
    face = b_oct.signs[cmp]
    prj = b_oct.projs[face]
    q = prj.matrix.T @ prj.orient
    e1_xyz = q[:, 0]  # 3-vector
    e2_xyz = q[:, 1]  # 3-vector
    pts_to_use = None
    dom = pts.domain
    match dom.name:
        case 'c_oct':
            pts_to_use = pts
        case 'b_oct':
            pts_to_use = reg.project(pts, ['b_oct', 'c_oct'])
        case 's_oct':
            pts_to_use = reg.project(pts, ['s_oct', 'b_oct', 'c_oct'])
        case _:
            raise ValueError("Points must be in c_oct, b_oct, or s_oct domain")
    j_pts = ake.jacobian(pts_to_use.coords)
    det = np.linalg.det(j_pts)
    kmin = np.argmin(det)
    kmax = np.argmax(det)
    mn, mx = pts.coords[kmin], pts.coords[kmax]
    v1 = j_pts @ e1_xyz  # (N, 3)
    v2 = j_pts @ e2_xyz  # (N, 3)
    cross = np.cross(v1, v2)  # (N, 3)
    area_scale = np.linalg.norm(cross, axis=1)  # (N,)
    area_clip = np.clip(area_scale, 1e-20, None)
    return np.log(area_clip)  # authalic log-density ℓ

