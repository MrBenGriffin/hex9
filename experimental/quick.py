import sys, numpy as np
sys.path.insert(0, '.')
from hhg9 import Registrar, Points

if __name__ == '__main__':
    reg = Registrar()
    ll_dom = reg.domain('g_gcd')  # lat lon.
    ll = Points(np.array([[55.94480923423753, -3.187282666241978]]), domain=ll_dom)
    by = reg.project(ll, ['g_gcd', 'b_oct'])
    b_oct = reg.domain('b_oct')
    print(by.coords)
    # This is not the same as the inverse of the warp.
    prev = b_oct.warp.undo(by.coords)
    print("not same:", prev)
    # This is the same as the inverse of the warp.
    br = reg.project(ll, ['g_gcd', 'c_ell', 'c_oct', 'b_raw'])
    print(br.coords)
    dr = reg.project(br, ['b_raw', 'b_oct'])
    print(dr.coords)
    rv = reg.project(dr, ['b_oct', 'b_raw', 'c_oct', 'c_ell', 'g_gcd'])
    print(rv.coords)


