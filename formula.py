from halfhex import HalfHexFn

# def solve():
#     x, s = sp.symbols('x s')
#     # big R / side length of hex in terms of small radius r. (ie, r = half-height)
#     # This is zero - centred.
#     f = Piecewise(
#         (sp.sqrt(3) * s + sp.sqrt(3) * x, (x > -s) & (x < -0.5 * s)),
#         (sp.sqrt(3) * s * 0.5, sp.Abs(x) <= 0.5 * s),
#         (sp.sqrt(3) * s - sp.sqrt(3) * x, (x > 0.5 * s) & (x < s))
#     )
#     # This is positive zero.
#     # f = Piecewise(
#     #     (sp.sqrt(3) * x, (x > 0.) & (x < 0.5 * s)),
#     #     (sp.sqrt(3) * 0.5 * s, (x >= 0.5 * s) & (x < 1.5 * s)),
#     #     ((sp.sqrt(3) * 0.5 * s)-sp.sqrt(3)*(x-1.5 * s), (x >= 1.5 * s) & (x < 2.0 * s))
#     # )
#     # hx = sp.integrate(f, x)
#     hxi = sp.simplify(f)
#     pprint(hxi, use_unicode=True)
#     # pprint(hxi, use_unicode=True)
#     # vertical f(y) between top and and bottom of hex.
#     # 0 ...
#     # a_left = sp.integrate(2. * r, (x, -sp.sqrt(3.) / 2. * s, x))
#     # a_right = sp.integrate(2. * r, (x, x, sp.sqrt(3.) / 2. * s))
#     # a_left_simplified = sp.simplify(a_left)
#     # a_right_simplified = sp.simplify(a_right)
#     # print(a_left_simplified, a_right_simplified)
#


if __name__ == '__main__':
    hf = HalfHexFn(100.)
    print(hf.area_props_at_lst([27, 61]))

