from h9 import H9
import math


def print_lut():
    fn = (lambda a, b: (a - b) % 3)
    for n in range(9):
        print(f'?{n}')
        g, i = divmod(n, 3)  # g = 0/1/2 for 0..2/3..5/6..8
        for p in range(9):
            v = [f'{p}X{n}X', f'{p}a{n}', f'{p}b{n}', f'{p}X{n}Y']
            idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
            rx = v[idx] if g != 2 else v[3] if idx == 0 else v[0]
            print(f'{p}{n}={rx}')
        print('')


def set_lut():
    long_winded_variable = 'XabY'
    fn = (lambda a, b: (a - b) % 3)
    lut = {}
    for n in range(9):
        g, i = divmod(n, 3)  # g = group: 012/345/678
        for p in range(9):
            idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
            rx = long_winded_variable[idx] if g != 2 else long_winded_variable[3] if idx == 0 else long_winded_variable[0]
            lut[(p, n)] = rx
    return lut


def bit(lut, p, c, ab: str = '') -> tuple:
    dx = {'a': 'b', 'b': 'a'}
    if c not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        return ab, f'{p}{ab}'
    else:
        r = lut[(p, c)]
        f = r if r not in 'XY' else ab if r == 'X' else dx[ab] if ab in dx else ''
        return f,  f'{p}{f}'


def c2(lut, addr: str):
    p, c = None, None
    ab = ''
    result = []
    for ph in addr[::-1]:
        c = p
        if ph in 'ab':
            ab = ph
            continue
        p = int(ph)
        ab, cx = bit(lut, p, c, ab)
        result.append(cx)
    return ''.join(result[::-1])


# Given a string eg '51313181b' expand to its full/canonical address.
def canon(lut, addr: str):
    return c2(lut, addr)


def b3(i):
    r, n, i = ('', '', i) if i >= 0 else ('', '-', -i)
    while i > 0:
        i, k = divmod(i, 3)
        r += f'{k}'
    return n + r[::-1]


def dm3(x):
    if x == 0:
        return 0, 0, 0
    sx, ax = int(math.copysign(1, x)), int(abs(x))
    dv, rm = divmod(ax, 3)
    return dv*sx, rm*sx, sx


if __name__ == '__main__':
    # hx2hh = set_lut()
    # print(canon(hx2hh, '0768b'))  # 4b1b8b4a
    # print('4b1b8b4a')
    # print(c2(hx2hh, '0730a'))     # 4b1b8b4a == 4b1b8b4a
    for i in range(-15, 15):
        print(f'{i}: {dm3(i)}')


    # 1827360847566784b
    # for z in [(None, 4), (7, 7), (7, 8), (3, 5), (0, 0), (3, 4), (5, 2), (7, 0), (8, 3)]:
    #     p0, c0 = z
    #     for b in ['a', 'b', '']:
    #         print((p0, f'{c0}{b}'), '=>', bit(hx2hh, p0, c0, b))
    # # print_lut()
