from h9 import H9


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
    v = 'XabY'
    fn = (lambda a, b: (a - b) % 3)
    lut = {}
    for n in range(9):
        g, i = divmod(n, 3)  # g = group: 012/345/678
        for p in range(9):
            idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
            rx = v[idx] if g != 2 else v[3] if idx == 0 else v[0]
            lut[(n, p)] = rx
    return lut


# Given a string eg '51313181b' expand to its full/canonical address.
def canon(lut, addr: str):
    n = 9
    ab = ''
    result = ''
    fn = (lambda a, b: a if a not in 'XY' else b if a == 'X' else 'a' if a == 'b' else 'b')
    for c in addr[::-1]:
        if c in 'ab':
            ab = c
            continue
        p = int(c)
        if n == 9:
            result += f'{ab}{p}'
        else:
            f = fn(lut[(n, p)], ab)
            result += f'{f}{p}'
            ab = f
        n = p
    return result[::-1]


if __name__ == '__main__':
    hx2hh = set_lut()
    print(canon(hx2hh, '3187053527a'))
    print_lut()
