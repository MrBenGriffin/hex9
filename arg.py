from h9 import H9


if __name__ == '__main__':
    h9 = H9()
    a1 = H9.h9_hh8  # dicts
    # vx = list(a1.values())
    # vx.sort()
    rx0 = {a1[k]: k for k in a1.keys()}

    print(rx0)

