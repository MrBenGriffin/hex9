class HexUtil:
    @classmethod
    def to_offs(cls, hc: tuple) -> tuple:
        q, r = hc
        x = q
        y = r + (q - (q & 1)) // 2   # This from axial to offset 'odd-q'
        return x, -y

    @staticmethod
    def to_cube(hc: tuple):
        # This from offset 'odd-q' to qrs
        # col = x, row=y
        x, ny = hc
        y = -ny
        q = x
        r = y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        s = -q - r
        return q, r, s

    @staticmethod
    def to_axial(hc: tuple):
        # This from offset 'odd-q' to qr
        # col = x, row=y
        x, ny = hc
        y = -ny
        q = x
        r = y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        return q, r

    @classmethod
    def to_nested_axial(cls, hcl: tuple):
        # This from offset 'odd-q' to qr
        # col = x, row=y
        x, ny, lv = hcl
        return cls.to_axial((x, ny))
        # h = 3 ** lv
        # y = -ny
        # q = x + (3 ** (lv-1) if (lv > 0) else 0)  # TODO -lv values.
        # r = y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        # return h*q, h*r


if __name__ == '__main__':
    # axial -6, 0 returns -7:-6. offs -6, 0 returns axial -6, 3
    print(HexUtil.to_offs((-2, -1)))
    # print(HexUtil.to_axial((-2, 1)))
