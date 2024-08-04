class HexUtil:
    @staticmethod
    def to_offs(hc: tuple) -> tuple:
        q, r, s = hc
        x = q
        y = r + (q - (q & 1)) // 2   # This from qrs to offset 'odd-q'
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

    @staticmethod
    def to_nested_axial(hcl: tuple):
        # This from offset 'odd-q' to qr
        # col = x, row=y
        x, ny, lv = hcl
        h = 3 ** lv
        y = -ny
        q = x
        r = y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        return h*q, h*r
