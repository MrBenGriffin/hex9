import math


class HalfHexFn:
    # This is used for general methods over half-hexes.
    # Currently used by create_pixel_lut.
    # Provides the area values of a half-hex given the x offset(s)
    def __init__(self, side_length: float = 1.0):
        self.s = side_length
        self.r3 = math.sqrt(3)
        self.r3_2 = self.r3 * 0.5
        self.h = self.s * self.r3_2         # height of half_hex
        self.s_2 = self.s * 0.5
        self.t_a = self.h * self.s * 0.25  # triangle area is half*width*height = area_6
        self.full_area = self.t_a * 6.

    def y_at_x(self, x: float) -> float:
        w = x - self.s
        if (w > -self.s) & (w < -self.s_2):
            return self.r3 * (self.s + w)
        elif abs(w) <= self.s_2:
            return self.r3 * self.s_2
        elif (w > self.s_2) & (w < self.s):
            return self.r3 * (self.s - w)
        return math.nan

    def areas_at_x(self, x: float) -> tuple:
        w = x - self.s
        if (w > -self.s) & (w < -self.s_2):  # first quadrant.
            ls = self.r3_2 * pow(x, 2.)
            return ls, self.full_area - ls
        elif abs(w) <= self.s_2:
            ls = self.t_a + (x - self.s_2) * self.h
            return ls, self.full_area - ls
        elif (w > self.s_2) & (w < self.s):
            rx = self.r3_2 * pow(2.0 * self.s - x, 2.)
            return self.full_area - rx, rx
        return 0., self.full_area if x < 0. else self.full_area, 0.

    def area_props_at_x(self, x: float, n: float = 1.0) -> tuple:
        lp, rp = self.areas_at_x(x)
        return n * lp/self.full_area, n * rp/self.full_area

    def area_props_at_lst(self, lx: list) -> list:
        # return proportions of a half-hexagon split at varying points given by the ordered list lx.
        # will return n+1 values for the n values given in lx.
        remains = 1.0
        rx = []
        for x in lx:
            lp, remains = self.area_props_at_x(x, remains)
            rx.append(lp)
        rx.append(remains)
        return rx


if __name__ == '__main__':
    hf = HalfHexFn(100.)
    print(hf.area_props_at_lst([27, 61]))
