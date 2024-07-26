import math


class ShapeFunctions:
    def __init__(self, side_length: float = 1.0):
        # Defaults to a square.
        self.s = side_length                                # This defaults to 1.
        self.max_x = self.s
        self.r3 = math.sqrt(3)                              # √3   = tan(60)
        self.r3_2 = self.r3 * 0.5                           # √3/2 = sin(60)
        self.r3_4 = self.r3 * 0.25                          # √3/4 used for areas
        self.h = self.s
        self.a = self.s ** 2

    def x_height(self, x: float) -> float:
        if 0 <= x <= self.s:
            return self.h
        return 0

    def area(self, x: float, y: float) -> float:
        return x * y

    # Given a point at x return the areas to left and right.
    def areas_at_x(self, x: float) -> tuple:
        y = self.x_height(x)
        if y > 0:
            xa = self.area(x, y)
            result = xa, self.a - xa
        elif x >= self.max_x:
            result = self.a, 0.
        else:
            result = 0., self.a
        return result

    def areas(self, lx: list, native: bool = False) -> list:
        # areas of triangle split at varying points given by the ordered list lx.
        # will return n+1 values for the n values given in lx.
        used = 0.0
        remains = 0.0
        result = []
        for x in lx:
            left_area, remains = self.areas_at_x(x)
            left_area -= used
            used = used + left_area
            result.append(left_area)
        result.append(remains)
        # normalise if needs be.
        return result if native else [x / self.a for x in result]


class TriangleFns(ShapeFunctions):
    # This is used for general methods over EquilateralTriangles.
    # Provides the area values of a half-hex given the x offset(s)
    def __init__(self, side_length: float = 1.0):
        super().__init__(side_length)
        self.max_x = self.s
        self.h = self.s * self.r3_2                         # height of equilateral triangle
        self.s_2 = self.s * 0.5                             # x offset of apex.
        self.a = self.s ** 2. * self.r3_4                   # area s^2*√3/4
        self.a_2 = self.a * 0.5                             # area to apex.

    # Given a triangle with leftmost point at origin,
    # & Given a point at x, Return its height.
    # It makes no difference if triangles is point up or down.
    def x_height(self, x: float) -> float:
        if 0 < x <= self.s_2:
            return x * self.r3              # This is trig.  TanΘ = O/A. √3=tan(60).
        if self.s_2 < x < self.s:
            return self.r3 * (self.s - x)  # if x = 90,return y at 10.
        return 0

    def area(self, x: float, y: float) -> float:  # return area of shape to left of x.
        if 0 < y <= self.h and 0 < x <= self.s:
            if x > self.s_2:
                return self.a - y * (self.s - x) * 0.5  # This is a RA triangle, not an equilateral
            else:
                return y * x * 0.5
        return 0.


class HalfHexFns(ShapeFunctions):
    # This is used for general methods over regular half-hexes.
    # Currently used by create_pixel_lut. This is good for hex pixel calculations
    def __init__(self, side_length: float = 1.0):
        super().__init__(side_length)
        self.s = side_length
        self.max_x = self.s * 2.
        self.h = self.s * self.r3_2        # height of half_hex
        self.q1 = self.s * 0.5
        self.q3 = self.s * 1.5
        self.t_a = self.h * self.s * 0.25  # triangle area is half*width*height = area_6
        self.a = self.t_a * 6.            # 3. * self.s ** 2. * self.r3_4 works too.

    def x_height(self, x: float) -> float:
        if 0. < x <= self.q1:       # in first quadrant.
            return x * self.r3
        if self.q1 < x <= self.q3:
            return self.h
        elif self.q3 < x <= self.max_x:
            return (self.max_x - x) * self.r3
        return 0

    def area(self, x: float, y: float) -> float:   # return area of shape to left of x.
        w = x - self.s                     # used just to make the math symmetric.
        if -self.s <= w < -self.q1:        # first quadrant.
            return y * x * 0.5
        elif abs(w) <= self.q1:
            ls = self.t_a + (x - self.q1) * self.h
            return ls
        elif self.q1 <= w < self.s:       # last quadrant
            return self.a - y * (self.max_x - x) * 0.5
        return 0. if x < 0. else self.a


if __name__ == '__main__':
    pts = [50, 98]
    sq_fn = ShapeFunctions(100.)
    ab = sq_fn.areas(pts)
    print(ab, sum(ab), pts)

    tri_fn = TriangleFns(100.)
    ab = tri_fn.areas(pts)
    print(ab, sum(ab), pts)

    hex_fn = HalfHexFns(50.)
    ab = hex_fn.areas(pts)
    print(ab, sum(ab), pts)
