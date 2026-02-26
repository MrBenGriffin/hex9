# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0


import math
from dataclasses import dataclass

@dataclass
class SphericalPcStepper:
    w: int = 3600
    h: int = 1800
    use_lut_trig: bool = True
    use_lut_asin_atan2: bool = True
    asin_lut_size: int = 65536
    atan_lut_size: int = 65536

    def __post_init__(self):
        # ----- per-row latitude tables (plate carrée) -----
        self.sin_phi = [0.0] * self.h
        self.cos_phi = [0.0] * self.h
        self.lam = [0.0] * self.w  # longitude radians for each x (optional convenience)

        for y in range(self.h):
            lat_deg = 90.0 - (y + 0.5) * 180.0 / self.h
            phi = math.radians(lat_deg)
            self.sin_phi[y] = math.sin(phi)
            self.cos_phi[y] = math.cos(phi)

        for x in range(self.w):
            lon_deg = (x + 0.5) * 360.0 / self.w - 180.0
            self.lam[x] = math.radians(lon_deg)

        # ----- bearing tables: 0..359 degrees (change resolution if you want) -----
        self.sin_theta = [0.0] * 360
        self.cos_theta = [0.0] * 360
        for deg in range(360):
            th = math.radians(deg)
            self.sin_theta[deg] = math.sin(th)
            self.cos_theta[deg] = math.cos(th)

        # ----- asin LUT over t in [-1, 1] -----
        if self.use_lut_asin_atan2:
            n = self.asin_lut_size
            self.asin_lut = [0.0] * n
            for i in range(n):
                t = -1.0 + 2.0 * i / (n - 1)
                self.asin_lut[i] = math.asin(t)

            # atan LUT over r in [0, 1] for atan(r)
            m = self.atan_lut_size
            self.atan_lut = [0.0] * m
            for i in range(m):
                r = i / (m - 1)
                self.atan_lut[i] = math.atan(r)

        # ----- optional distance LUT: here in 0.1° steps up to 180° -----
        # If your game uses a different distance quantisation, change this.
        self.dist_step_deg = 0.1
        self.max_dist_deg = 180.0
        steps = int(self.max_dist_deg / self.dist_step_deg) + 1
        self.sin_delta = [0.0] * steps
        self.cos_delta = [0.0] * steps
        for k in range(steps):
            delta = math.radians(k * self.dist_step_deg)
            self.sin_delta[k] = math.sin(delta)
            self.cos_delta[k] = math.cos(delta)

    # ---------- small helpers ----------
    @staticmethod
    def _wrap_pi(x: float) -> float:
        # wrap to [-pi, pi)
        two_pi = 2.0 * math.pi
        x = (x + math.pi) % two_pi - math.pi
        return x

    def _asin_fast(self, t: float) -> float:
        # clamp then LUT+linear interpolation
        if t <= -1.0:
            return -math.pi / 2.0
        if t >= 1.0:
            return math.pi / 2.0

        n = self.asin_lut_size
        u = (t + 1.0) * 0.5 * (n - 1)
        i = int(u)
        f = u - i
        if i >= n - 1:
            return self.asin_lut[n - 1]
        return self.asin_lut[i] * (1.0 - f) + self.asin_lut[i + 1] * f

    def _atan_fast_0_1(self, r: float) -> float:
        # r in [0,1], LUT+linear interpolation
        if r <= 0.0:
            return 0.0
        if r >= 1.0:
            return math.pi / 4.0

        m = self.atan_lut_size
        u = r * (m - 1)
        i = int(u)
        f = u - i
        if i >= m - 1:
            return self.atan_lut[m - 1]
        return self.atan_lut[i] * (1.0 - f) + self.atan_lut[i + 1] * f

    def _atan2_fast(self, y: float, x: float) -> float:
        # atan2 via atan LUT on [0,1] with quadrant reconstruction
        if x == 0.0:
            if y > 0.0:
                return math.pi / 2.0
            if y < 0.0:
                return -math.pi / 2.0
            return 0.0

        ax = abs(x)
        ay = abs(y)

        if ay <= ax:
            r = ay / ax  # in [0,1]
            a = self._atan_fast_0_1(r)
        else:
            r = ax / ay
            a = math.pi / 2.0 - self._atan_fast_0_1(r)

        # apply quadrant
        if x > 0.0:
            return a if y >= 0.0 else -a
        else:
            return (math.pi - a) if y >= 0.0 else (a - math.pi)

    # ---------- main step ----------
    def step(self, x: int, y: int, bearing_deg: int, dist_deg: float):
        """
        Inputs:
          x,y          : pixel coords on 3600x1800 plate carrée (x eastward, y downward)
          bearing_deg  : integer degrees (0=N, 90=E, 180=S, 270=W)
          dist_deg     : angular distance in degrees

        Returns:
          (x2,y2) pixel coords (wrapped in x, clamped in y)
        """
        # current spherical position (radians)
        lam1 = self.lam[x % self.w]
        sin_phi1 = self.sin_phi[max(0, min(self.h - 1, y))]
        cos_phi1 = self.cos_phi[max(0, min(self.h - 1, y))]

        # bearing trig
        b = bearing_deg % 360
        if self.use_lut_trig:
            sin_th = self.sin_theta[b]
            cos_th = self.cos_theta[b]
        else:
            th = math.radians(bearing_deg)
            sin_th = math.sin(th)
            cos_th = math.cos(th)

        # distance trig (use distance LUT in 0.1° steps; fall back if out-of-grid)
        if self.use_lut_trig:
            k = int(round(dist_deg / self.dist_step_deg))
            if 0 <= k < len(self.sin_delta):
                sin_d = self.sin_delta[k]
                cos_d = self.cos_delta[k]
            else:
                d = math.radians(dist_deg)
                sin_d = math.sin(d)
                cos_d = math.cos(d)
        else:
            d = math.radians(dist_deg)
            sin_d = math.sin(d)
            cos_d = math.cos(d)

        # forward great-circle step on a sphere:
        # sin(phi2) = sin(phi1)*cos(d) + cos(phi1)*sin(d)*cos(theta)
        sin_phi2 = sin_phi1 * cos_d + cos_phi1 * sin_d * cos_th
        # clamp for safety
        if sin_phi2 < -1.0:
            sin_phi2 = -1.0
        elif sin_phi2 > 1.0:
            sin_phi2 = 1.0

        if self.use_lut_asin_atan2:
            phi2 = self._asin_fast(sin_phi2)
        else:
            phi2 = math.asin(sin_phi2)

        # lon step:
        # lam2 = lam1 + atan2( sin(theta)*sin(d)*cos(phi1), cos(d) - sin(phi1)*sin(phi2) )
        y_atan = sin_th * sin_d * cos_phi1
        x_atan = cos_d - sin_phi1 * sin_phi2

        if self.use_lut_asin_atan2:
            dlam = self._atan2_fast(y_atan, x_atan)
        else:
            dlam = math.atan2(y_atan, x_atan)

        lam2 = self._wrap_pi(lam1 + dlam)

        # map back to pixels
        lon2_deg = math.degrees(lam2)
        lat2_deg = math.degrees(phi2)

        x2 = int(math.floor((lon2_deg + 180.0) * self.w / 360.0)) % self.w
        y2 = int(math.floor((90.0 - lat2_deg) * self.h / 180.0))
        y2 = max(0, min(self.h - 1, y2))
        return x2, y2


if __name__ == "__main__":
    stepper = SphericalPcStepper(
        w=3600, h=1800,
        use_lut_trig=True,
        use_lut_asin_atan2=False
    )

    # Example: from London-ish pixel, move 5 degrees eastward along a bearing
    # (bearing 90° is due east; on a sphere this will drift in latitude unless you're on the equator)
    x0, y0 = 1800, 900  # roughly (lon=0, lat=0) with this pixel convention
    x1, y1 = stepper.step(x0, y0, bearing_deg=90, dist_deg=5.0)
    print("from:", (x0, y0), "to:", (x1, y1))

    # Compare against full math (disable LUTs) for the same move
    stepper_ref = SphericalPcStepper(
        w=3600, h=1800,
        use_lut_trig=False,
        use_lut_asin_atan2=False
    )
    xr, yr = stepper_ref.step(x0, y0, bearing_deg=90, dist_deg=5.0)
    print("reference:", (xr, yr))
