"""
Part of the H9 project
GCD is recognised as [latitude,longitude]
"""
from numpy.typing import NDArray
from hhg9.base import Domain, Points


class SphericalGCD(Domain):

    def __init__(self, registrar):
        super().__init__(registrar, 'g_sph')
        self.lat_min, self.lat_max = (-90., 90.)
        self.lon_min, self.lon_max = (-180., 180.)

    def valid(self, _: Points) -> NDArray:
        raise NotImplementedError

    def adopt(self, pts: NDArray):
        """
        Take an array and adopt as this domain.
        """
        return pts.view(Points).set_domain(self)
