"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base import CompositeDomain, ComponentDomain, Points
from hhg9.base.point_format import PointFormat


class OctantCartesian(ComponentDomain):
    """
    This a 3D side of an Octahedron.
    """

    def __init__(self, registrar, dom, name: str, sign: tuple):
        super().__init__(registrar, name)
        self.dom = dom
        self.points = None
        self._sign = sign

    def sig(self) -> tuple:
        return self._sign

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        vals = np.array([pts]) if len(np.array(pts).shape) == 1 else np.array(pts)
        return np.sign(vals) == self.sig


class OctahedralCartesian(CompositeDomain):
    """
    Basic octahedral properties and methods.
    """

    def __init__(self, registrar: object) -> object:
        super().__init__(registrar, 'c_oct')

        # Define properties using symmetry
        self.sides = {}
        self.vertices = {}
        self.signs = {}

        # Define axes of symmetry for octahedron
        axes = [['A', 'P'], ['E', 'W'], ['N', 'S']]
        for i, axis in enumerate(axes):
            for j, v in zip([1, -1], axis):
                vertex = [0, 0, 0]
                vertex[i] = j
                self.vertices[v] = tuple(vertex)

        # Generate all face permutations using np.meshgrid
        face_permutations = np.stack(np.meshgrid(*axes), axis=-1).reshape((-1, 3))

        # Define face sides and their properties
        for face_arr in face_permutations:
            face = ''.join(face_arr[::-1])  # Reverse order to match face naming
            triple = np.asarray([self.vertices[s] for s in face_arr])
            sign = tuple(np.sum(triple, axis=1).tolist())
            sig = f'{self.name}:{face}'
            self.sides[face] = OctantCartesian(registrar, self, sig, sign)
            self.sides[face].points = triple
            self.sides[face].sig = sign
            self.signs[sign] = face  # Store based on sum signature
            self.components[sign] = self.sides[face]

    @classmethod
    def valid(cls, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        pbs = np.sum(np.abs(pts), axis=-1) - 1.
        return np.abs(pbs) < 1e-15

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
