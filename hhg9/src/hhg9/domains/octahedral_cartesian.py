# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'c_oct' cartesian octahedral surface xyz
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat


class OctantCartesian(ComponentDomain):
    """
    This a 3D side of an Octahedron.
    """

    def __init__(self, registrar, dom, name: str, sign: tuple):
        super().__init__(registrar, name, dom, None, sign, 3)
        self.points = None

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return a boolean mask indicating which 3D points belong to this octant face,
        i.e. their coordinate-wise signs match the face sign triple.
        Accepts a single point shape (3,) or an array of points shape (N,3).
        """
        vals = np.atleast_2d(np.asarray(pts, dtype=float))
        target = np.array(self.sig(), dtype=float)  # (+/-1, +/-1, +/-1)
        matches = np.sign(vals) == target
        mask = np.all(matches, axis=-1)
        return mask if np.ndim(pts) > 1 else mask[0]


class OctahedralCartesian(CompositeDomain):
    """
    Basic octahedral properties and methods in 3D cartesian space.
    self.signs maps sign triple → face name.
    """

    def __init__(self, registrar: object) -> object:
        super().__init__(registrar, 'c_oct', 3)

        # Define properties using symmetry
        self.sides = {}
        self.vertices = {}
        self.signs = {}

        # Stable ids for faces: maintain both name↔id and sign↔id maps
        self.face_id = {}         # face name -> id (0..7)
        self.id_face = {}         # id -> face name
        self.sign_to_id = {}      # sign triple -> id

        # Define axes of symmetry for octahedron
        axes = [['A', 'P'], ['E', 'W'], ['N', 'S']]
        for i, axis in enumerate(axes):
            for j, v in zip([1, -1], axis):
                vertex = [0, 0, 0]
                vertex[i] = j
                self.vertices[v] = tuple(vertex)

        # Generate all face permutations using np.meshgrid
        face_permutations = np.stack(np.meshgrid(*axes), axis=-1).reshape((-1, 3))
        # Generate a canonical list of edges also. Why?
        # It could generate the ID and name of the 12 root0 hexagons.
        self.edges = np.unique(face_permutations[:, [[0, 1], [1, 2], [0, 2]]].reshape((-1, 2)), axis=0)[:, ::-1]

        # self.verts = np.array((8, 3), dtype=np.float64)
        # Define face sides and their properties
        from hhg9 import Points
        for face_arr in face_permutations:
            face = ''.join(face_arr[::-1])  # Reverse order to match face naming
            triple = np.asarray([self.vertices[s] for s in face_arr])
            sign = tuple(np.sum(triple, axis=1).tolist())
            sig = f'{self.name}:{face}'
            self.sides[face] = OctantCartesian(registrar, self, sig, sign)
            self.sides[face].points = triple
            self.signs[sign] = face  # Store based on sum signature
            self.components[sign] = self.sides[face]
        octants = np.array(list(self.components.keys()), dtype=int)
        oids = Points.calc_octant_ids(octants)
        for fid_, triple in zip(oids, octants):
            fid = int(fid_)
            sign = tuple(triple.tolist())
            face = self.signs[tuple(sign)]
            self.face_id[face] = fid
            self.id_face[fid] = face
            self.sign_to_id[tuple(sign)] = fid
            # DO NOT uncomment the below until ready to migrate components entirely to ID.
            # self.components[fid] = self.sides[face]

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

    def face_for_sign(self, sign: tuple) -> str:
        return self.signs[sign]

    def id_for_sign(self, sign: tuple) -> int:
        return self.sign_to_id[sign]