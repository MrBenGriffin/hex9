import os
import math
import numpy as np
import svg
import json
from photo import Photo
from drawing import Drawing


def xyz_ll(xyz: tuple) -> tuple:
    x, y, z = xyz
    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))


class IcoVertex:
    def __init__(self, idx: int, obj):
        self.idx = idx
        self.name = obj['name']
        self.xyz = np.array(obj['xyz'])
        self.ll = xyz_ll(self.xyz)

    def jmap(self):
        return {'name': self.name, 'xyz': list(self.xyz), 'll': self.ll}

    def rotate(self, transform):
        self.xyz = transform @ self.xyz
        self.ll = xyz_ll(self.xyz)


class IcoSide:
    sh = [3., 3. * math.sqrt(3.) / 2.]  # Class variable

    def __init__(self, obj, vertices):
        self.name = obj['name']
        self.vx = obj['vx']
        self.grid = obj['grid']
        self.up = obj['up']
        mx, my = self.sh
        gx, gy = self.grid
        p0x, p0y = gx * mx * 0.5, gy * my
        if self.up:
            p1x, p1y = p0x - mx * 0.5, p0y - my
            p2x, p2y = p0x + mx * 0.5, p0y - my
        else:
            p2x, p2y = p0x - mx * 0.5, p0y + my
            p1x, p1y = p0x + mx * 0.5, p0y + my

        self.xy = tuple([(p0x, p0y), (p1x, p1y), (p2x, p2y)])  # This is the 2D triangle.
        self.xyz = tuple([vertices[d].xyz for d in self.vx])   # This is the unit sphere triangle.
        self.ll = tuple([xyz_ll(uv) for uv in self.xyz])

    def jmap(self):
        return {'name': self.name, 'grid': self.grid, 'up': self.up, 'vx': self.vx}


class IcoSphere:
    def __init__(self, file: str, transform=None):
        self.vertices = {}
        self.mapping = {}
        self.sides = {}
        self.grid = 0, 0, 11, 4
        with (open(file, 'r') as infile):
            obj = json.load(infile)
            infile.close()
            for idx, vertex in enumerate(obj['vertices']):
                self.vertices[idx] = IcoVertex(idx, vertex)
            if transform is not None:
                for idx in self.vertices.keys():
                    self.vertices[idx].rotate(transform)
            if 'mapping' in obj and obj['mapping']:
                self.mapping = obj['mapping']
            if 'sides' in obj and obj['sides']:
                self.sides = {side['name']: IcoSide(side, self.vertices) for side in obj['sides']}
            if 'grid' in obj and obj['grid']:
                self.grid = tuple(obj['grid'])

    def jmap(self):
        vertices = [v.jmap() for v in self.vertices.values()]
        sides = [s.jmap() for s in self.sides.values()]
        obj = {'vertices': vertices, 'mapping': self.mapping, 'sides': sides, 'grid': self.grid}
        return json.dumps(obj)


def rot(x, y, z):
    xr = np.deg2rad(x)
    yr = np.deg2rad(y)
    zr = np.deg2rad(z)

    rotation_alpha = [
        [1, 0, 0],
        [0, np.cos(xr), -np.sin(xr)],
        [0, np.sin(xr), np.cos(xr)],
    ]
    rx = np.array(rotation_alpha)

    rotation_beta = [
        [np.cos(yr), 0, np.sin(yr)],
        [0, 1, 0],
        [-np.sin(yr), 0, np.cos(yr)]
    ]
    ry = np.array(rotation_beta)

    rotation_gamma = [
        [np.cos(zr), -np.sin(zr), 0,],
        [np.sin(zr), np.cos(zr), 0],
        [0, 0, 1],
    ]
    rz = np.array(rotation_gamma)

    return np.matmul(np.matmul(rx, ry), rz)


if __name__ == '__main__':
    np.set_printoptions(precision=12, suppress=True)
    alt_mat = rot(0, -58.282525588539, 0)
    sphere = IcoSphere('assets/maps/phi.json', alt_mat)
    j_sphere = sphere.jmap()
    print(j_sphere)


