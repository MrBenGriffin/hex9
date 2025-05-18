import math
import numpy as np

# This calculates the rotation matrix between two icosahedra mapped onto a unit sphere.

if __name__ == '__main__':
    phi = 2.0 * math.cos(math.pi / 5.0)
    ico = np.array([[0, -1, -phi],
                    [0, -1, +phi],
                    [0, +1, -phi],
                    [0, +1, +phi],
                    [-1, -phi, 0],
                    [-1, +phi, 0],
                    [+1, -phi, 0],
                    [+1, +phi, 0],
                    [-phi, 0, -1],
                    [-phi, 0, +1],
                    [+phi, 0, -1],
                    [+phi, 0, +1]])
    for i in range(12): ico[i] = ico[i] / np.linalg.norm(ico[i])  # Normalise

    alts = np.array([
        [0.0, -0.5257311121191336, -0.85065080835204],
        [0.0, -0.5257311121191336, 0.85065080835204],
        [0.0, 0.5257311121191336, -0.85065080835204],
        [0.0, 0.5257311121191336, 0.85065080835204],
        [-0.5257311121191336, -0.85065080835204, 0.0],
        [-0.5257311121191336, 0.85065080835204, 0.0],
        [0.5257311121191336, -0.85065080835204, 0.0],
        [0.5257311121191336, 0.85065080835204, 0.0],
        [-0.85065080835204, 0.0, -0.5257311121191336],
        [-0.85065080835204, 0.0, 0.5257311121191336],
        [0.85065080835204, 0.0, -0.5257311121191336],
        [0.85065080835204, 0.0, 0.5257311121191336]
    ])


    U = np.zeros((3, 3))
    V = np.zeros((3, 3))
    independent = (0, 4, 8)
    for r in range(3):
        for c in range(3):
            U[r, c] = ico[independent[c], r]
            V[r, c] = alts[independent[c], r]

    R = V @ np.linalg.inv(U)

    for i in range(12):
        print("Vertex ", i, "   alts[i] = ", alts[i], "    R.ico = ", R @ ico[i].T, "    R.ico = ", R @ ico[i])

    z = (R @ ico.T).T
    print(f"z: {z}")

    print("\nRotation matrix:\n", R)
    print("\nCheck determinant = ", np.linalg.det(R))
