import numpy as np

def triangle_pixels(origin, side_length, apex_up=True):
    ox, oy = origin
    height = np.sqrt(3) / 2 * side_length

    # Triangle vertices in real coordinates
    if apex_up:
        v0 = np.array([ox + side_length / 2, oy])         # apex
        v1 = np.array([ox, oy + height])                  # bottom-left
        v2 = np.array([ox + side_length, oy + height])    # bottom-right
    else:
        v0 = np.array([ox + side_length / 2, oy + height])  # apex (down)
        v1 = np.array([ox, oy])                             # top-left
        v2 = np.array([ox + side_length, oy])               # top-right

    # Bounding box in integer pixel space
    x_min = int(np.floor(min(v0[0], v1[0], v2[0])))
    x_max = int(np.ceil(max(v0[0], v1[0], v2[0])))
    y_min = int(np.floor(min(v0[1], v1[1], v2[1])))
    y_max = int(np.ceil(max(v0[1], v1[1], v2[1])))

    # Precompute triangle area
    def edge(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    pixels = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            p = np.array([x + 0.5, y + 0.5])  # pixel center
            w0 = edge(v1, v2, p)
            w1 = edge(v2, v0, p)
            w2 = edge(v0, v1, p)
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                pixels.append((x, y))

    return np.array(pixels)


if __name__ == '__main__':
    pix = triangle_pixels((755.5, 50.22), 1500)
    import matplotlib.pyplot as plt
    plt.scatter(pix[:, 0], pix[:, 1], s=1)
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()  # Optional: top-down coordinate system
    plt.show()
