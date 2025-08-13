"""
Part of the H9 project - Preparation 0006
Load the Population Octahedron numpy data file
Last Tested 11 August 2025 √
"""
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import Normalize
from hhg9 import Registrar, H9Engine, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
from matplotlib import pyplot as plt, image, patches
from matplotlib.collections import PolyCollection


def heatmap(df, addresses, polys, layers):
    """
    Generates a hexbin heatmap from binned address data in barycentric space.
    Args:
        df (pd.DataFrame): A DataFrame with 'address' and 'count' columns.
        addresses (dict): the address->index map
        polys (np.array): An ordered numpy array of polygons.
        layers (tuple): min/max layers to show.
    Returns:
        list of layers of polygons.
    """
    if df.empty:
        print("Heatmap data is empty. Nothing to plot.")
        return

    layers_to_plot = range(*layers)
    final_layer = layers[1] - 1
    final_pops = []
    polygons_to_plot = []
    for layer in layers_to_plot:
        last_col = f'L{layer-1}'
        next_col = f'L{layer}'
        # Check if columns exist before trying to access them
        if last_col not in df.columns:
            continue
        if next_col not in df.columns:  # This handles the deepest possible layer
            mask = df[last_col].notna()
        else:
            mask = df[last_col].notna() & df[next_col].isna()
        layer_data = df[mask]
        if layer_data.empty:
            continue
        polygons_for_layer = []
        address_cols = [col for col in layer_data.columns]
        for index, row in layer_data.iterrows():
            address_tuple = tuple(row[address_cols].dropna().astype(int))
            if layer == final_layer:
                pref_mask = pd.Series(True, index=df.index)
                for k, prefix_val in enumerate(address_tuple):
                    pref_mask &= (df[f'L{k}'] == prefix_val)
                final_pops.append(pref_mask.sum())
            poly_indices = addresses[address_tuple]
            poly_vertices = polys[poly_indices]
            polygons_for_layer.append(poly_vertices)
        polygons_to_plot.append(np.array(polygons_for_layer))
    return polygons_to_plot, final_pops


class AddressCounter:
    """Count things"""
    def __init__(self, unique_polygons, shape_array):
        self.unique_polygons = unique_polygons
        max_depth = max(len(key) for key in unique_polygons)
        self.df = pd.DataFrame(unique_polygons.keys(),
                               columns=[f'L{i}' for i in range(max_depth)])

    def prefix_mask(self, prefix_key):
        """
        Performs a fast, vectorized match of addresses matching a prefix,
        Returning the mask of those which match the prefix.
        This returns a `pandas` mask, so use to_numpy() to convert for numpy array filtering.
        """
        mask = pd.Series(True, index=self.df.index)
        for k, prefix_val in enumerate(prefix_key):
            mask &= (self.df[f'L{k}'] == prefix_val)
        return mask

    def count_children(self, prefix_key):
        """
        Performs a fast, vectorized count of addresses matching a prefix.
        """
        mask = self.prefix_mask(prefix_key)
        return mask.sum()

    def count_by_length(self, n):
        """
        Counts the number of addresses that have a specific length n.
        """
        if n == 0:
            return 0

        if f'L{n}' not in self.df.columns:
            return 0  # Or handle as an error

        last_col = f'L{n - 1}'
        next_col = f'L{n}'

        # Create the boolean mask and return the sum
        mask = self.df[last_col].notna() & self.df[next_col].isna()
        return mask.sum()


if __name__ == '__main__':
    file = 'jpn'
    h9 = H9Engine()            # formatter.
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    eg = EllipsoidGCD(reg)            # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)   # c_sph <=> (c_oct <=> b_oct)
    ak.set_accuracy(0.01)  # cm

    theta = np.load(f'src/{file}_theta.npy')
    centroid = np.load(f'src/{file}_centroid.npy')
    cos_, sin_ = np.cos(theta), np.sin(theta)
    matrix = np.array([[cos_, -sin_], [sin_, cos_]])

    hex_layers = 15
    pop_data = np.load(f'src/{file}_pop_data.npy')
    # gcd_data = np.load(f'src/{file}_lon_lat_pop.npy')
    # pop_data = gcd_data[: -1]
    pos_b = np.load(f'src/{file}_bry.npy')
    pos_c = np.load(f'src/{file}_bry_cmp.npy')
    b_pts = Points(pos_b, b_oct, components=pos_c, samples=pop_data)

    # do the heatmap work.
    # convert the barycentric points to ucg_regions
    # and return the full unique mesh.
    co, mo = b_pts.cm()
    uri = h9.ugc_regions(b_pts.coords, mo, hex_layers)
    ums, shapes = h9.enmesh(uri)
    # count the population within each shape.
    # and limit the number layers to be shown to a range.
    counter = AddressCounter(ums, shapes)
    layers = 4, 10
    layer_polys, final_pops = heatmap(counter.df, ums, shapes, layers)
    # layer_polys is a list of numpy - lets transform them as a single array.
    lengths = [len(arr) for arr in layer_polys]  # flatten/unflatten for transforms.
    split_indices = np.cumsum(lengths)[:-1]
    # transform the values.
    points = np.concatenate(layer_polys).reshape(-1, 2)
    squared = (points-centroid) @ matrix + centroid
    final_polys = np.split(squared.reshape(-1, 5, 2), split_indices)
    grid_shape = np.load(f'src/{file}_rot_bry_border.npy')
    minx, miny = min(grid_shape[:, 0]), min(grid_shape[:, 1])
    maxx, maxy = max(grid_shape[:, 0]), max(grid_shape[:, 1])
    ratio = (maxy-miny) / (maxx-minx)

    dpi = 100
    img_w_pix = 6000
    img_h_pix = int(img_w_pix * ratio)
    fig_w_in, fig_h_in = img_w_pix / dpi, img_h_pix / dpi
    fig = plt.figure(figsize=(img_w_pix / dpi, img_h_pix / dpi), dpi=dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    img_file = f'src/{file}_grid.png'
    if os.path.isfile(img_file):  # must be barycentric.
        rgb = image.imread(img_file, 'png')
        img = np.stack([np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), ] * 3 + [rgb[..., 3]], axis=-1).astype(
            rgb.dtype)
        extent = np.load(f'src/{file}_bg_extent.npy')
        (l, b, r, t) = extent
        ax.imshow(img, extent=(l, r, b, t), alpha=1.0)
    else:
        raise FileNotFoundError(f'{img_file} not found.')

    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=np.min(final_pops), vmax=np.max(final_pops))
    colors = cmap(norm(final_pops))
    colors[:, 3] = 0.10  # Set alpha

    if final_polys is not None:
        for level, polys in enumerate(final_polys):
            layer = level + layers[0]
            if layer == layers[1] - 1:
                collection = PolyCollection(
                    polys,
                    facecolors=colors,
                    ec=(0, 0, 0, 0.2),  # Add a black edge for clarity
                    linewidth=0.25,
                    antialiaseds=True
                )
            else:
                collection = PolyCollection(
                    polys,
                    ec='k',   # Add a black edge for clarity
                    fc='none',
                    linewidth=2,
                    alpha=0.01,
                    antialiaseds=True
                )
            ax.add_collection(collection)
    ax.axis('off')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', 'box')
    # plt.show()
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    im = data.reshape((img_h_pix, img_w_pix, -1))
    plt.close(fig)
    pil_img = Image.fromarray(im)
    pil_img.save(f'src/{file}_heatmap.png')
