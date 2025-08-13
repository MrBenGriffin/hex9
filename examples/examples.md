# The Python Project
The current set of python files in this repo are a set of (mainly) 
self-contained classes that have acted as a testbed for the `hex9` grid system. 
While not particularly suitable for production systems, it should be good 
enough for playing with, or evaluating the grid system itself.

While much of the work involves providing suitable GIS operations for 
manipulating addresses and coordinates, there are two areas that are likely 
to be of some interest.

* The Hex9 Grid itself
* The Octahedral Projection of the Ellipsoid (and/or Sphere).

The latter exploits the Hex9 grid for conversion of ellipsoidal to octahedral 
addresses. 

## Examples
The example files `ex0000_` each represent a stage of functionality from the 
most simple (load a Plate Carrée for projection) to the more complex. This 
has helped determine where there may be any bugs or flaws in the overarching 
system.  While I could have used any number of GIS libraries to do the heavy 
lifting, the process of development was quite fluid, and it seemed sensible 
to provide a flexible abstract layer between any GIS libraries used.
Some of these examples do use libraries.  Much of the existing codebase 
rests upon `geographiclib` - an awesome craft in itself.

*Up to example ex0040, there are **no** octahedral nor hex grid calculations.*

### ex0010_plate_px
Roundtrip Load/Save of a Plate Carrée Image. Display via matplotlib.

| √ | Domain                | Sig   | Octant Domain     |
|--:|:----------------------|-------|-------------------|
| √ | Plate Carrée          | p_plt | -                 |
| X | GeneralGCD            | g_gcd | -                 |
| X | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |

| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | plt_gcd | p_plt |  X  | g_gcd |  X  |
| EllipsoidGCD          | ell_gcd | c_ell |  X  | g_gcd |  X  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  X  |


### ex0020_plate_glb
Roundtrip Conversion of a Plate Carrée Image to GCD. 
Display via matplotlib Basemap and 2D.

| √ | Domain                | Sig   | Octant Domain     |
|--:|:----------------------|-------|-------------------|
| √ | Plate Carrée          | p_plt | -                 |
| √ | GeneralGCD            | g_gcd | -                 |
| X | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |

| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | plt_gcd | p_plt |  √  | g_gcd |  √  |
| EllipsoidGCD          | ell_gcd | c_ell |  X  | g_gcd |  X  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  X  |

### ex0030_plate_sph
Convert Plate Carrée to Cartesian XYZ Ellipsoid and display, with roundtrip.
Display via matplotlib 2D and 3D.

| √ | Domain                | Sig   | Octant Domain     |
|--:|:----------------------|-------|-------------------|
| √ | Plate Carrée          | p_plt | -                 |
| √ | GeneralGCD            | g_gcd | -                 |
| √ | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | plt_gcd | p_plt |  √  | g_gcd |  √  |
| EllipsoidGCD          | ell_gcd | c_ell |  √  | g_gcd |  √  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  X  |


### ex0041_cache
The AKOctahedralEllipsoid forward is c_oct->c_ell, which is rapid and easily 
vectorised.  However, it does not lend itself well to a simple backward 
method, and so some form of root-finding must be deployed.  

There are general purpose root-finders (such as offered by scipy, etc.) and 
likewise it is not too hard to train a neural network for good estimates. 
However, for precision, neither are particularly strong, and they tend to 
end up mashing around when dealing with some of the more fragile areas of 
the forward function.

This is where domain specialisation really helps, and as the grid system 
itself offers a thorough subdivision of the octahedron down to numerical 
limits we can exploit it, via a branch-and-bound scheme, to do the heavy 
lifting for us.  

While the current code has not been optimised, it follows this strategy 
of exploiting the strength of the h9 grid system behind the scenes. 

Needless to say, due to this dependency, there's an innate fragility during 
development, which requires a degree of robustness to the h9 grid.

Also, (and as this was a recent development), the AKOctahedralEllipsoid 
projection works directly from GeneralGCD to OctahedralBarycentric, 
but currently I have force-fitted it (via internal transforms) to act as the 
c_ell-c_oct projection.

As the process is quite slow, the best strategy for 'projecting' is to use 
an image as a sample, and then query the sample for each point of the 
octahedron. However, for the purposes of test that the ellipsoid->octahedron 
projection works, it is useful demonstrate that. Because the process is slow,
it is a good idea to generate a cache of the projection. 

This is what this example does.  *(OctahedralBarycentric is implicitly used 
 by AKOctahedralEllipsoid)

| √ | Domain                 | Sig   | Octant Domain     |
|--:|:-----------------------|-------|-------------------|
| √ | Plate Carrée           | p_plt | -                 |
| √ | GeneralGCD             | g_gcd | -                 |
| √ | EllipsoidCartesian     | c_ell | -                 |
| √ | OctahedralCartesian    | c_oct | OctantCartesian   |
| * | OctahedralBarycentric  | b_oct | OctantBarycentric |
| X | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | plt_gcd | p_plt |  √  | g_gcd |  √  |
| EllipsoidGCD          | ell_gcd | c_ell |  √  | g_gcd |  √  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  √  |


### ex0042_plate_oct
This example displays the cache from ex0041, and demonstrates the forward 
operation (octahedral to ellipsoid).
It also reads the source image, converts the colours to samples,
In both cases, it implements the 'adopt' method of their respective domain.
The octahedron is displayed, and then the points are projected back to Plate 
Carree, and displayed once more.

| √ | Domain                 | Sig   | Octant Domain     |
|--:|:-----------------------|-------|-------------------|
| √ | Plate Carrée           | p_plt | -                 |
| * | GeneralGCD             | g_gcd | -                 |
| * | EllipsoidCartesian     | c_ell | -                 |
| √ | OctahedralCartesian    | c_oct | OctantCartesian   |
| √ | OctahedralBarycentric  | b_oct | OctantBarycentric |
| X | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  | √ |  Rev  | √ | 
|-----------------------|:-------:|:-----:|:-:|:-----:|:-:|
| PlatePixelGCD         | plt_gcd | p_plt | X | g_gcd | X |
| EllipsoidGCD          | ell_gcd | c_ell | X | g_gcd | X |
| AKOctahedralEllipsoid | oct_ell | c_oct | √ | c_ell | X |


### ex0045_plate_net
Similar to 42, this loads the cache from ex0041, and demonstrates the forward 
operation (octahedral to octahedral net) using the plate carree pixels.
This shows holes / dots, because of distortion difference between 
plate carree and barycentric. For surface mapping, we will henceforth use a 
back-projection and sampling technique from the destination field.

| √ | Domain                 | Sig   | Octant Domain     |
|--:|:-----------------------|-------|-------------------|
| √ | Plate Carrée           | p_plt | -                 |
| * | GeneralGCD             | g_gcd | -                 |
| * | EllipsoidCartesian     | c_ell | -                 |
| * | OctahedralCartesian    | c_oct | OctantCartesian   |
| √ | OctahedralBarycentric  | b_oct | OctantBarycentric |
| √ | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  | √ |  Rev  | √ | 
|-----------------------|:-------:|:-----:|:-:|:-----:|:-:|
| PlatePixelGCD         | plt_gcd | p_plt | X | g_gcd | X |
| EllipsoidGCD          | ell_gcd | c_ell | X | g_gcd | X |
| AKOctahedralEllipsoid | oct_ell | c_oct | √ | c_ell | X |


### ex0060_addr_exm
This is a non-graphical example which chooses a selection of landmarks 
(categorised by Octant) across the globe, for which there are 
well-known co-ordinates, and which can be found easily via online mapping services.  

Having set the AKOctahedralEllipsoid accuracy to 1nm, (maximum) this then 
converts each GCD address into it's hexagon grid reference, then 
roundtrip reverts to the original address using that grid reference only.

Projection differences are reported along the projection chain, and the 
geographic difference is then reported in nanometres, via 
geographiclib's `Geodesic.WGS84` inverse.

An example landmark is Stonehenge:
```
Stonehenge              51°10'43.672800075871"N, 1°49'34.031280757836"W (Reference Coordinates)
Stonehenge  ∂0.000000nm 51°10'43.672800075871"N, 1°49'34.031280757836"W (roundtrip via GCD<->Ellipsoid)
Stonehenge  ∂1.074432nm 51°10'43.672800075845"N, 1°49'34.031280757844"W (roundtrip via GCD<->Octahedral)
Stonehenge  ∂1.918622nm 51°10'43.672800075819"N, 1°49'34.031280757874"W (roundtrip via GCD<->Barycentric)
Stonehenge  NWΛ01357247546287033010625120268308D7 (Grid Address)
Stonehenge  ∂1.918622nm 51°10'43.672800075819"N, 1°49'34.031280757874"W (roundtrip via Grid Address)
```

Maximum errors tend to be found at the octahedral vertices and edges - with a 
deviation of around 6nm at the North Pole, for example. 
Needless to say, nobody will be seriously expecting to use this toolchain to 
geolocate distances as small as a few nanometres, but it provides an 
idea of where noise can be found creeping in.

### ex0061_neighbours
While not very friendly to the eyes, this shows displays the native region 
list for the example addresses above, then calculates the **neighbouring 
half-hexagon** for the significant leaf-node, round-trips the result, and 
reports on any errors.  Neighbour-finding is a little more complex than one would 
expect, in that the neighbour's parent might belong to a different parent 
(and that's something which can easily cascade). 
(categorised by Octant) across the globe, for which there are 
well-known co-ordinates, and which can be found easily via online mapping services.  


### ex0075_osm_mesh
The osm mesh example relies upon OpenStreetMap and cartopy to retrieve and 
store a series of images based upon a single address. The purpose of which 
is to demonstrate how each level of the Grid address reduces the land area 
being addressed, and these are then used in ex0076. 

The meshes being stored are mere boundary sets in plate carree, from which 
we can later use as sample sources to depict the relevant half-hexagon.

While Stonehenge has been used as a reference, any geographic location may 
be similarly used.

### ex0076_stonehenge.
Using the series of plate carree images retrieved in ex0075, this example 
now generates the relevant half-hexagon for each address, then using the 
relevant image as a sample source, generates the half-tile represented by 
the hexagon and it's mode. This could be authored better, but is reasonably 
straightforward to understand.

The population examples (pr000) are probably more useful now.

### ex0080_london
This should do something! 


### ex0090_smp_grid
This demonstrates using a plate carree image as a sample source.
The source image is loaded and projected onto WGS84 Ellipsoid, and 
registered into a KDTree.
The octahedral net pixels are identified then a copy is projected to 
EllipsoidGCD, and query the KDTree for a sample value. 
These samples may then be used to display the Octahedral Net.

Again, while this works - there is a 'better' version of 
this in pr0005, which uses a rotation component to improve the use of the 
sample space.

### ex0094_bary
This renders the individual barycentric triangles which are the rotated 
faces of the octahedron onto 2D space and form the basis of hexgrid 
calculations. They serve several purposes for debugging and address 
verification but are probably not so interesting beyond that. Normally one 
prefers to see them stitched into a net.

### ex0100_h9
This example renders each pixel into it's hexgrid address, and then colours 
the pixel according to a digit location, for the first 5 address points.
It demonstrates how the grid address maps onto the octahedron.

### ex0110_poly
Here we take the reference location of Stonehenge, convert it to it's region 
list format, and then test each layer for it's neighbour, generating the 
neighbour in a subplot.

### ex0111_covr
Here we attempt to demonstrate that every combination of neighbour correctly 
finds it's neighbour. (There are 18 to show).

### ex0112_nbhr
We go through a series of short addresses, and test the neighbour 
calculation for roundtrip errors.

### ex0113_nbhr
We test our reference addresses and test the neighbour 
calculation for roundtrip errors, with a different printout.

# The following scripts belong to a pipeline for generating heatmaps.
### pr0001_csv
This converts a CSV into a np file.
The CSV may be downloaded from https://data.humdata.org/organization/meta
and consists of 3 values: longitude/latitude/populaton.
They represent sparse grid values at 30m.
* Input  
   * src/{file}_general_2020.csv 
* Output 
   * src/{file}_lon_lat_pop.npy

### pr0002_prj
Here we take the file and forward project the GCD addresses,
converting them to barycentric octahedral co-ordinates, which are
x,y and octant (in cmp).
This uses root-finding, and is batch-processed across multiple cores.
* Input
    * src/{file}_lon_lat_pop.npy
* Output
    * src/{file}_bry.npy
    * src/{file}_bry_cmp.npy

### pr0002b_prj
Take the same data as before, extract the population data, 
find the boundaries of the gcd, project and store, both as gcd and as 
barycentric, having added padding of 2.5% border.

* Input
    * src/{file}_lon_lat_pop.npy
* Output
    * src/{file}_pop_data.npy
    * src/{file}_lat_lon_bounds.npy
    * src/{file}_bounds_bry.npy
    * src/{file}__bounds_bry_cmp.npy

### pr0003_gcd_img
Using gcd_bounds saved at pr0002; retrieve the background 
map from OSM via Cartopy for the area and store it as a png.
* Input
    * src/{file}_lat_lon_bounds.npy
* Output
    * src/{file}_gcd.png

### pr0004_theta
Using gcd_bounds saved at pr0002; project the bounds onto barycentric, 
calculate the optimal rotation angle and centroid of the grid to represent 
the area. Draw (for visualisation) the barycentric area before and after 
rotation.
* Input
    * src/{file}_lat_lon_bounds.npy
* Output
    * src/{file}_theta.npy
    * src/{file}_centroid.npy
    * src/{file}_bry_border.npy
    * src/{file}_rot_bry_border.npy

### pr0005_grid
Now convert the gcd image into its barycentric equivalent.
Store the final grid, and it's extent for placing as a backdrop for a heatmap.
* Input
   * src/{file}_lat_lon_bounds.npy
   * src/{file}_bry_border.npy
   * src/{file}_bounds_bry_cmp.npy
   * src/{file}_theta.npy
   * src/{file}_centroid.npy
   * src/{file}_rot_bry_border.npy
   * src/{file}_gcd.png
* Output 
   * src/{file}_bg_extent.npy
   * src/{file}_grid.png

### pr0007_hh_heatmap
Finally, we can generate the heatmap over the top of the grid image.
* Input      
   * src/{file}_theta.npy
   * src/{file}_centroid.npy
   * src/{file}_pop_data.npy
   * src/{file}_bry.npy
   * src/{file}_bry_cmp.npy
   * src/{file}_rot_bry_border.npy
   * src/{file}_bg_extent.npy
   * src/{file}_grid.png
* Output
   * src/{file}_heatmap.png
