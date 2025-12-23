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
| √ | Plate Carrée          | p_pix | -                 |
| X | GeneralGCD            | g_gcd | -                 |
| X | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |

| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | pix_gcd | p_pix |  X  | g_gcd |  X  |
| EllipsoidGCD          | ell_gcd | c_ell |  X  | g_gcd |  X  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  X  |


### ex0020_plate_glb
Roundtrip Conversion of a Plate Carrée Image to GCD. 
Display via matplotlib Basemap and 2D.

| √ | Domain                | Sig   | Octant Domain     |
|--:|:----------------------|-------|-------------------|
| √ | Plate Carrée          | p_pix | -                 |
| √ | GeneralGCD            | g_gcd | -                 |
| X | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |

| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | pix_gcd | p_pix |  √  | g_gcd |  √  |
| EllipsoidGCD          | ell_gcd | c_ell |  X  | g_gcd |  X  |
| AKOctahedralEllipsoid | oct_ell | c_oct |  X  | c_ell |  X  |

### ex0030_plate_sph
Convert Plate Carrée to Cartesian XYZ Ellipsoid and display, with roundtrip.
Display via matplotlib 2D and 3D.

| √ | Domain                | Sig   | Octant Domain     |
|--:|:----------------------|-------|-------------------|
| √ | Plate Carrée          | p_pix | -                 |
| √ | GeneralGCD            | g_gcd | -                 |
| √ | EllipsoidCartesian    | c_ell | -                 |
| X | OctahedralCartesian   | c_oct | OctantCartesian   |
| X | OctahedralBarycentric | b_oct | OctantBarycentric |
| X | OctahedralNet         | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | pix_gcd | p_pix |  √  | g_gcd |  √  |
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
| √ | Plate Carrée           | p_pix | -                 |
| √ | GeneralGCD             | g_gcd | -                 |
| √ | EllipsoidCartesian     | c_ell | -                 |
| √ | OctahedralCartesian    | c_oct | OctantCartesian   |
| * | OctahedralBarycentric  | b_oct | OctantBarycentric |
| X | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  |  √  |  Rev  |  √  | 
|-----------------------|:-------:|:-----:|:---:|:-----:|:---:|
| PlatePixelGCD         | pix_gcd | p_pix |  √  | g_gcd |  √  |
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
| √ | Plate Carrée           | p_pix | -                 |
| * | GeneralGCD             | g_gcd | -                 |
| * | EllipsoidCartesian     | c_ell | -                 |
| √ | OctahedralCartesian    | c_oct | OctantCartesian   |
| √ | OctahedralBarycentric  | b_oct | OctantBarycentric |
| X | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  | √ |  Rev  | √ | 
|-----------------------|:-------:|:-----:|:-:|:-----:|:-:|
| PlatePixelGCD         | pix_gcd | p_pix | X | g_gcd | X |
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
| √ | Plate Carrée           | p_pix | -                 |
| * | GeneralGCD             | g_gcd | -                 |
| * | EllipsoidCartesian     | c_ell | -                 |
| * | OctahedralCartesian    | c_oct | OctantCartesian   |
| √ | OctahedralBarycentric  | b_oct | OctantBarycentric |
| √ | OctahedralNet          | n_oct | OctantNet         |


| Domain Projections    |   Sig   |  Fwd  | √ |  Rev  | √ | 
|-----------------------|:-------:|:-----:|:-:|:-----:|:-:|
| PlatePixelGCD         | pix_gcd | p_pix | X | g_gcd | X |
| EllipsoidGCD          | ell_gcd | c_ell | X | g_gcd | X |
| AKOctahedralEllipsoid | oct_ell | c_oct | √ | c_ell | X |

### ex0051_subregions
This is a visual demonstrator showing how an address is repeatedly
broken down into one of 9 sub-regions at each layer.

### ex0060_addresses
This is a non-graphical example which chooses a seeded random collection of
100,000 GCD addresses across the globe, projects them to barycentric octahedral 
(hex 9), and back-projects them to GCD. It then measures the round-trip 
distance via GeographicLib, and stores the result in a CSV file.  
Typically results are under 7 nanometres.  It also converts to the 
hex-variation address format, reverts that, and then 
back-projects that value also - which typically might add about 1 nanometre
noise.  If one's purpose is to measure global positions accurately to less than 
1µm, I would strongly recommend using 96 or 128-bit maths.

### ex0061_more_addr
This is a non-graphical example which chooses a selection of landmarks 
(categorised by Octant) across the globe, for which there are 
well-known co-ordinates, and which can be found easily via online mapping services.  

Having set the AKOctahedralEllipsoid accuracy to 1nm, (maximum) this then 
converts each GCD address into it's hexagon grid reference, then 
roundtrip reverts to the original address using that grid reference only.

Projection differences are reported along the projection chain, and the 
geographic difference is then reported in nanometres, via 
geographiclib's `Geodesic.WGS84` inverse.

An example landmark is Greenwich Park East:
```
Greenwich Park East (1, 1, 1), mode:0
Regions [73 58 73 43 58 43 58 58 58 43 43 58 58 43 73 58 73 58 43 58 73 43 58 73
 73 73 73 42 57 57 58 38 42 42 52 62 58 53]
NEA 51°28'40.188392445132"N, 0°0'0.000000036000"E (Reference Coordinates)
NEA 51°28'40.188392445132"N, 0°0'0.000000035968"E (Label GCD Coordinates)
∂0.627073nm (roundtrip via GCD<->Barycentric)
∂0.627073nm (roundtrip via GCD<->Bary Regions)
∂0.627073nm (roundtrip via GCD<->Hex9 Label)
H9 (from array):0a02a2aaa22aa20a0a2a02a0000977a69935a8
H9.adr:4348683362836823268348322246337341885
H9.key:4348683362836823268348322246337341800
Reference BRY: 0.304133195545653323,-0.289722433974214599
Label RT  BRY: 0.304133195545653323,-0.289722433974214599

```

Needless to say, nobody will be seriously expecting to use this toolchain to 
geolocate distances as small as a few nanometres, but it provides an 
idea of where noise can be found creeping in.

### ex0062_seamstitch
Various metrics and investigations to ensure that borders, seams, and poles
behave well under stress.

### ex0063_grid
Examining local authalicity (equal-area) constraints on a single octant.
This uses the underlying triangular grid.  For the hexgrid variant, check
out ex0080

### ex0064_vertices
Various metrics and visualisations to map address roundtrip deviation.
There are tools for analysis that may be useful here.

### ex0075_osm_mesh
The osm mesh example relies upon OpenStreetMap and cartopy to retrieve and 
store a series of images based upon a single address. The purpose of which 
is to demonstrate how each level of the Grid address reduces the land area 
being addressed, and these are then used in ex0076. 

The meshes being stored are mere boundary sets in plate carree, from which 
we can later use as sample sources to depict the relevant half-hexagon.

While Stonehenge has been used as a reference, any geographic location may 
be similarly used.

This was all managed via half-hexagons and is now less ... interesting.

### ex0076_stonehenge.
Using the series of plate carree images retrieved in ex0075, this example 
now generates the relevant half-hexagon for each address, then using the 
relevant image as a sample source, generates the half-tile represented by 
the hexagon and it's mode. This could be authored better, but is reasonably 
straightforward to understand.

This was all managed via half-hexagons and is now less ... interesting.
The population examples (pr000) are probably more useful now.

### ex0080_authalics
This draws a hexagonal grid across the sphere, demonstrating the variation 
in area (from ideal) that any given hexagon has.  It's easily notable that 
there is far more variety at each of the six poles - but it's also very 
predictable. Tools have been developed for identifying degree of deviation.

### ex0081_areas
This uses two separate means to identify points on the sphere which ideally 
authalic. Finding acceptable authalic reference points is useful only for 
referencing at a given layer. This is ongoing work - but is reliable up to 
around layer 18 (small).

### ex0094_smp_bary
Compose pixel grids of a selection of octahedral faces.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere 
and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
The octahedral→spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.

### ex0094A_smp_bary
This is functionally no different from 0094, as we are currently not implementing warps.
Though it *does* do a roundtrip to simplex.

### ex0095_smp_grid
This demonstrates using a plate carree image as a sample source.
The source image is loaded and projected onto WGS84 Ellipsoid, and 
registered into a KDTree.
The octahedral net pixels are identified then a copy is projected to 
EllipsoidGCD, and query the KDTree for a sample value. 
These samples may then be used to display the Octahedral Net.
This is useful for authoring new nets.
There is another version of this in pr0005.

### ex0101_h9
This example renders each pixel into it's hexgrid address, and then colours 
the pixel according to a digit location, for the first 5 hexagon layers, for 
various nets. It demonstrates how the grid address maps onto the octahedron.

### ex0110_poly_neighbours
Here we take the reference location of Stonehenge, convert it to its region 
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
(Neighbour finding was a bit complex at first)

### ex0200_heatmap
Generate the tokyo heatmap as found on the readme.
This depends upon running the hh_heatmaps scripts beforehand!
(see the hh_heatmaps set below).

### ex0250_geotiff
This is a nice toy that takes the CONUS land usage dataset,
and, given a location in the US, generates the land usage
as a hex-grid. It's stand-alone, and uses several tricks,
including implementing an ad-hoc domain/projection. 
However, you will need osgeo and gdal to use it.
This is still a toy - I have cut corners when it comes to octant seams,
so New Orleans (at -90) is likely to fail, but it's not so hard to fix.



## ex4***
These are poc, quick tests extracted from __main__ in the code.

### ex4000_alg_packing
Some checks for the packing algorithm.

### ex4001_prj_akoctahedral
These are some sanity checks for AKOctahedral projection

### ex4002_h9_addressing
These are some sanity checks for h9/addressing

### ex4002_h9_addressing
These are some sanity checks for h9/addressing

## ex10***
Typically short, validation scenarios.

### ex_10001_octant_ids
Check the octant ids are aligned.

### ex_10003_region_rt
Roundtrip region addresses

### ex_10005_grid
Validate various features and settings of the region/cell grid.

### ex_10006_c2
Validate various features and settings of the c2 tables.

### ex_10007_hex_grid
hex address roundtrip feature validation.


##  hh_heatmaps
### hh_heatmaps/pr0001_csv
This converts a CSV into a np file.
The CSV may be downloaded from https://data.humdata.org/organization/meta
and consists of 3 values: longitude/latitude/populaton.
They represent sparse grid values at 30m.
* Input  
   * src/{file}_general_2020.csv 
* Output 
   * src/{file}_lon_lat_pop.npy

### hh_heatmaps/pr0002_prj
Here we take the file and forward project the GCD addresses,
converting them to barycentric octahedral co-ordinates, which are
x,y and octant (in cmp).
This uses root-finding, and is batch-processed across multiple cores.
* Input
    * src/{file}_lon_lat_pop.npy
* Output
    * src/{file}_bry.npy
    * src/{file}_bry_cmp.npy

### hh_heatmaps/pr0003_prj
(1) Load the population numpy data file
(2) Derive or resolve a boundary and project it onto the Barycentric Octahedral Net.
find the boundaries of the gcd, project and store, both as gcd and as 
barycentric, having added padding of 2.5% border.

* Input
    * src/{file}_lon_lat_pop.npy
* Output
    * src/{file}_pop_data.npy
    * src/{file}_lat_lon_bounds.npy
    * src/{file}_bounds_bry.npy
    * src/{file}__bounds_bry_cmp.npy

### hh_heatmaps/pr0004_gcd_img
Using gcd_bounds saved at pr0003;  retrieve an image from OpenStreetMap.
Then store it.


### hh_heatmaps/pr0005_align
Load the GCD quadrilateral, project onto Barycentric Net Coordinates.
Display it, and store it as a single value, along with
the octahedral rectangle coordinates.

### hh_heatmaps/pr0005_theta
Using gcd_bounds saved at pr0003; project the bounds onto barycentric, 
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

### hh_heatmaps/pr0006_grid or hh_heatmaps/pr0006_grid_alt
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

### hh_heatmaps/pr0007_hh_heatmap
Finally, we can generate the heatmap over the top of the grid image.
This may need fixing!
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
