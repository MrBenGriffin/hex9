A vast amount of work has been accomplished.
The system in general is in place, but it needs tidying
Documentation needs finishing - especially for geodesic.

In May '25 I realised that the planar enumeration would not work
so had to re-work the entire enumeration for octahedral.
Once that worked, it occurred that I could use the grid itself
for refining the address values.

My [new documentation](enumeration.md) 

[Documentation for the examples](examples/examples.md)

This project is centred on the idea of hierarchic hexagonal grids, with the 
primary insight that, while one cannot tile hexagons with hexagons one can 
tile half-hexagons with half-hexagons.  When one is looking at the 1:9 
ratio, there are 49 such tilings of which 1 pair works particularly well for 
hierarchic hexagonal tiling.

![49_tilings](images/49_tilings.png)

The use of half-hexagons ('regular' trapezoids composed of three equilateral 
triangles) as the primary fundamental resolves many of the issues that arise 
from hexagonal hierarchies.  

The work I did back in 2010 or so [past research](assets/docs/past.md) was 
based on something very similar to the H3 methods currently funded by Uber 
(the ride company), but I didn't like the ragged edges, and rotations 
required at each layer - moreover at that time I did not discover an address 
system that worked intuitively.

![h9a9.png](assets/docs/h9a9.png)

Below is the basic unit hexagon, showing its division into the 18 half-hexagons that compose it.  
The numbering is one way of indexing the half-hexagons, and is suitable for 
planar tiling.  When tiling the octahedron, one must adopt something a 
little more complex (only a small amount) in order to address edge 
transitivity. 

![index_units.png](assets/docs/index_units.png)

The plane can be tiled using the following:
![tiling.png](assets/docs/tiling.png)

A hexagonal grid hierarchy can be seen below, with the outer hexagon in white, 
then the successive lower hierarchies in green, blue and red. The hierarchy is unlimited in depth.

![hierarchy](assets/docs/hierarchy.jpg)

While independent of the display projection, the method *cannot* be used on any polyhedron projection, as the fundamental 
shape involves chained mirror-pairs:  It can only work on polyhedra 
which have an even number of edges attached to each vertex.
Therefore, it does NOT lend itself well when using global addresses on  
icosahedral maps, that have five edges from each vertex.  

This is not a major issue, however - the Octahedron is highly suitable for 
this hierarchy, and, though the octahedron itself tends towards greater 
distortion when used as a projection, for the purposes of generating 
hexagonal grid addresses, it works perfectly adequately.

![Octahedral Projection](images/net_2700.jpg)

The following shows an early version of the grid focussed on London.

![gis](assets/docs/gis.jpg)

### Grid referencing.
We can use an address system that resolves hierarchy and location extremely 
easily. One thing that I like is that we can use subtended regions rather 
than axis-oriented addresses. Why is that nice?  It offers a few benefits
- first of all there is only one string (unlike, EG lat/long or OS grids - which still confuse those not familiar with 
), and where locality can be kept relevant without having to consider a remote origin.

Another 'feature', is that from any given root, the length of the address tells us about the level of hierarchy. 
Moreover, merely by shortening the address, we may derive the layer ancestry.

Grid coordinates are best done, for this, using base 9, and using a signifier for the half-hex specialisation.
One can work out the entire half-hex address from a given hex address - but it does require following some rules.
