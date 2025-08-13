# Hex addresses.
Find a way of taking all the hh-addresses that have mode 0 and moving them to mode 1 (?and then coalescing their sample values),
Then using those to render the relevant hexagon layers (instead of half-hexagons).

# Octant seams (neighbours)
I still need to calculate the seam-hopping neighbours.
This is probably best done by indicating those values which are no longer in context,
and recognising the C1 of the neighbour - routine.

# pr0004_theta.py
This should look to see which (horizontal/vertical) is most conformant,
and definitely try to get at least one straight edge.

# data clipping.
With Japan, looking at Kyushu, it would be better for me to elide the entire population data for the country to that within a set of manual boundaries.
This is roughly in place, but currently scattered in more than once place.

# Major tidy up of code.
There are, maybe, 3 different generations of code in the h9Engine class at the moment, which is far from ideal.

# Continue to rewrite / update documentation.
# More unit tests...
