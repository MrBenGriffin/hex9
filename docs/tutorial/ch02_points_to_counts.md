# Chapter 2 — From points to counts

**Task:** take a real dataset of point observations and count what lands in
each cell.

Chapter 1 encoded one coordinate. This chapter encodes 161 015 of them — a
population grid for Bhutan — and turns them into something the points alone
cannot give you: a summary with a definite spatial meaning.

## The dataset

`examples/src/bhutan_pop.npz` ships with the repository: one row per grid
point, holding latitude, longitude, and an estimate of how many people live in
the small patch around that point.

```python
import numpy as np

b = np.load('examples/src/bhutan_pop.npz')['bhutan_pop']
lat, lon, pop = b[:, 0], b[:, 1], b[:, 2]
print(f'{len(pop)} points, {pop.sum():.0f} people')
```

```text
161015 points, 780514 people
```

:::{margin}
```{admonition} Bhutan sits on a seam
:class: h9-scaffold

Bhutan straddles 90°E, which is an edge of the octahedron the grid is built
on. Nothing in this chapter has to know that: encoding is per-point, and the
addresses on either side of the seam are ordinary addresses. It becomes
interesting only when you draw the result — a later chapter.
```
:::

Three quarters of a million people is right for Bhutan, so the file passes the
first sanity check. What the points do *not* tell you is anything summary-like:
where people are concentrated, how much of the country is inhabited at all.
Every row is one small patch, silent about its neighbours.

## Encode everything, once

Chapter 1's batching advice applies immediately: both arguments are
array-like, so all 161 015 points go in one call, with one `Registrar` built
once. This takes a few seconds — it is the only heavy step in the chapter.

```python
from hhg9 import Registrar
from hhg9.h9.uuid_address import h9_encode, h9_bin

reg = Registrar()
u = h9_encode(lat, lon, reg=reg)
cells = np.array(h9_bin(u, 8))
```

Layer 8 is the ~1.2 km cell — the honest home, Chapter 1 argued, of a
two-decimal coordinate. It is a starting point, not the answer; choosing
properly is the last section of this chapter.

## Counting is a group-by

Here is the point of the whole exercise. A cell address is a *value* — it can
be compared for equality, sorted, hashed. So "how many people in each cell" is
not a spatial operation at all. It is the group-by you already know:

```python
uniq, inv = np.unique(cells, return_inverse=True)
people = np.bincount(inv, weights=pop)
print(f'{len(uniq)} occupied cells at layer 8')
print(f'{people.sum():.0f} people after binning')
```

```text
10298 occupied cells at layer 8
780514 people after binning
```

:::{margin}
```{admonition} The same thing, elsewhere
:class: h9-scaffold

In pandas this is `df.groupby('cell')['pop'].sum()`; in SQL it is
`GROUP BY cell`. Once the address column exists, any tool that can group by a
column can aggregate spatially — no spatial extension required.
```
:::

Notice what was *not* involved: no polygon file for Bhutan, no point-in-polygon
tests, no CRS negotiation, no edge cases where a point sits on a boundary
between two areas. The cells partition the sphere, so every point lands in
exactly one cell — nobody is lost, nobody is counted twice. The two sums above
agree to the last bit.

## Ask the result a question

The counts are indexed by addresses, and addresses decode. So the result can
answer questions directly:

```python
from hhg9.h9.uuid_address import h9_decode

top = int(np.argmax(people))
lats, lons = h9_decode([uniq[top]])
print(uniq[top])
print(f'{people[top]:.0f} people around {lats[0]:.4f}, {lons[0]:.4f}')
```

```text
54284673-1fff-ffff-ffff-fffffffffff2
9104 people around 27.4464, 89.6584
```

That is the Thimphu valley — the capital. No lookup table was consulted: the
busiest cell's own digits carry its position, exactly as in Chapter 1.

## Choosing the layer

Layer 8 was a guess. The right way to choose is to look at what the choice
does, and since the full-depth addresses in `u` are already in hand, sweeping
the layers is one line per layer:

```python
for L in range(5, 14):
    cl = np.array(h9_bin(u, L))
    uq, iv = np.unique(cl, return_inverse=True)
    ppl = np.bincount(iv, weights=pop)
    print(f'L{L}: occupied {len(uq):6d}  busiest {ppl.max():9.0f}')
```

| layer | cell across | occupied cells | busiest cell |
|---|---|---|---|
| 5 | 33 km | 70 | 127 768 |
| 6 | 11 km | 495 | 92 737 |
| 7 | 3.7 km | 2 950 | 35 997 |
| 8 | 1.2 km | 10 298 | 9 104 |
| 9 | 410 m | 28 998 | 2 708 |
| 10 | 137 m | 70 376 | 487 |
| 11 | 46 m | 140 464 | 122 |
| 12 | 15 m | 161 015 | 61 |
| 13 | 5 m | 161 015 | 61 |

Both ends of the table fail, and they fail differently.

Too coarse, and the summary stops summarising anything spatial: at layer 5 the
busiest single cell holds 127 768 people — a sixth of the country in one
33 km hexagon. That is a headline, not a map.

Too fine, and the summary stops summarising at all: by layer 12 there are
161 015 occupied cells — one per input point. You are no longer aggregating;
you are copying the input into different notation.

And then the table freezes. Layer 13 is *identical* to layer 12 — same cell
count, same maximum — because there is nothing left to merge. The dataset has
a resolution of its own, and the sweep just measured it.

:::{dropdown} The data's own grid, recovered from the counts
:class-container: h9-evidence

The input is a regular grid: the smallest spacing between distinct latitudes
in the file is 1.0 arcsecond — about 31 m on the ground. A layer-12 cell is
about 15 m across, smaller than that spacing, so no two grid points can share
one: every finer layer gives the same 161 015 singleton cells.

Which is worth pausing on. Nothing in the file *says* it is a 1-arcsecond
product; that is metadata you would normally hope someone recorded. Here the
layer sweep recovers it from the coordinates alone — the layer at which
occupancy saturates is the layer at which your data has run out of
information. Binning below it manufactures precision, which is Chapter 1's
sin in a new costume.
:::

For a national map of where people live, the useful range is plainly layers
6–8: thousands of cells, none of them dominant. The point is not that one of
them is correct — it is that the choice is now visible, cheap to explore, and
recorded in the result itself, because every address states its layer.

:::{admonition} Coming from H3 or S2?
:class: h9-compare

This workflow is the standard one: `latLngToCell` on every row, then
`GROUP BY`. No surprises. One detail to notice for later: the layer sweep
above ran entirely on the stored addresses — `h9_bin(u, L)` — without ever
touching the coordinates again. Truncating an address to a coarser cell is
exact in Hex9, which on a hexagonal grid is not an obvious thing to be able
to say: hexagons do not tile into bigger hexagons. That is Chapter 3.
:::

## Where this is going

The counts in this chapter are already a map; it just has not been drawn yet.
Drawn properly — cells filled by population on a log scale, over imagery, with
the coarser layers as a frame — the same binning looks like this:

```{figure} ../_static/ch02_bhutan_teaser.jpg
:alt: Bhutan population, binned to hex cells and rendered over satellite imagery
:width: 100%

The Bhutan grid from this chapter, binned at layer 7 and rendered by
`examples/ex0255_bhutan.py`. The valleys light up; the ridgelines stay dark.
Rendering is a later chapter — nothing here needed it.
```

## Next

The layer sweep leaned on something quietly remarkable: coarsening an address
never re-visited the coordinates, and never introduced error. Chapter 3 opens
that up — what parent and child mean on a grid whose cells do not nest, and
why truncation is exact anyway.
