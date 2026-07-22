# Chapter 1 — Your first cell

**Task:** take a location on the Earth and find the grid cell that contains it.

That is the whole of this chapter. Everything else in the tutorial — binning,
covering a region, joining two datasets — is built out of this one operation.

## Encode a point

```python
from hhg9.h9.uuid_address import h9_encode

print(h9_encode([52.520934], [13.405314]))
```

```text
[UUID('43472104-7065-3436-4868-812417650202')]
```

:::{margin}
```{admonition} Latitude first
:class: h9-scaffold

Coordinates go in as **latitude, longitude** — the order they are spoken and
written, and the opposite of the `x, y` order used by GeoJSON and PostGIS.
```
:::

That UUID *is* the cell. It is not a key into a table somewhere: the digits are
the address, and they decode back to a position without consulting anything.

```python
from hhg9.h9.uuid_address import h9_decode

lats, lons = h9_decode(h9_encode([52.520934], [13.405314]))
print(f'{lats[0]:.9f}, {lons[0]:.9f}')
```

```text
52.520934000, 13.405314000
```

Nine decimal places, and the input comes back unchanged.

## A coordinate does not say how big it is

`52.52, 13.40` is a perfectly good location, and it is also perfectly silent
about what it claims. It might be a reading good to the nearest kilometre. It
might be a millimetre-accurate survey value that happens to end in zeros. The
notation is identical either way: every coordinate carries an implied
uncertainty that it never states, and the reader is left to guess it.

A cell does not leave that open. It says *this area, right here* — a definite
interior, a definite edge, pegged by six vertices. Two points in the same cell
are not approximately together; by the grid's definition of place, they are in
the same place.

That is the trade on offer. You give up the pretence of a dimensionless point,
and in exchange your precision stops being something the reader has to infer
from how many digits you happened to type.

:::{margin}
```{admonition} Decimal degrees, in metres
:class: h9-scaffold

| places | ground distance |
|---|---|
| 2 | ~1.1 km |
| 4 | ~11 m |
| 6 | ~0.11 m |
| 8 | ~1.1 mm |
| 9 | ~0.11 mm |
| 15 | ~1.1 Å |

An atom is about 1 Å across. Quoting fifteen decimal places for a street
address asserts a precision a thousand times finer than the paving.
```
:::

### Pick the layer that matches your data

Because each layer subdivides by nine, choosing a layer *is* choosing how much
precision you are willing to assert. There are 12 cells at layer 0 and 12·9ᴸ at
layer L:

| layer | cells | cell across | matches |
|---|---|---|---|
| 0 | 12 | 8 090 km | — |
| 4 | 78 732 | 100 km | — |
| 8 | 5.2 × 10⁸ | 1.2 km | a 2-decimal coordinate |
| 12 | 3.4 × 10¹² | 15 m | consumer GNSS |
| 16 | 2.2 × 10¹⁶ | 0.19 m | a 6-decimal coordinate |
| 20 | 1.5 × 10²⁰ | 2.3 mm | survey / RTK |
| 30 | 5.1 × 10²⁹ | 39 nm | full UUID depth |

So `52.52, 13.40` — two decimals — is honestly a layer-8 cell, about 1.2 km
across. Encoding it at layer 30 does not make it more accurate; it just stops
recording that you never knew the last twenty digits.

`h9_bin` truncates an address to a given layer, which is how you say what you
actually mean:

```python
import numpy as np
from hhg9.h9.uuid_address import h9_encode, h9_bin

lat = np.array([52.5203, 52.5199, 52.5206, 52.5201])
lon = np.array([13.4003, 13.3998, 13.4006, 13.4000])

u = h9_encode(lat, lon)
print(len(set(u)), 'distinct cells at layer 30')
print(len(set(h9_bin(u, 8))), 'distinct cell at layer 8')
print(h9_bin(u, 8)[0])
```

```text
4 distinct cells at layer 30
1 distinct cell at layer 8
43472104-7fff-ffff-ffff-fffffffffff0
```

Four separate places at full depth; one place at layer 8. The trailing `f`s are
visible in the address — a coarser cell is a shorter address, padded out.

Note that this is a real boundary, not a rounding. Points a few metres apart
can fall into different layer-8 cells if the edge runs between them, exactly as
two adjacent postcodes differ across a street. That edge is the point: it is
what makes the area explicit.

:::{admonition} Then why does a full address go to layer 30?
:class: h9-scaffold

Not because anyone can measure 39 nm. Layer 30 is the depth at which the
address is *losslessly invertible* — it is a container sized to hold whatever
precision you have, so that encoding never becomes the thing that lost your
data. What you should then store is a truncation matched to your real
uncertainty.

Worth keeping in perspective: consumer GNSS is good to a few metres, RTK to a
centimetre or two, and geodetic work to millimetres. Tectonic plates move a few
centimetres a year, so a coordinate quoted without an epoch quietly decays past
centimetre precision on its own.
:::

### Measuring, rather than eyeballing

Printing the recovered coordinates with numpy's defaults shows eight decimals,
which would suggest the round trip lost about a millimetre. It did not — that
is just the display. And as we saw above, nine decimals shows no loss at all.
You cannot settle this by counting digits.

Measure the residual on the ground instead:

```python
import numpy as np
from hhg9.h9.uuid_address import h9_encode, h9_decode
from hhg9.algorithms.distance import wgs84

src = np.array([[52.520934, 13.405314]])
lats, lons = h9_decode(h9_encode(src[:, 0], src[:, 1]))
print(f'{wgs84(src, np.column_stack([lats, lons]))[0]:.3e} m')
```

```text
1.841e-08 m
```

Eighteen nanometres. That is not floating-point noise, and it is not luck —
it is the size of the cell.

:::{dropdown} Why eighteen nanometres, and not zero?
:class-container: h9-evidence

A full UUID address is the canonical bin at `UUID_DEPTH`, which is **30**. Every
layer subdivides, so a layer-30 cell is extremely small — and encoding is
quantisation, so decoding returns the cell's representative point, not the point
you put in. The residual you measure is therefore the cell's own extent.

Sampling 400 points scattered over a 0.11 m box around the same location:

| quantity | value |
|---|---|
| median residual | 1.30 × 10⁻⁸ m |
| maximum residual | 2.20 × 10⁻⁸ m |
| distinct cells | 400 of 400 |

Each residual is the distance from a point to its own cell's representative, so
the set of them bounds the cell radius from below. A layer-30 cell averages
39 nm across, giving a radius near 20 nm — which is exactly where the measured
maximum sits. Every one of those 400 points landed in a cell of its own.

For comparison, float64 spacing at latitude 52.5 is about 8 × 10⁻¹⁰ m, an order
of magnitude below the residual — which is how we know we are measuring the grid
and not the arithmetic.

The practical consequence: at full depth the address is, for any purpose you
have, the point. Cells only start behaving like *areas* when you truncate the
address, which is Chapter 3.
:::

:::{admonition} Coming from H3 or S2?
:class: h9-compare

`h9_encode` is the analogue of `latLngToCell` / `S2CellId::FromLatLng`, and at
this level it holds no surprises. Two differences worth parking for later: the
address is a 128-bit UUID rather than a 64-bit integer, so full depth is much
finer than you may be used to; and truncating that address is an exact
operation rather than an approximate one. Both are Chapter 3.
:::

## A note on batching

`h9_encode` builds a `Registrar` — the object that manages coordinate domains
and the projections between them — on every call when you do not supply one.
That is fine once, and wasteful in a loop.

Both arguments are array-like, so the first answer is to encode in bulk rather
than one point at a time. Where you genuinely need repeated calls, construct
the registrar once and pass it in:

```python
from hhg9 import Registrar
reg = Registrar()
u1 = h9_encode(lats_a, lons_a, reg=reg)
u2 = h9_encode(lats_b, lons_b, reg=reg)
```

## Next

Chapter 2 takes a real dataset, encodes every point in it, and counts what
lands in each cell — which means finally caring about resolution.
