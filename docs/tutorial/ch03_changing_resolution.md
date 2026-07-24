# Chapter 3 — Changing resolution

**Task:** you have addresses stored at full depth; move them to any other
layer — and know which of two different questions "what is this cell's
parent?" is asking.

Chapter 2 ended on a puzzle. The layer sweep ran on stored addresses alone —
`h9_bin(u, L)` — and never touched the coordinates again. That looked like
truncation: chop digits, get the coarser cell. This chapter is about why it is
not truncation, why it cannot be, and what you get instead — which turns out
to be more useful than the thing it replaces.

## One cell, three ways to write it

Carrying straight on from Chapter 2's session:

```python
from hhg9.h9.uuid_address import h9_bin, h9_label

c = h9_bin(u, 7, reg=reg)[0]
print(c)
print(h9_label(c))
print(h9_label(c, with_tail=False))
```

```text
55285678-ffff-ffff-ffff-fffffffffff1
55285678.1
55285678
```

Three spellings, three different standings:

| form | example | what it is |
|---|---|---|
| address | `55285678-ffff-…-fff1` | the cell's identity as a 128-bit UUID; decodes back to a place |
| address, printable | `55285678.1` | the same identity as text — *label* `.` *tail* — and still decodes |
| label | `55285678` | a unique name for the cell: safe to group and join on, but does not decode |

:::{margin}
```{admonition} You have met tails already
:class: h9-scaffold

Chapter 1's bin ended `…fff0`; Chapter 2's busiest cell ended `…fff2`. That
final nibble was the tail all along — it rides in the last position of the
UUID, after the sentinel padding.
```
:::

The **tail** — the digit after the dot — is one nibble of metadata recording
how the cell is registered on its parents. It is never an address digit;
only the label's digits take part in geometry. Why a cell needs such a thing
is the heart of this chapter, and the short version is the same as a house on
a boundary street: when geometry genuinely straddles, something must record
which side owns you, by convention rather than by measurement.

The label's standing is worth being precise about, because it is unusual. No
two cells at a layer share a label, so equality is trustworthy: Chapter 2's
group-by would have worked just as well on labels as on addresses. What the
bare label cannot do is *decode* — without the tail the geometry cannot be
rebuilt — and, as the next section shows, what it invites you to do with its
prefixes is exactly the thing you must not.

:::{margin}
```{admonition} "Unique" is checked, not assumed
:class: h9-scaffold

Machine-verified: every cell at layers 1–3 globally (9 828 of them), and
every cell set in this tutorial down to layer 12, has a distinct label.
```
:::

There is a fourth spelling, of a different character entirely:

```python
from hhg9.h9.uuid_address import h9_curve_uuid, h9_curve_label, h9_curve_index

cv = h9_curve_uuid([c])[0]
print(h9_curve_label(cv))
print(f'{h9_curve_index([c])[0]} of {12 * 9**7}')
```

```text
c23442752
11424773 of 57395628
```

The **curve label** is the cell's position on a space-filling curve that
visits every layer-7 cell exactly once — this cell is number 11 424 773 of
the 57 395 628 that exist. It earns its keep at the end of the chapter.

## The trap: a label's prefix is not its owner

A hexagon cannot be tiled by hexagons. Subdivide a Hex9 cell by nine and six
children sit wholly inside it — but three land on the rim, each shared with a
neighbouring parent. Every layer, every cell, a third of all children
straddle. Each straddling child has a valid spelling from each of its two
parents, and exactly one of the two is its **owner** — that is what the tail
records, and it is a convention, like the boundary house's postcode.

The consequence: a cell's label starts with digits that need not name its
owner. With the layer-8 cells from Chapter 2:

```python
from hhg9.h9.uuid_address import h9_cell_ancestor

cells8 = sorted(set(h9_bin(u, 8, reg=reg)))
owners = h9_cell_ancestor(cells8, 7)
diff = [(c8, o7) for c8, o7 in zip(cells8, owners)
        if h9_label(c8, with_tail=False)[:8] != h9_label(o7, with_tail=False)]
print(f'{len(diff)} of {len(cells8)} cells: owner is not the digit prefix')
c8, o7 = diff[0]
print('cell :', h9_label(c8))
print('owner:', h9_label(o7))
```

```text
1535 of 10298 cells: owner is not the digit prefix
cell : 542622167.0
owner: 54262276.0
```

Look closely at the pair: the owner is not the cell's first eight digits with
the last one dropped — it differs *in the middle* (`…216…` against `…276…`).
Chopping digits off `542622167` would name a different hexagon than the one
this cell actually sits in. For roughly a sixth of all cells — 1 535 of
10 298 here — string truncation quietly gives the wrong answer.

:::{margin}
```{admonition} Why a sixth?
:class: h9-scaffold

Three of every nine children straddle two parents, and the spelling routes
through one of the two — so about half of the straddlers, one sixth of all
cells, are spelled through the parent that does *not* own them.
```
:::

This is why `h9_bin` decodes and re-bins rather than chopping: it answers
*which coarser cell contains this*, not *what did the spelling look like*.

The trap is worth naming because it is seductive: the prefix is right five
times out of six, so it survives casual testing and fails in production,
quietly, on a sixth of your data. Humans fall into it; so, just as reliably,
do AI coding assistants asked to "get the parent cell" — the pattern *shorter
string = coarser cell* is true in almost every other hierarchical grid, and
they import it.

## Two verbs

So there are two honest operations here, and they answer different questions:

- **Ownership** — *which layer-L cell contains this?* This is the verb for
  aggregation, joins, and deduplication: `h9_bin` asks it of a point,
  `h9_cell_ancestor` of a cell, both exact from any depth. Everything in
  Chapter 2 was this verb.
- **Lineage** — *what path do the digits spell?* This is the verb for
  navigation and for ordering, one parent-step at a time.

The two agree completely when you roll up from full depth:

```python
bins7 = h9_bin(u, 7, reg=reg)     # where is each point, at layer 7
anc7 = h9_cell_ancestor(u, 7)     # who owns each point's full-depth cell
print('disagreements:', sum(a != o for a, o in zip(bins7, anc7)), 'of', len(u))
```

```text
disagreements: 0 of 161015
```

Neither verb is approximate. The trap — and it is a trap with history in
grid systems generally — is using the lineage verb for an ownership job.

:::{admonition} Coming from H3 or S2?
:class: h9-compare

In H3, `cell_to_parent` *is* digit truncation, and H3's documentation is
straightforward that child cells are only approximately contained in their
parents: rolling points up from resolution 15 to 5 places about 6.5% of them
in a parent that does not contain them (measured 6.50% in this repository's
commutation study, `docs/dggs/dggs_commute.py`, matching the analytic ≈6.52%
bound). That is not carelessness — with aperture 7 there is no exact roll-up
to be had, and H3 trades it knowingly for other strengths. Hex9's design
keeps the two verbs as two operations, so the aggregation verb can be exact:
the same study measures its deep roll-up at 0.00%, at every layer. S2, on a
square grid, has no such distinction to make — squares nest.
:::

:::{margin}
```{admonition} Why exactness is available at all
:class: h9-scaffold

Not because Hex9's hexagons nest — no hexagons do. One storey below the
hexagons sits a tree of *half-hex* cells that tile their parents exactly,
nine to one, at every layer. Ownership questions are answered on that exact
tree, and a hexagon — which is a registered pair of half-hexes — reads its
answer off at the end. Addressing, ancestry and ownership are the same
arithmetic there.
```
:::

## Nine children, three of them shared

Downward is the same story. Every cell owns exactly nine children: six
interior, plus three of the rim-straddlers — including, sometimes, straddlers
whose label is spelled through the *other* parent. The printable address
round-trips via
`h9_from_label`, so we can pick up the parent from the example above:

```python
from hhg9.h9.uuid_address import h9_from_label, h9_descendants, h9_dec

b_oct = reg.domain('b_oct')
p = h9_from_label('54262216.4')
kids = h9_descendants(h9_dec([p], b_oct), 7, 1, reg=reg)[0]
pb = h9_label(p, with_tail=False)
fosters = [k for k in kids if not h9_label(k, with_tail=False).startswith(pb)]
print(len(kids), 'children,', len(fosters), 'spelled through a different parent')
f = fosters[0]
print('child :', h9_label(f), ' curve', h9_curve_label(h9_curve_uuid([f])[0]))
print('parent:', h9_label(p), '  curve', h9_curve_label(h9_curve_uuid([p])[0]))
```

```text
9 children, 3 spelled through a different parent
child : 542622565.0  curve c235033074
parent: 54262216.4   curve c23503307
```

The child's label shares almost nothing with its parent's — `542622565`
against `54262216` — and yet look at the curve labels: the child is the
parent's plus one digit. Where the h9 spelling breaks, the curve does not.

## The curve: where truncation is honest

That is the curve label's whole appeal. Its digits are ranks in the
*ownership* tree — each cell's nine owned children, foster spellings and
all, occupy one contiguous block:

```python
ki = sorted(h9_curve_index(kids))
pi = h9_curve_index([p])[0]
print('parent index', pi)
print('child indices', ki[0], '..', ki[-1],
      '— contiguous:', ki == list(range(9 * pi, 9 * pi + 9)))
```

```text
parent index 11457943
child indices 103121487 .. 103121495 — contiguous: True
```

So on the curve, dropping one digit is always the true owning parent — the
prefix operation the h9 label could not support. Sorted curve labels put every
family in one run, which makes coarse-to-fine layouts and range scans natural;
that is Chapter 7's territory.

One caution keeps the two-verbs lesson in force. One step is always the true
parent, but *iterating* steps walks the lineage chain — parent of parent of
parent — and ownership is not transitive on hexagons. Truncate all the way
down from full depth and the drift shows:

```python
from hhg9.h9.uuid_address import h9_curve_bin

cu = h9_curve_uuid(u)             # full-depth curve addresses
own = h9_curve_uuid(anc7)         # each point's owner, in curve form
trunc = h9_curve_bin(cu, 7)       # 23 digits dropped
m = sum(t != o for t, o in zip(trunc, own))
print(f'{m} of {len(u)} points ({m / len(u):.1%}) truncate to a cell '
      f'that does not contain them')
```

```text
26804 of 161015 points (16.6%) truncate to a cell that does not contain them
```

The same sixth again — the straddler band, compounded down the chain. Nothing
here is broken: each single step was a true parent. It is simply the other
verb. Rule of thumb: **curve prefixes for ordering and navigation; `h9_bin`
and `h9_cell_ancestor` for anything that aggregates.**

## Next

So far every cell arrived by encoding a data point. Chapter 4 goes the other
way: start from a shape — Bhutan's border — and produce the cells that cover
it, inhabited or not.
