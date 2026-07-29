# Tutorial

This tutorial is organised around **tasks**, not around the API. Each chapter
answers a question of the form *"how do I actually do X?"*, and the library
appears only as the means to that end.

You do not need to read it in order, but it is written so that you can.

The chapters are ordinary markdown — nothing to install, nothing to run — but
converted chapters are also *executable*: each is a jupytext-paired notebook
ending in a machine-verification cell that re-derives every printed value in
the chapter. To run one yourself, `pip install jupytext nbclient ipykernel`
and use `docs/tutorial/verify.sh`, or open the derived `.ipynb` in Jupyter.
Reading requires none of that.

## Two ways in

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} New to discrete global grids?
Start at Chapter 1 and read straight through. Background you may need is
carried in the gold asides; you can ignore them freely if a term is already
familiar.

+++
[Chapter 1 — Your first cell](ch01_first_cell.md)
:::

:::{grid-item-card} Already using H3, S2 or rHEALPix?
The mechanics in Chapters 1–2 will hold no surprises. The chapters where Hex9
behaves differently from what you are used to are the hierarchy and the
addressing.

+++
[Chapter 3 — Changing resolution](ch03_changing_resolution.md) ·
[Chapter 7 — Storing and querying](ch01_first_cell.md)
:::
::::

## Chapters

```{toctree}
:maxdepth: 1

ch01_first_cell
ch02_points_to_counts
ch03_changing_resolution
```

*Chapters 4–7 are outlined but not yet written.*
