# Probes

Small, self-contained diagnostic scripts (run from the repo root, e.g.
`python experimental/probes/probe_canon_symbolic.py`). Each prints its findings;
none write to the tree or the library.

## uint64 / canonicalisation set (2026-06-23)

- **`probe_u64_resolution.py`** — round-trip resolution of `ua` / `ua36` / `r35`.
  Regression check for the single-nibble-tail fix: `ua` (one u64) = L14 sub-metre.
- **`probe_tail_ka.py`** — the key/address (k/a) distinction. Raw path: p_mo set
  ~50% (k≠a); canonical/UUID path: p_mo≡0 (address≡bin); production `uk` u64 is
  degenerate (tail always 0).
- **`probe_canonicalise.py`** — characterises canonicalisation: ~50% need it; of
  those ~57% are a pure tail p_mo-flip, ~43% a short body cascade (1–2 wing
  digits); octant-switch rare.
- **`probe_canon_symbolic.py`** — the "always canonicalise" quick win.
  `canonicalise_regions()` is a region-level (packer-agnostic) canonicaliser
  mirroring `uuid_address._coalesce_bin`. The fold is symbolic
  (`region_neighbours`); only the octant-seam hop re-derives the chain
  geometrically (y-flip).

### Findings from `probe_canon_symbolic`

- In-octant canonicalisation is **fully symbolic and validated**: on a generic
  global grid it matches the `neighbours()` reference 100%, and the canonical
  property (leaf-parent mode == 0) holds for every in-scope point.
- With clean off-seam sampling, `canonicalise_regions` matches the reference
  **100% including ~40k genuine octant-hop folds** (the geometric y-flip hop is
  correct). Points placed *exactly* on a seam have ambiguous cell assignment and
  must not be used to measure agreement (they produced the earlier spurious 8%).
- The **octant seam** is still the only non-symbolic step. A uniform-cascade hop
  (e.g. Greenwich −1e-8 W, chain `[22,62,62…]→[73,43,43…]`) has
  `region_neighbours`' chain == the geometric flip chain exactly; but for general
  hops they match only ~10% — same physical cell, different octant *frame*. So
  reconciling them symbolically needs the `OctConst` c2-reflection
  (`nb_c2map`/`nb_c2p`), not just `oid_nb`.
- Mirror test (`scratchpad`): points a hair either side of a seam canonicalise
  consistently — N/S equator pair → identical chain in the *same* octant (one
  physical cell); E/W Greenwich pair → identical chain in *mirror* octants.

### On-seam → mode-0 convention (suspected, checked)

The "on-seam point always has a mode-0 parent" convention is **not enforced by
the projection** in either backend. On the equator (a seam at every longitude),
the raw leaf-parent mode equals the *octant's* mode (`oid_mo`): mode-0 in
NEA/NWP octants, **mode-1 in NEP/NWA**. The default `b_oct` backend is the
**C lib (libhex9)**, but C and pure-Python (`no_lib()`) AGREE on the equator
modes (48/48); they differ only at the antimeridian lon=−180 (C→oct1, PY→oct3 —
a wrap edge case). So this is not a C-vs-Python divergence; mode-0 is currently
achieved by *post-hoc canonicalisation*, not at projection time.
