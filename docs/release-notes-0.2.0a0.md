# hhg9 0.2.0a0

*Alpha release. Hierarchical hexagonal grid on an octahedral reference ellipsoid.*

## ⚠️ Breaking changes

- **New UUID addressing mechanism.** Hex addresses now use a redesigned UUID
  scheme with canonicalisation of both the `uint64` and UUID forms. Addresses
  produced by 0.1.x are **not compatible** — re-encode any persisted IDs.
  (`915c9f0`, `20dcc77`, `10fb3f1`, `889cd14`, `5690bbc`)

## Highlights

- **libhex9 acceleration.** Optional C++ backend gives ~70× faster lat/lon →
  UUID encoding. Pure-Python path remains the default. (`c93eb0a`, `accel/libhex9.py`)
- **Authalic warp, retrained.** F6-retrained WGS84 and Sphere L5 warp fields
  (edge-tangent, exact CT tangency) with <0.1 % global area deviation — residual
  concentrated at the six octahedral vertices; support for alternative ellipsoids
  beyond WGS84. (`a68c2d6`, `a01a8b8`)
- **Shared-vertex hex meshes.** New `HexMesh` for delamination-free hex grids;
  region-clipped mesh build with no global enumeration. (`6c8b8f0`)
- **Composition.** Layer-overlay `Composition` class for compositing hex layers.
- **Densified grid** and assorted geometry utilities (`points_in_triangles`,
  `ensure_cw`, net-polygon helpers).
- **RDE tooling** for distortion analysis; numerous new examples incl. hex SVG
  rendering.

## Fixes & internals

- Canonicalisation: `seam = oid_mo = 0` flattening; reverted seam-hop to
  geometric y-flip re-decompose (`6aa166e`, `5690bbc`).
- Extensive test-suite expansion — **464 passing**, 18 skipped.
- PostGIS/QGIS SQL refinements; PROJ CRS groundwork.

## Packaging

- Description updated to "octahedral reference ellipsoid".
- Distribution trimmed: stale `WGS84_pkdo_grid_80.npz` / `WGS84_pkdo_warp.npz`
  removed; wheel ships only the two runtime warp fields (~9.4 MB).

## Install

```
pip install hhg9==0.2.0a0
```

Full commit log: `0.1.2a0..0.2.0a0`.
