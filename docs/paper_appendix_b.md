## Appendix B — OGC / ISO Terminology Mapping

This appendix maps Hex9 vocabulary to the two standards the dual claim of §9
engages: OGC Abstract Specification Topic 21 (20-040r3, 2021) for the DGGS
role, and ISO 19111:2019 for the CRS role. The full taxonomy is maintained in
the implementation glossary; the tables here cover the terms a standards
reviewer needs.

Status codes: **C** — confident (the standard defines the term with a direct
correspondence); **P** — provisional (a reasonable correspondence whose exact
wording is to be confirmed against the spec text); **—** — Hex9-specific, with
no standard equivalent.

A vocabulary note on Topic 21: the 2021 edition (Part 1) renamed the 2017
edition's *cell* to *zone*, and a cell identifier to a *zone identifier* (ZID).
This paper keeps "cell" in prose for readability; the OGC-facing term is "zone".

### Hex9 → OGC Topic 21 (DGGS)

| Hex9 term | OGC Topic 21 term | Status |
|---|---|---|
| Hex9 (the system) | Discrete Global Grid System (DGGS) | C |
| `x_grid` at level L | Discrete Global Grid (DGG) — one tessellation / level | C |
| `x_cell` | zone (2021); cell (2017) | C |
| `x_adr` / `x_list` | zone identifier (ZID) | C |
| octant (8 seed faces) | base polyhedron face / level-0 partition | P |
| 12 root `x_cells` | level-0 zones | C |
| refinement level L | refinement level / resolution | C |
| aperture 9 (shifted) | aperture (refinement ratio) | C |
| `x_dig` | sub-zone index within a parent zone | P |
| parent/child `x_cell` | zone hierarchy / containment | C |
| AK+Warp realisation | DGGS Reference Frame (cell geometry on the globe) | P |
| mode (0/1 parity) | — (internal face orientation) | — |
| `c2` edge label | sub-face identifier | P |
| `d_cell` (half-hexagon) | — (sub-cell primitive) | — |

### Hex9 → ISO 19111:2019 (referencing by coordinates)

| Hex9 term | ISO 19111 term | Status |
|---|---|---|
| WGS84 reference ellipsoid | ellipsoid / geodetic reference frame (datum) | C |
| Prime Meridian anchoring (Axiom 7) | prime meridian | C |
| `x_adr` carried to the limit (§10e, App. A) | coordinate (a point in a CRS) | C |
| Hex9 as a locating system | coordinate reference system (CRS) | P |
| b_oct | a coordinate system realised by AK+Warp | P |
| encode / decode (point ↔ address) | coordinate operation / conversion | P |
| octant 2D plane coordinates | coordinate system (CS) axes | P |

The dual claim of §9, §10e, and Appendix A is exactly the bridge between these
two tables: one Hex9 address is a Topic 21 zone identifier when truncated at a
level and an ISO 19111 coordinate in the limit. Final verification of the exact
spec wording against the current editions of both standards is a pre-submission
task (§15).
