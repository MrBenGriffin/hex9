## Hex9 Geometry Constants
###  Introduction

This document defines the geometric constants used across Hex9 and explains their relationships.
It provides enough context to understand how these constants are derived and why they’re chosen, without diving into the entire Hex9 system.
For a broader explanation of projections, grid indexing, and hierarchical 
addressing, please at the other documents!

###  Purpose
Choosing a single source of truth for, eg, √3, √2 and so on avoids small 
variations in the float64 ULP (Unit at Last Place) which often happens with 
composite calculations and can stabilise many downstream edge cases.

Much of the geometry here directly corresponds to the nature of the unit 
octahedron, which has an edge length of √2, and is the optimal intermediary 
for ellipsoidal projection onto the equilateral triangle tiling of the
the plane. This is because, of all triangular platonic solids, the 
vertices of the octahedron have an even number of edges. While the more 
common (but not exclusive) choice for global hex grid systems is the 
icosahedron, (because it affords less distortion in projection), here we 
have chosen to consider separation of concerns as a maxim, which frees us of 
some of the oddities that the icosahedron adds to continuous grids.

Given the inherent √3 relation to the equilateral, along with the √2 nature 
of the unit octahedron, we choose these as our fundamental constants.  

Other constants are derived from these, and serve to offer us reliable 
anchors for the remaining geometric calculations of the underlying grid.

Each (equilateral) face of the octahedron is subdivided into nine smaller 
triangles at each layer of the Hex9 grid. We call a nine-triangle group a 
`supercell` (parent), and tiles that are congruent with the 9 'children' 
within it are called `cell`. The equilateral tiling of the plane consists of 
cells. A cell at layer `i` becomes the supercell for layer `i+1`, creating a 
consistent hierarchical refinement.  Any given octahedral face itself is 
known as a `root supercell`.

The orientation of each cell within the planar tiling is defined by its mode:
	•	Mode Λ (1): triangle aligned with the root supercell
	•	Mode V (0): triangle rotated 180° relative to the root supercell

This mode distinction is important for barycentric addressing, 
projection transforms and neighbour relationships.

To support efficient addressing, we also define a rectilinear helper lattice:
integer steps (U,V) provide a consistent unit system for positioning cell 
barycentres in Cartesian space.  Every cell’s barycentre can be computed 
from its integer coordinates (x, y) as:

```
barycentre.x = x * H9K.lattice.U
barycentre.y = y * H9K.lattice.V
```

Not every lattice step (x, y) corresponds to a valid barycentre, but every 
barycentre has integer coordinates.  By convention, the barycentre of a 
supercell — regardless of its mode — is anchored at the origin.

Any cartesian address on the plane anchored by a supercell is called 
a `barycentric address`, and we talk freely about `barycentric co-ordinates`,
or projecting onto (or from) the `barycentric plane`.

| Symbol | Formula | Approx.  | Meaning / Usage                    | Status      |
|--------|---------|----------|------------------------------------|-------------|
| R3     | √3      | ≈ 1.732  | Equilateral triangle ratio         | Fundamental |
| W      | √2      | ≈ 1.414  | Supercell edge length (octahedron) | Fundamental |
| H      | W·R3/2  | ≈ 1.224  | Supercell triangle height          | Derived     |
| Ḣ      | H/3     | ≈ 0.408  | Vertical spacing of cells          | Derived     |
| Ẇ      | 2·Ḣ     | ≈ 0.816  | Barycentric offset of V cells      | Derived     |
| TR     | H/R3    | ≈ 0.707  | +X bound of Λ/V classifier plane   | Derived     |
| TL     | −TR     | ≈ −0.707 | −X bound of Λ/V classifier plane   | Derived     |
| RH     | R3/2    | ≈ 0.866  | Ratio of height to edge width      | Derived     |
| U      | W/6     | ≈ 0.235  | Horizontal lattice step size       | Derived     |
| V      | H/9     | ≈ 0.136  | Vertical lattice step size         | Derived     |

### Vertices
Vertex values are in H9K.Limits (eg `H9K.Limits.TR`)

    TR: Positive right-hand x-limit:                   H/R3
    TL: Negative left-hand x-limit:                   -TR
    ΛC: Mode 1 (Up) supercell barycentric ceiling:     2Ḣ
    VC: Mode 0 (Down) supercell barycentric ceiling:    Ḣ
    ΛF: Mode 1 (Up) supercell barycentric floor:       -Ḣ         
    VF: Mode 0 (Down) supercell barycentric floor:    -2Ḣ

The mode 0 (downward pointing) vertices of an octant/supercell are

    (0, VF), (TL, VC), (TR, VC)

The mode 1 (upward pointing) vertices of an octant/supercell are

    (0, ΛC), (TL, ΛF), (TR, ΛF)


###  Summary
These constants are referenced throughout Hex9’s projection, addressing, 
and lattice code. For further discussion of the hierarchical subdivision, 
address formats, and neighbour algorithms please read the documentation for 
those.


