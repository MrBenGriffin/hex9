## 3. Mode Transport

  Section 2 established that every edge crossing carries a mode flip. The transport
  operator on each edge is that flip — the orientation difference between the two
  faces sharing the edge, inherited directly from the intrinsic orientability of §1.
  It is combinatorial in origin: no metric or embedding is required to define it.

  We now ask what it means for this transport to be globally consistent.

  Define a **mode transport** as a rule that assigns, to any path through the
  triangular field (a sequence of edge crossings from face to face), a net mode
  change in ℤ₂. The transport is **consistent** if the net change depends only on
  the endpoints of the path — not on the route taken; equivalently, traversing any
  closed loop returns the mode to its starting value. This is the condition
  differential geometry calls trivial holonomy and gauge theory calls a flat ℤ₂
  connection; we will simply say the transport is **flat**. On any orientable
  surface flatness is equivalent to the existence of a globally consistent mode
  field: fix the mode of any one face and every other face's mode follows. Any
  refinement must preserve flatness; otherwise it introduces a transport defect.

  The requirement becomes non-trivial when we demand that consistency extends to all
  refinements of the triangulation. A refinement introduces new faces, edges, and
  vertices — and the transport must remain consistent at every scale. This is the
  content of Axiom 4, and its consequences for the admissible aperture class are
  developed in §6.

  What the transport implies immediately is simpler: any structure built over the
  triangular field — any labelling, orientation, or subdivision — must account for
  the mode flip at every edge crossing, or introduce a defect. The mode is not merely
  a local colouring; it is a global invariant that any grid must either propagate
  correctly or break.

  At individual edges the transport is binary and clean. At vertices — where multiple
  edges meet — the accumulated transports must also close consistently. What this
  implies for the global topology is the subject of §4.
