
# warp/ — Status: Parked

(w10, w20, w30, w40, w49, w90)

## Overview

The warp subsystem was intended to add a higher-order, Monge–Ampère–driven 
geometric correction layer on top of the baseline octahedral projection (b_oct → s_oct → n_oct).

Work progressed through:

	•	w10 – centroid sampling
	•	w20 – Bernstein φ fit (authalicity preconditioning)
	•	w30 – export of preconditioned grids
	•	w40 – MA solve for ψ (nonlinear potential)
	•	w49 – warper integration
	•	w90 – diagnostics and testbed scripts

During experimentation a structural issue emerged: global delamination, 
along the octant edges, with parts of the domain pushed outward in opposite 
directions. This behaviour persisted across parameter sweeps 
(Dirichlet/Neumann weights, edge bands, cross-term gamma, Laplacian h, ridge λ, 
tethering, and homotopy schedules), indicating model-level inconsistency 
rather than tunable instability.

Because the underlying projection and subdivision system is in active stabilisation, 
maintaining the warp code in the current experimental state introduces unnecessary risk.

## Why It’s Parked

Warping via ψ(u,v) must satisfy simultaneously:

	•	local authalic correction (Monge–Ampère)
	•	global injectivity
	•	boundary correspondence across the entire octahedral net
	•	compatibility with existing domains, projections, and root-finding
	•	reversibility (warp ↔ unwarp)

The current formulation meets some but cannot satisfy all simultaneously without 
structural changes to the basis and the boundary model.
Symptoms include:

	•	edge delamination
	•	mirrored divergence at seam-adjacent points
	•	consistent loss or thinning of regions (e.g. UK)
	•	non-physical displacement magnitudes along seams

The issue is not numerical instability but incompatibility of the constraints 
with the current Bernstein-only representation and boundary assumptions.

Until the core of the project is fully stable, continuing to iterate on warp risks:

	•	breaking round-trip properties
	•	compromising test reliability
	•	causing regressions in the geoparametric grid
	•	introducing inconsistent behaviour into downstream systems (registrar, formats, domains)

## Why It’s Parked

Warping via ψ(u,v) must satisfy simultaneously:
	•	local authalic correction (Monge–Ampère)
	•	global injectivity
	•	boundary correspondence across the entire octahedral net
	•	compatibility with existing domains, projections, and root-finding
	•	reversibility (warp ↔ unwarp)

The current formulation meets some but cannot satisfy all simultaneously without structural changes to the basis and the boundary model.
Symptoms include:
	•	edge delamination
	•	mirrored divergence at seam-adjacent points
	•	consistent loss or thinning of regions (e.g. UK)
	•	non-physical displacement magnitudes along seams

The issue is not numerical instability but incompatibility of the constraints with the current Bernstein-only representation and boundary assumptions.

Until the core of the project is fully stable, continuing to iterate on warp risks:
	•	breaking round-trip properties
	•	compromising test reliability
	•	causing regressions in the geo-parametric grid
	•	introducing inconsistent behaviour into downstream systems (registrar, formats, domains)


## Resume Point

The recommended re-entry point is:

	1.	Re-establish a clean, stable w40_solve_monge_ampere that compiles and runs.
	2.	Add a dedicated boundary-condition layer (Dirichlet-fixing of a,b,c edges).
	3.	Add injectivity and displacement diagnostics before any corrective solver steps.
	4.	Build a reduced-degree prototype (n=4–6) with fixed boundaries to verify behaviour.
	5.	Only after successful injective warps → reintegrate into warper.py.

This will allow development to resume without reintroducing instability into the main codebase.

