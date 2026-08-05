// h9_hinged_hex.scad — one Hex9 cell at true Earth scale, hinged on its long diagonal.
//
// The long diagonal of a Hex9 cell lies on an octahedron edge, so the fold angle
// is the octahedral dihedral, acos(-1/3) = 109.4712°. This prints flat as a single
// piece with a living hinge: a V-groove whose bevel faces are cut at the half-fold
// angle 35.2644°. Fold it shut and the faces butt flush at exactly the dihedral —
// the mechanical stop IS the angle the construction demands. Magnet pockets in the
// bevel faces (glue in attracting pairs) hold it closed; open, it lies flat.
//
// Scale: `level` sets true Earth size via the authalic-ideal cell area
// A = 4*pi*R_auth^2 / (12 * 9^level). At level 16: side 93.963 mm, long diameter
// 187.926 mm, across flats 162.749 mm — fits a 220 mm bed. (Level 18 gives a
// 20.9 mm pocket charm.) The regular hexagon is the equal-area idealisation:
// real cells are quasi-authalic and not exactly regular in shape.
//
// Hinge exactness: gap0, the half-width of the hinge land, is DERIVED, not free.
// Treating the web as bending about its mid-plane, gap0 = (web/2)*tan(fold) makes
// each rotated bevel face land exactly in the central plane, so face contact
// occurs at the exact dihedral. `face_clearance` backs the faces off for plastic
// squish (each 0.05 mm stops ~0.5 deg short — tune on a test print, aim for 0).
//
// Printing: flat side down. The z=0 face is the OUTER (map) side and comes off
// the bed smooth; groove, bosses and magnet pockets face up. No supports. A PETG
// or PP hinge survives many folds; PLA manages a handful — fold gently once the
// magnets are in. Magnets: 6 x 2 mm discs, pockets sized 6.2 x 2.1 with a 0.2 mm
// proud face; check polarity (attracting across the seam) before gluing.

/* [Model] */
mode = "flat"; // [flat: print-in-place hinged, folded: display solid at the dihedral]
level = 16; // Hex9 level for true-scale sizing
label_text = true; // engrave "Hex9 L<level>" on the inner face

/* [Engraving] */
// Map engraving on the outer (bed-side) face: land edges + L1 cell boundaries,
// generated per tile by gen_engrave.py into engrave_data.scad. Coordinates are
// emitted as the OUTSIDE view, so they are mirrored in X here — the underside
// is seen mirror-wise from below, and two mirrors read correctly on the print.
engrave = true;
engrave_depth = 0.4; // groove depth into the map face, mm

include <engrave_data.scad>; // engrave_polys / engrave_side / engrave_address

/* [Plate] */
thickness = 3.0; // plate thickness, mm
web = 0.6; // living-hinge web thickness, mm (2-3 print layers)
face_clearance = 0.0; // extra half-gap, mm; >0 stops short of the exact dihedral

/* [Magnets] */
use_magnets = true;
magnet_d = 6.2; // pocket diameter for 6 mm discs
magnet_depth = 2.1; // pocket depth for 2 mm discs
boss_h = 5.0; // boss height above the plate (deepens the bevel face for pockets)
boss_w = 14.0; // boss width along the hinge
boss_d = 9.0; // boss depth away from the hinge

/* [Quality] */
$fn = 48;

// ---- derived geometry ------------------------------------------------------
R_AUTH = 6371007.1810; // WGS84 authalic radius, m
DIHEDRAL = acos(-1/3); // 109.4712 deg
FOLD = (180 - DIHEDRAL) / 2; // 35.2644 deg per half

cell_area = 4 * PI * R_AUTH * R_AUTH / (12 * pow(9, level)); // m^2
s = 1000 * sqrt(2 * cell_area / (3 * sqrt(3))); // hexagon side, mm
h = s * sqrt(3) / 2; // half across-flats, mm

gap0 = (web / 2) * tan(FOLD) + face_clearance; // exact-stop hinge half-width
face_top = thickness + boss_h;
cut_top = face_top + 2;
boss_x = [-s / 2, s / 2];

echo(str("Hex9 L", level, ": side ", s, " mm, long diameter ", 2 * s,
         " mm, across flats ", 2 * h, " mm, hinge land ", 2 * gap0, " mm"));

// ---- one half (+y), hinge along the x axis ---------------------------------
module half_body() {
    linear_extrude(thickness)
        polygon([[s, 0], [s / 2, h], [-s / 2, h], [-s, 0]]);
    if (use_magnets)
        for (xb = boss_x)
            translate([xb - boss_w / 2, 0, thickness - 0.2])
                cube([boss_w, boss_d, boss_h + 0.2]);
}

// V-groove bevel: removes everything on this half's hinge side above the web,
// leaving the bevel face at angle FOLD from vertical. Cuts plate and bosses alike.
module bevel_cut() {
    translate([-s - 2, 0, 0]) rotate([90, 0, 90])
        linear_extrude(2 * s + 4)
            polygon([[-1, web], [gap0, web],
                     [gap0 + (cut_top - web) * tan(FOLD), cut_top], [-1, cut_top]]);
}

// Magnet pocket, drilled normal to the bevel face of the +y half.
module pocket(xb) {
    z_c = web + 0.58 * (face_top - web);
    y_c = gap0 + (z_c - web) * tan(FOLD);
    // rotate([-(90+FOLD)]) maps +z onto (0, cos FOLD, -sin FOLD): into the material.
    translate([xb, y_c, z_c]) rotate([-(90 + FOLD), 0, 0])
        translate([0, 0, -0.1]) cylinder(d = magnet_d, h = magnet_depth + 0.1);
}

module label_cut() {
    depth = 0.4;
    translate([0, h * 0.55, thickness - depth])
        linear_extrude(depth + 0.1)
            text(str("Hex9 L", level), size = min(8, s / 10),
                 halign = "center", valign = "center");
}

module half(with_label = false) {
    difference() {
        half_body();
        bevel_cut();
        if (use_magnets) for (xb = boss_x) pocket(xb);
        if (with_label && label_text) label_cut();
    }
}

// ---- engraving -------------------------------------------------------------
function _cum(v, i = 0, a = 0) =
    i >= len(v) ? [a] : concat([a], _cum(v, i + 1, a + v[i]));

// One polygon with holes: rings = [exterior, hole, hole, ...].
module _rings(rings) {
    offs = _cum([for (r = rings) len(r)]);
    polygon([for (r = rings, p = r) p],
            [for (i = [0:len(rings) - 1])
                [for (j = [0:len(rings[i]) - 1]) offs[i] + j]]);
}

// Cut the engraving into this half's outer face. Applied AFTER the body
// mirror (the two halves carry different artwork, so the engraving must not
// be mirrored with the body), clipped to this half's side of the hinge.
module engraved(sgn) {
    do_engrave = engrave && len(engrave_polys) > 0;
    if (do_engrave)
        assert(abs(engrave_side - s) < 0.05,
               "engrave_data.scad was generated for a different level/scale");
    difference() {
        children();
        if (do_engrave)
            intersection() {
                translate([0, 0, -0.05]) mirror([1, 0, 0])
                    linear_extrude(engrave_depth + 0.05)
                        for (p = engrave_polys) _rings(p);
                translate([-s - 5, sgn > 0 ? 0 : -h - 5, -1])
                    cube([2 * s + 10, h + 5, engrave_depth + 2]);
            }
    }
}

// ---- assembly --------------------------------------------------------------
// Folding rotates each half about the web mid-plane (z = web/2) by FOLD.
module maybe_folded(sgn) {
    if (mode == "folded")
        translate([0, 0, web / 2]) rotate([sgn * FOLD, 0, 0])
            translate([0, 0, -web / 2]) children();
    else
        children();
}

maybe_folded(1) engraved(1) half(with_label = true);
maybe_folded(-1) engraved(-1) mirror([0, 1, 0]) half(with_label = false);
