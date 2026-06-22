# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""Disable the obsolete "ugc" / GridEngine reference-engine tests.

These exercise an old in-test reimplementation of the addressing engine
(ugc_regions / ugc_dec / ugc_lut) that has been superseded by hhg9.h9.region.
They fail against the current code and are kept only for historical reference.

Two mechanisms, because one whole module breaks at *import* (collection) while
the other has a mix of live and dead tests:

  * collect_ignore   — skip test_octant_seams.py entirely (it imports the
                       removed module hhg9.h9.cells, so collection itself fails).
  * modifyitems skip — skip the named obsolete tests in test_h9_engine.py while
                       leaving its ~50 still-valid tests running.

To re-enable, delete this file (or the relevant entry) once the tests are
modernised or removed.
"""
import pytest

collect_ignore = ["test_octant_seams.py"]

# Obsolete tests in test_h9_engine.py (the ugc/GridEngine reference impl).
_OBSOLETE_UGC_TESTS = {
    "test_boundary_midpoints_do_not_go_invalid_early",
    "test_diag_first_step",
    "test_encode_one_step_child_is_legal",
    "test_encoder_tail_is_mode_locked",
    "test_first_step_math_matches_encoder",
    "test_lateral_symmetry",
    "test_neighbour_decode_encode_roundtrip_reg01",
    "test_neighbour_decode_encode_roundtrip_reg02",
    "test_offsets_self_classify",
    "test_reg_valid_points",
    "test_regression_named_points_depth3",
    "test_roundtrip_conversion",
    "test_shallow_roundtrip",
    "test_terminate_always_normalises_tail_simple",
    "test_trace_roundtrip",
    "test_ugc_regions_invalid_point",
}


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(
        reason="obsolete ugc/GridEngine reference engine (superseded by hhg9.h9.region)")
    for item in items:
        if item.originalname in _OBSOLETE_UGC_TESTS:
            item.add_marker(skip)
