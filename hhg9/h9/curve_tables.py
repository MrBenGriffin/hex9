# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Closed 36-state transducer tables for the H9 Hamiltonian curve.

The hex9 space-filling curve's tree is the LINEAGE tree — iterated
one-generation canonical parents — and the curve address of a cell is
emitted by a row transducer walked down that chain: the T1 row is the
state, the input at each layer is (foster, digit) where ``digit`` is the
cell's own last body nibble and ``foster`` marks a lineage parent whose
address is not the cell's body prefix.

    S      = ROOT_STATE[root_digit]            # 12 L0 hexagons, 6 rows
    index  = AXIOM_POS[root_digit]             # curve order of the roots
    per layer:  r = T1[S, foster, digit]       # base-9 rank digit
                index = index * 9 + r
                S = T2[S, r]

T1 is PARTIAL: -1 entries are (state, input) pairs that cannot occur on
canonical lineage chains, so a -1 lookup is a totality violation (bad or
non-canonical input), not a table gap. T2 is total. The machine closure
(exactly 36 states, empty frontier) was derived and verified in the
hexsphere suite (or-tools-fun/hexsphere/hex_frontier.py, tables
h9curve_rowtables_L5c.pkl): frontier chains terminate at L5, verified via
targeted L6 sampling, and exercised at pure-table depth to L21+ on the
Greenwich seam (hex_deep_sample.py: totality, bijectivity, local
Hamiltonicity).

Encodings below: one char per entry; states in base-36 (0-9a-z), rank
digits 0-8, '.' = unreachable.
"""

import numpy as np

# Curve order of the 12 L0 root hexagons: _AXIOM[i] is the root digit at
# curve position i.
_AXIOM = '0457362ab918'

# Start state per root digit 0..11 (only 6 distinct rows appear at L0).
_ROOT_STATE = '141420535320'

# T1[state] = (non-foster row, foster row), each indexed by digit 0..8.
_T1 = (
    ('285136074', '.........'),
    ('803652714', '.........'),
    ('825174036', '.........'),
    ('714263805', '.........'),
    ('063714852', '.........'),
    ('075236184', '.........'),
    ('085236174', '.........'),
    ('0.371.8.2', '.6...4.5.'),
    ('8.365.7.4', '.0...2.1.'),
    ('8.517.0.6', '.2...4.3.'),
    ('174625083', '.........'),
    ('306147285', '.........'),
    ('741562830', '.........'),
    ('138265047', '.........'),
    ('6.375.8.4', '.0...2.1.'),
    ('471632580', '.........'),
    ('528361407', '.........'),
    ('813652704', '.........'),
    ('147326058', '.........'),
    ('2.513.0.4', '.8...6.7.'),
    ('750623841', '.........'),
    ('0.523.1.4', '.8...6.7.'),
    ('582741603', '.........'),
    ('7.426.8.5', '.1...3.0.'),
    ('7.156.8.0', '.4...2.3.'),
    ('1.462.0.3', '.7...5.8.'),
    ('360527481', '.........'),
    ('417256308', '.........'),
    ('603752814', '.........'),
    ('1.732.0.8', '.4...6.5.'),
    ('5.836.4.7', '.2...1.0.'),
    ('4.725.3.8', '.1...6.0.'),
    ('3.052.4.1', '.6...7.8.'),
    ('4.163.5.0', '.7...2.8.'),
    ('0.523.1.4', '.7...6.8.'),
    ('8.365.7.4', '.1...2.0.'),
)

# T2[state] = next state per emitted rank digit 0..8, base-36.
_T2 = (
    '820376375',
    'h9a19dc75',
    '8a19a19ab',
    'e6gf19a1j',
    'm3763763l',
    'h9ik76h2j',
    'h9ik76375',
    'm37l376nl',
    'h9a89do75',
    '8p19a89ab',
    'e6376rq1j',
    'm3763763l',
    'e6376rq1j',
    'e4519dc75',
    'h9a89as7l',
    'e6gf19a1j',
    '820376375',
    'e4519dc75',
    'e6gf19a1j',
    '89037l375',
    'h9ik76h2j',
    'h9tk7l375',
    '8a19a19ab',
    'e6uf89a1j',
    'el376vq1j',
    'e637lrw1j',
    'h9a19as4l',
    'e6376rq1j',
    'h9a19as4l',
    'e6gx19a8j',
    '820n7637y',
    'e6n76rq8j',
    'z9a19ps4l',
    'elgf19p1j',
    'h9tk7lh2j',
    'e4589do75',
)


def _b36(c: str) -> int:
    return int(c, 36)


# CURVE_AXIOM_POS[root_digit] -> curve position 0..11 of that L0 hexagon.
CURVE_AXIOM_POS = np.empty(12, dtype=np.uint8)
for _i, _c in enumerate(_AXIOM):
    CURVE_AXIOM_POS[int(_c, 16)] = _i

# CURVE_ROOT_STATE[root_digit] -> start state.
CURVE_ROOT_STATE = np.array([_b36(c) for c in _ROOT_STATE], dtype=np.uint8)

# CURVE_T1[state, foster, digit] -> rank digit 0..8, or -1 (unreachable).
CURVE_T1 = np.array(
    [[[-1 if c == '.' else int(c) for c in row] for row in rows]
     for rows in _T1], dtype=np.int8)

# CURVE_T2[state, rank] -> next state (total).
CURVE_T2 = np.array([[_b36(c) for c in row] for row in _T2], dtype=np.uint8)

# CURVE_AXIOM[pos] -> root digit at curve position 0..11 (inverse of AXIOM_POS).
CURVE_AXIOM = np.array([int(c, 16) for c in _AXIOM], dtype=np.uint8)

# Per-state inverse of T1: rank -> (foster, digit). Each state's T1 is a
# bijection — its 9 children take ranks 0..8 exactly once across the
# reachable (foster, digit) inputs — so the inverse is TOTAL, which is what
# makes the curve address symbolically invertible down to the digit/foster
# string (only foster address-spellings need a geometric oracle).
CURVE_T1_INV_FOSTER = np.full((36, 9), 0xFF, dtype=np.uint8)
CURVE_T1_INV_DIGIT = np.full((36, 9), 0xFF, dtype=np.uint8)
for _s in range(36):
    for _f in range(2):
        for _d in range(9):
            _r = CURVE_T1[_s, _f, _d]
            if _r >= 0:
                CURVE_T1_INV_FOSTER[_s, _r] = _f
                CURVE_T1_INV_DIGIT[_s, _r] = _d
if (CURVE_T1_INV_DIGIT == 0xFF).any():   # bijectivity integrity check
    raise AssertionError('curve_tables: T1 rows are not rank-bijective')
