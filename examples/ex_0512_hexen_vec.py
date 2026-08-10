# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Hexen at herd scale: the ex_0511 ecology as a struct-of-arrays sim.

Same field, same constants, same ecology as ex_0511 — but built for
traffic.  Two doctrines replace the object-per-animal design:

STRUCT OF ARRAYS.  A population is not a list of Movers; it is a bank
of numpy columns (stomach, heading, genome, ...) and every verb of the
tick — upkeep, scent, flight, targeting, courtship, feeding — is one
vectorised pass over a whole species.  The field itself is INTERNED:
the walk-layer cells of the k-disk are sorted once into a u128 table,
and from then on a cell is an int32 index — hedges are a bool column,
occupancy is a bincount, "who is in that hex" is a CSR slice, and the
per-hex sight cache of ex_0511 becomes boolean ROWS of an H×H matrix
that whole populations gather from at once.

GRID MOTION.  ex_0511 moved animals in float degrees and asked the
wheel afterwards which hex they had wandered into.  Here an animal's
position IS a hex9 address at MOTION_LAYER (three layers under the
walk grid, 27 steps to the pitch), and a tick's movement is an integer
number of neighbour-steps.  Steering is VECTOR INTENT, GRID ACTUALITY:
each mover carries an ideal point — its position as pure vector motion
would have it — and the lattice walk chases that point, neighbour by
neighbour, carrying the sub-cell remainder across ticks.  The track is
therefore honest to the vector (no lattice-axis locking) while every
realised position is a hex9 cell: a step is a GRID measure, speed is
steps-per-tick, and the octant seam is somebody else's problem (the
neighbour algebra crosses it exactly).
Movement splits COARSE PLAN from FINE TRANSITION (Ben's framing, and
it is the hierarchy earning its keep exactly as the grass and scent
layers do): the walk-layer neighbour table plans — the Reynolds lean
pushes the heading off nearby hedges before any stepping, and when no
closed neighbour lies near the travel bearing the tick's whole
segment is provably hedge-free and the ideal point is simply binned
(a JUMP: one batched encode, at any MOTION_LAYER).  The fine lattice
is only WALKED where the plan is contested — near hedges and the
field edge — where a blocked neighbour is never stepped into and
"best aligned passable" slides the animal along the wall.  What pays
for the residency algebra is cheap ancestry: hex9.cell_ancestor rolls
a motion cell up to its canonical walk-layer owner in microseconds
(the tail nibble is metadata and is masked out of every identity), so
passability is one wheel call and a searchsorted.

Run:  python examples/ex_0512_hexen_vec.py
  ->  output/hexen_vec.mp4 (animation), output/hexen_vec.png (curves)
      python examples/ex_0512_hexen_vec.py --live
  ->  real-time window (runs until closed)
"""
import sys
import time
from itertools import count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hex9
from ex_0510_fox_and_rabbits import (LAYER, gc_angle, gc_bearing, wrap180,
                                     encode_keys, _full_of, _pitch_of,
                                     _xyz, _lonlat)
from ex_0511_hexen import (BELLY_JITTER, BINKY, BITE_RANGE, BUNNY_BITE,
                           Bunny, CARRION, CARRION_BITE, CARRION_ROT, CENTRE,
                           DEATH_WAIT, FIELD_K, FLEE_RANGE, FOX_BITE, Fox,
                           GENES, GRASS, GRASS_LAYER, HEAT_RANGE, LITTER,
                           MATE_COST, MAX_EYE_K, MAX_TURN, MOVE_COST,
                           RUN_RESET, SENSE_COST, SCENT_COST, SPD, START_BUNNY,
                           START_FOX, START_GRASS, STOMACH, WAIT_COST,
                           WANDER_RESET)

hex9.init()
RNG = np.random.default_rng(2012)
_UID = count()

# ── stamina: the chase is a SPRINT, and sprints end ──────────────────
# Ben's read of the 112k-tick arms race (prey dash closed the speed
# gap; long tail-chases stopped paying): give the genome a reserve.
# A gene only means something with two sides — the benefit is holding
# a sprint, the cost is aerobic upkeep charged with the other senses.
# The hypothesis this exists to test: foxes evolve LESS stamina, not
# more, once chases stop paying for themselves.  ex_0511 is left
# untouched: the gene is added to the vec engine's own GENES table
# and flows through crossover/mutation/census like any other allele.
GENES = dict(GENES, stamina=(20.0, 180.0, 5.0),    # reserve, in ticks
             flee=(0.8, 4.0, 0.08),    # vigilance: bolt distance in
             #                           pitches.  Two-sided through
             #                           the existing budget: earlier
             #                           flight is safer but interrupts
             #                           grazing and burns the reserve
             pack=(0.25, 2.5, 0.05),   # sociability: the fox↔wolf
             #                           axis.  ONE gene for both ends
             #                           of the lonely band — high
             #                           pack pulls harder from closer
             #                           (LONELY_AT/pack, pull×pack).
             #                           Costs and benefits are both
             #                           emergent: clusters pair and
             #                           survive dilution, crowds
             #                           strip prey.  Expect it to
             #                           breathe with the wave.
             eff=(0.25, 1.0, 0.012))   # trophic thrift/greed — THE
#                                        GROUP-SELECTION EXPERIMENT.
#                                        Within a deme greed always
#                                        wins (more banked per bite,
#                                        no individual cost); only a
#                                        deme that eats itself bare
#                                        and DIES can select against
#                                        it, and recolonisation
#                                        through the doorways is what
#                                        spreads restraint.  Watch the
#                                        per-deme eff medians.
FOX_STAMINA = 100.0                # enough to finish ONE honest chase
BUNNY_STAMINA = 60.0               # a sprinter's reserve: two bursts
SPRINT_AT = 0.102                  # pitch/tick: flee, binky, chase and
#                                    pounce sprint; court/roam/graze jog
WINDED_SPEED = 0.070               # all a blown animal can manage
PUFF_RECOVER = 0.25                # reserve regained per non-sprint tick
BLOWN_AT = 0.3                     # hysteresis: once blown, an animal
#                                    rests to this fraction of its
#                                    reserve before it will sprint again
#                                    — pursue/rest cycles, not flutter
STAMINA_COST = 0.002               # upkeep per tick per tick-of-reserve
# ── senescence: nothing lived here of old age until now ─────────────
# Ben's catch: a SOLO fox could last near-forever — any efficiency
# that lets a pair breed makes a single fox trivially self-sustaining
# — and a lone immortal blocks the genocide-respawn (the population
# never empties), so the species gets neither kits nor a fresh
# cohort.  Senescence goes through the BUDGET, not a death cliff:
# past old_age, upkeep climbs by 1/senesce per tick of age, the
# economics decay, and the field does the killing — the death is a
# starvation, so it leaves a carcass like any other.
OLD_AGE = {'F': 1800, 'B': 1300}   # onset, ticks (~4 breeding cycles)
SENESCE = {'F': 300, 'B': 300}     # ticks per +1x upkeep past onset
# ── community coherence: two's company, three's a crowd ─────────────
LONELY_AT = 8.0                    # pitches: a fox whose NEAREST
#                                    colleague is beyond this feels a
#                                    pull toward them — the counter-
#                                    force to territory's push, so the
#                                    comfortable spacing is a BAND
#                                    (repelled inside 6, drawn beyond
#                                    8) and total isolation corrects
#                                    itself.  Born of Ben's edge-deme
#                                    case: prey split through two
#                                    exits, the foxes split with it,
#                                    and two lone foxes are two deaths.
LONELY_PULL = 0.6                  # gain of the pull (repulsion is 1)

# ── sex-dimorphic ardour expression ──────────────────────────────────
MALE_ARDOUR = 0.6                  # a male SEEKS at this fraction of
#                                    his ardour threshold — mating is
#                                    cheap for him, so his gate is low
#                                    (but not zero: roaming for romance
#                                    is still paid for in stomach)
MALE_MIN = 0.15                    # and he'll TAKE an opportunity at
#                                    any belly above near-starvation
MOTION_VIS = 0.4                   # a MOTIONLESS fox is only this
#                                    fraction as noticeable as a
#                                    sprinting one (prey vision is
#                                    motion-tuned): rabbits bolt on
#                                    SALIENCE, distance over
#                                    conspicuousness — so a fox that
#                                    waits is a fox that ambushes,
#                                    and the wait/pounce niche costs
#                                    no stamina at all.  The flee
#                                    gene trades against this:
#                                    vigilance can re-evolve upward
#                                    if ambushing starts to pay.
GRASS_HALF = 350.0                 # scarcity: the mouthful is a function
#                                    of the SWARD (Holling II on clump
#                                    biomass — half the full bite at
#                                    this size).  A lush clump feeds at
#                                    the full rate, a nibbled one gives
#                                    a poor return, so thin swards stop
#                                    paying and the warren stops booming
#                                    on next-to-nothing.  The clump is
#                                    never eradicated — the roots
#                                    doctrine stands; only the REWARD
#                                    thins with the grass.
# sprint-era speeds: a time-limited sprint is allowed to be FASTER
# than ex_0511's unlimited chase was — measured at the old 0.125 a
# full reserve could not close from detection range (0.020 closing ×
# 100 ticks = 2.0 pitches vs 3.45 seen) and the fox line starved on
# every seed; at 0.165 a standing chase completes in ~60 ticks and
# the fox count returns to the ex_0511 ballpark.  Duration-for-pace
# is the whole trade of having a reserve at all.
SPD = dict(SPD, f_chase=0.165, f_pounce=0.190)
FOX_EFFICIENCY = 0.45              # ex_0511's 0.345 was tuned for a
#                                    different machine: sprint-era
#                                    chases cost ~10/tick and grass
#                                    scarcity thins the bellies a kill
#                                    banks.  Probed (2 seeds × 3k,
#                                    trefoil): 0.345 → F1/F5 dying;
#                                    0.45 → F12/F8, 17 litters both
#                                    seeds; 0.55 → WORSE (F6/F5, the
#                                    over-predation cycle).  A proper
#                                    sweep is still owed — per THE
#                                    lesson, 2 seeds is indicative.
NG = len(GENES)

# the walking grain of the motion lattice: 3 layers under the walk
# grid puts 27 micro-steps in a pitch, so the slowest gait (wander,
# 0.030 pitch) is under a step per tick and the fastest (pounce ×
# dash) is six — the ideal point carries every fraction across ticks,
# so the dash gene stays a real, selectable quantity
MOTION_LAYER = LAYER + 4
STEPS_PER_PITCH = 3.0 ** (MOTION_LAYER - LAYER)
DEV_CAP = 95.0                     # a step must be roughly forward: past
#                           Al         this deviation the animal stops and
#                                    turns instead (the veer, gridwise)
ARRIVE = 0.55                      # fine-pitches: close enough to the
#                                    ideal point to stop stepping — the
#                                    sub-cell remainder CARRIES to next
#                                    tick, which is what keeps the walk
#                                    honest to the vector
WALL_DEV = 55.0                    # a step this far off-aim is a SLIDE:
#                                    the wall is winning, so the heading
#                                    is dragged toward the step taken —
#                                    a cornered animal rolls around the
#                                    obstacle instead of clawing at it
SIGHT_SAMPLES = 8                  # occlusion samples per pitch (ex_0510
#                                    uses 3, which can hop the pinch
#                                    between two adjacent hedges)

# genome columns, in GENES order
(G_EYE, G_ANG, G_DASH, G_ARD, G_SCENT, G_STAM, G_FLEE, G_PACK,
 G_EFF) = range(9)

# target kinds
T_NONE, T_BUNNY, T_CARRION, T_GRASS = 0, 1, 2, 3


# ── key algebra: identity, ancestry — no wheel calls ─────────────────

U128 = np.dtype([('hi', '>u8'), ('lo', '>u8')])


def u128(keys):
    """(n,16) uint8 keys -> sortable ids.  The tail nibble is MASKED:
    it is per-cell metadata (rid), not address, and two spellings of
    one cell must collide."""
    a = np.ascontiguousarray(keys, dtype=np.uint8).copy()
    a[:, 15] &= 0xF0
    return a.view(U128).ravel()


def ancestor(keys, layer):
    """(n,16) keys (bin or full) -> u128 of the canonical `layer`
    ancestor, via the wheel's cell_ancestor (the deep re-bin of the
    cell's mode-0 interior — NOT address truncation, which parts ways
    with ownership at nested splits; and not parent∘parent, which
    decoheres).  Costs microseconds per thousand keys."""
    return u128(np.asarray(
        hex9.cell_ancestor(np.ascontiguousarray(keys, dtype=np.uint8),
                           layer), dtype=np.uint8))


def encode_full(lon, lat):
    return np.asarray(
        hex9.encode(np.atleast_1d(np.asarray(lon, dtype=np.float64)),
                    np.atleast_1d(np.asarray(lat, dtype=np.float64))),
        dtype=np.uint8).reshape(-1, 16)


def decode_keys(keys):
    lon, lat = hex9.decode(np.ascontiguousarray(keys, dtype=np.uint8))
    return np.asarray(lon), np.asarray(lat)


def sphere_advance(lon, lat, bearing, dist):
    """Destination after `dist` degrees along a great-circle bearing —
    the exact form, so aim points behave at any latitude and across
    the seam."""
    th, d = np.radians(bearing), np.radians(dist)
    p1, l1 = np.radians(lat), np.radians(lon)
    sp2 = np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th)
    p2 = np.arcsin(np.clip(sp2, -1.0, 1.0))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * sp2)
    return np.degrees(l2), np.degrees(p2)


# ── the field, interned ──────────────────────────────────────────────

# the trefoil layout (--patches): three roughly-hex lobes, mostly
# walled, joined by small gaps — Ben's metapopulation experiment.
# Local communities can specialise; a thinned patch can be rescued
# through a gap; and since scent is global BY DESIGN, mate scarcity
# is itself the migration pressure that pushes animals through.
PATCH_DC = 9.0                     # lobe centres, pitches from CENTRE —
#                                    close enough that neighbouring
#                                    lobes OVERLAP by ~2.4 pitches: the
#                                    shared band is what gets walled,
#                                    and the doorway is cut through it
PATCH_R = 9.0                      # lobe radius, pitches
PATCH_S = PATCH_DC * np.sqrt(3.0)  # deme spacing on the super-lattice
#                                    (pairwise centre distance, ~15.6)
PATCH_K = {3: 20, 7: 26, 9: 41, 19: 41}   # universe disk per count: the
#                                    demes sit on a SUPER-HEX lattice
#                                    (centre, one ring, two rings),
#                                    every adjacent pair walled and
#                                    doored identically — "too small,
#                                    too few" answered with the same
#                                    geometry, size after size
RELICT_MIN = {'F': 8, 'B': 20}     # the line is remembered WITH its
#                                    diversity: the relict pool is
#                                    snapshotted as the population
#                                    falls THROUGH this size, not at
#                                    the last survivor — 130 founders
#                                    cloned off one genome let drift
#                                    swamp selection (bunny eff slid
#                                    0.99→0.81 over five bottlenecks)


class Field:
    """The k-disk as int32 hex ids: sorted u128 table + columns.

    Everything static lives here at full width — centroid tables, the
    walk-layer neighbour table, and the H×H pairwise bearing/distance
    matrices that let a whole species read its sight lines in one
    gather.  Occlusion rows (which cells can see which, hedges in the
    way) are computed lazily per source cell, exactly as ex_0511
    cached them per hex — the genome masks are applied by the caller."""

    def __init__(self, obstacle_count=200, layout='scatter', gap=1,
                 demes=3):
        from hhg9.algorithms.pickers import gcd_cap_rnd
        lon0, lat0 = CENTRE
        self.centre_key = encode_keys(*CENTRE)[0]
        self.pitch = _pitch_of(self.centre_key, LAYER)
        if layout == 'trefoil' and demes not in PATCH_K:
            raise ValueError(f'demes must be one of {sorted(PATCH_K)}')
        disk = np.asarray(hex9.k_disk(
            _full_of(self.centre_key), LAYER,
            PATCH_K[demes] if layout == 'trefoil' else FIELD_K),
            dtype=np.uint8).reshape(-1, 16)
        dlon, dlat = decode_keys(disk)

        pc_lon = pc_lat = None
        self.n_demes = 0
        if layout == 'trefoil':
            # the lobes; the void between them is simply NOT FIELD
            # (out-of-field reads as closed everywhere already)
            if demes == 3:
                brg = np.array([30.0, 150.0, 270.0])
                dc = np.full(3, PATCH_DC)
            else:                      # centre lobe + one or two rings,
                rings = 2 if demes == 19 else 1     # from hex axials
                qr = [(q, r) for q in range(-rings, rings + 1)
                      for r in range(-rings, rings + 1)
                      if max(abs(q), abs(r), abs(q + r)) <= rings]
                if demes == 9:         # ... plus WINGS east and west
                    qr += [(2, 0), (-2, 0)]
                x = np.array([q + r / 2.0 for q, r in qr]) * PATCH_S
                y = np.array([r * np.sqrt(3.0) / 2.0
                              for q, r in qr]) * PATCH_S
                brg = np.degrees(np.arctan2(x, y))
                dc = np.hypot(x, y)
            self.n_demes = demes
            pc_lon, pc_lat = sphere_advance(
                np.full(demes, lon0), np.full(demes, lat0),
                brg, dc * self.pitch)
            self.pc_lon, self.pc_lat = pc_lon, pc_lat
            dp = gc_angle(dlon[:, None], dlat[:, None],
                          pc_lon[None, :], pc_lat[None, :]) / self.pitch
            keep = (dp <= PATCH_R).any(axis=1)
            disk = disk[keep]

        u = u128(disk)
        order = np.argsort(u)
        self.keys16 = disk[order]              # raw keys, tails intact
        self.u = u[order]
        self.H = len(self.u)
        lon, lat = decode_keys(self.keys16)
        self.lon, self.lat = lon, lat

        # hedges: the same random-cap scatter as ex_0511's World.
        # NB the cap radius is KILOMETRES: ex_0511's 20 km ≈ 17
        # pitches — it happened to match the original k-18 disk,
        # which is why it worked, and why bigger layouts must scale
        # it (Ben's catch: at 19 demes a fixed cap leaves the whole
        # outer ring rock-free — no cover, no ambush sites, a
        # different ecology in exactly the demes meant to replicate
        # the inner ones)
        if layout == 'trefoil':
            cap_km = np.radians((PATCH_K[demes] + 1) * self.pitch) \
                * 6371.0088
        else:
            cap_km = 25.0
        h = gcd_cap_rnd([lat0, lon0], obstacle_count, cap_km)
        hid = self.lookup(u128(np.frombuffer(
            b''.join(encode_keys(h[:, 1], h[:, 0])),
            dtype=np.uint8).reshape(-1, 16)))
        self.hedge = np.zeros(self.H, dtype=bool)
        self.hedge[hid[hid >= 0]] = True
        cid = self.lookup(u128(np.frombuffer(
            self.centre_key, dtype=np.uint8).reshape(1, 16)))[0]
        if cid >= 0:
            self.hedge[cid] = False

        self.patch = np.full(self.H, -1, dtype=np.int8)
        if layout == 'trefoil':
            dp = gc_angle(self.lon[:, None], self.lat[:, None],
                          pc_lon[None, :], pc_lat[None, :]) / self.pitch
            self.patch = np.argmin(dp, axis=1).astype(np.int8)
            cell_xyz = _xyz(self.lon, self.lat)
            rock = self.hedge.copy()   # the scatter, before any walls
            wallmask = np.zeros(self.H, dtype=bool)
            doors = []
            pairs = [(i, j) for i in range(demes) for j in range(i + 1,
                                                                 demes)
                     if gc_angle(pc_lon[i], pc_lat[i], pc_lon[j],
                                 pc_lat[j]) / self.pitch <= 1.2 * PATCH_S]
            for i, j in pairs:
                # the wall is the BISECTOR BAND — every cell about as
                # close to one centre as the other.  A crossing step
                # changes the sign of (d_i - d_j) and neighbours differ
                # by at most ~2 pitches, so the band cannot be stepped
                # over: airtight, the full length of the interface,
                # all three bands meeting at the walled tri-point.  At
                # 1.35 pitches half-width the wall runs ~3 cells thick,
                # so the doorway through it is a short TUNNEL — too
                # much traffic flowed through a pinhole in a sheet.
                # confined to cells the pair OWNS: a ring-pair's
                # bisector otherwise runs radially INWARD through the
                # centre deme (cells equidistant to two ring lobes sit
                # deep inside deme 0) and carves it into pie slices —
                # every i↔j ownership change is still covered by the
                # (i,j) band, so airtightness is untouched
                band = np.flatnonzero(
                    (np.abs(dp[:, i] - dp[:, j]) <= 1.35)
                    & (np.minimum(dp[:, i], dp[:, j]) <= PATCH_R)
                    & ((self.patch == i) | (self.patch == j)))
                if not len(band):
                    continue
                # the doorway: a channel `gap` cells wide, cut where
                # the great circle through the two lobe centres pierces
                # the wall — mid-interface, away from the tri-point
                n = np.cross(_xyz(pc_lon[i], pc_lat[i]),
                             _xyz(pc_lon[j], pc_lat[j]))
                n /= np.linalg.norm(n)
                off = np.degrees(np.arcsin(np.clip(
                    cell_xyz[band] @ n, -1.0, 1.0))) / self.pitch
                door = band[np.abs(off) <= 0.55 * gap]
                self.hedge[band] = True
                self.hedge[door] = False       # even over a scatter rock
                wallmask[band] = True
                doors.append(door)
            if doors:
                # a scatter rock beside a doorway can seal the passage
                # even though the door cell itself is open (the reason
                # field viability used to depend on the obstacle seed):
                # clear ROCKS — never wall cells — from the approaches
                da = np.concatenate(doors)
                if len(da):
                    near = (gc_angle(self.lon[:, None], self.lat[:, None],
                                     self.lon[da][None, :],
                                     self.lat[da][None, :]) / self.pitch
                            <= 1.6).any(axis=1)
                    self.hedge[near & rock & ~wallmask] = False
        self.open_ids = np.flatnonzero(~self.hedge)

        # pairwise bearing/distance (pitches) — 2×H² f32, built in row
        # chunks so the f64 temporaries stay bounded (at 19 demes H is
        # ~4700 and a naive broadcast would spike most of a gigabyte)
        self.pair_br = np.empty((self.H, self.H), dtype=np.float32)
        self.pair_d = np.empty((self.H, self.H), dtype=np.float32)
        for s in range(0, self.H, 512):
            e = min(s + 512, self.H)
            self.pair_br[s:e] = gc_bearing(lon[s:e, None], lat[s:e, None],
                                           lon[None, :], lat[None, :])
            self.pair_d[s:e] = gc_angle(lon[s:e, None], lat[s:e, None],
                                        lon[None, :], lat[None, :]) \
                / self.pitch

        # occlusion rows, lazy
        self.vis = np.zeros((self.H, self.H), dtype=bool)
        self._have = np.zeros(self.H, dtype=bool)

        # motion-cell cache: everything a micro-step needs — neighbour
        # keys, centroid bearings, passability, centroids — is static
        # field furniture, computed on a cell's first visit and a pure
        # array gather ever after.  The herd walks the same ground
        # over and over; the wheel is only consulted for new ground.
        self._mslot = {}
        cap = 8192
        self._mc_nb = np.zeros((cap, 6, 16), dtype=np.uint8)
        self._mc_br = np.zeros((cap, 6), dtype=np.float32)
        self._mc_pass = np.zeros((cap, 6), dtype=bool)
        self._mc_fid = np.full((cap, 6), -1, dtype=np.int32)
        self._mc_lon = np.zeros((cap, 6), dtype=np.float32)
        self._mc_lat = np.zeros((cap, 6), dtype=np.float32)
        # the slot GRAPH: a cell's neighbours' own cache slots, learnt
        # as edges are walked — steady-state stepping never touches
        # the dict, only these int gathers
        self._mc_nbslot = np.full((cap, 6), -1, dtype=np.int32)
        self._mn = 0

        # walk-layer neighbour table: the COARSE PLAN grid.  Ids,
        # centroids, bearings and openness for all six neighbours of
        # every cell (out-of-field neighbours included, as closed) —
        # the lean, the ahead-hex preference and the jump clearance
        # all read from here, and none of it changes after init.
        nbs, cnt = hex9.neighbors(encode_full(lon, lat), LAYER)
        nbs = np.asarray(nbs, dtype=np.uint8)
        cnt = np.asarray(cnt).astype(int)
        S = nbs.shape[1]
        slot_ok = np.arange(S)[None, :] < cnt[:, None]
        safe = np.where(slot_ok[..., None], nbs,
                        self.keys16[:, None, :])
        nlon, nlat = decode_keys(safe.reshape(-1, 16))
        nlon, nlat = nlon.reshape(self.H, S), nlat.reshape(self.H, S)
        nid = self.lookup(u128(safe.reshape(-1, 16))).reshape(self.H, S)
        nid[~slot_ok] = -1
        self.nb_id = nid
        self.nb_lon = nlon.astype(np.float32)
        self.nb_lat = nlat.astype(np.float32)
        self.nb_br = gc_bearing(lon[:, None], lat[:, None],
                                nlon, nlat).astype(np.float32)
        self.nb_open = slot_ok & (nid >= 0) \
            & ~self.hedge[np.maximum(nid, 0)]

        if layout == 'trefoil':
            # the layout is only an experiment if the demes actually
            # connect: verify every lobe core reaches every other
            # through the doorways, by BFS, before any animal walks it
            core = []
            for p in range(demes):
                cand = np.flatnonzero((self.patch == p) & ~self.hedge)
                core.append(int(cand[np.argmin(          # nearest OPEN
                    gc_angle(self.lon[cand], self.lat[cand],  # cell to
                             pc_lon[p], pc_lat[p]))]))   # the centre —
            #                          a rock can sit on the centre cell
            seen = np.zeros(self.H, dtype=bool)
            edge = np.array([core[0]])
            seen[core[0]] = True
            while len(edge):           # dedup EVERY wave: an undeduped
                nxt = self.nb_id[edge][self.nb_open[edge]]   # frontier
                edge = np.unique(nxt[~seen[nxt]])   # multiplies copies
                seen[edge] = True      # exponentially on dense fields
            if not all(seen[c] for c in core):
                missing = [p for p in range(demes) if not seen[core[p]]]
                raise RuntimeError(
                    f'demes {missing} unreachable '
                    f'({int(seen[self.open_ids].sum())}'
                    f'/{len(self.open_ids)} open cells reached) — '
                    f'widen the gap or shift PATCH_DC/R')
            # ... and EVERY GATE must itself work: global connectivity
            # can route around one dead doorway (a seed-202 run played
            # 48k ticks with a blocked NW gate nobody could emigrate
            # through).  For each walled pair, walk core to core using
            # ONLY that pair's own demes — the route has no way round.
            for i, j in pairs:
                inpair = ((self.patch == i) | (self.patch == j)) \
                    & ~self.hedge
                seen = np.zeros(self.H, dtype=bool)
                edge = np.array([core[i]])
                seen[core[i]] = True
                while len(edge):
                    nxt = self.nb_id[edge][self.nb_open[edge]]
                    edge = np.unique(nxt[inpair[nxt] & ~seen[nxt]])
                    seen[edge] = True
                if not seen[core[j]]:
                    raise RuntimeError(
                        f'gate {i}<->{j} is blocked — widen the gap, '
                        f'reseed the rocks, or shift PATCH_DC/R')

    def lookup(self, u):
        """u128 ids -> hex ids (int32), -1 when outside the field."""
        i = np.searchsorted(self.u, u)
        i = np.minimum(i, self.H - 1).astype(np.int32)
        return np.where(self.u[i] == u, i, np.int32(-1))

    def ensure_rows(self, ids):
        """Compute missing occlusion rows: every cell within the eye
        limit whose centroid sight line clears the hedges.  Sampled at
        SIGHT_SAMPLES per pitch — ex_0510's pitch/3 can hop clean over
        the pinch where two hedges meet at a vertex, which reads as a
        fox seeing (and clawing) through the gap.  The hedge itself
        stays visible: you can see the hedge, just not through it."""
        for i in ids[~self._have[ids]]:
            near = np.flatnonzero(self.pair_d[i] <= MAX_EYE_K + 0.45)
            near = near[near != i]
            d = self.pair_d[i, near]
            ns = np.maximum(2, np.ceil(d * SIGHT_SAMPLES).astype(np.int64))
            a = _xyz(self.lon[i], self.lat[i])
            bxyz = _xyz(self.lon[near], self.lat[near])
            pts, owner = [], []
            for jj in range(len(near)):
                t = ((np.arange(ns[jj]) + 0.5) / ns[jj])[:, None]
                pts.append(a[None, :] * (1 - t) + bxyz[jj][None, :] * t)
                owner.append(np.full(ns[jj], jj))
            slon, slat = _lonlat(np.concatenate(pts))
            owner = np.concatenate(owner)
            hid = self.lookup(ancestor(encode_full(slon, slat), LAYER))
            bad = (hid >= 0) & self.hedge[np.maximum(hid, 0)] \
                & (hid != i) & (hid != near[owner])
            blocked = np.zeros(len(near), dtype=bool)
            np.logical_or.at(blocked, owner, bad)
            self.vis[i, near[~blocked]] = True
            self.vis[i, i] = True
            self._have[i] = True

    def motion_slots(self, mkeys):
        """Cache slots for (n,16) motion cells, filling misses in one
        batched wheel round (reify centroid -> neighbours -> decode ->
        ancestor)."""
        masked = np.ascontiguousarray(mkeys).copy()
        masked[:, 15] &= 0xF0
        kb = [r.tobytes() for r in masked]
        slots = np.fromiter((self._mslot.get(k, -1) for k in kb),
                            dtype=np.int64, count=len(kb))
        miss = np.flatnonzero(slots < 0)
        if len(miss):
            fresh = {}
            for i in miss:
                fresh.setdefault(kb[i], i)
            rows = np.fromiter(fresh.values(), dtype=np.int64)
            mk = np.ascontiguousarray(mkeys[rows])
            clon, clat = decode_keys(mk)
            nbs, cnt = hex9.neighbors(encode_full(clon, clat), MOTION_LAYER)
            nbs = np.asarray(nbs, dtype=np.uint8)
            cnt = np.asarray(cnt).astype(np.int64)
            u, S = nbs.shape[0], nbs.shape[1]
            slot_ok = np.arange(S)[None, :] < cnt[:, None]
            safe = np.where(slot_ok[..., None], nbs, mk[:, None, :])
            nlon, nlat = decode_keys(safe.reshape(-1, 16))
            fid = self.lookup(ancestor(safe.reshape(-1, 16),
                                       LAYER)).reshape(u, S)
            while self._mn + u > len(self._mc_br):
                for f in ('_mc_nb', '_mc_br', '_mc_pass', '_mc_fid',
                          '_mc_lon', '_mc_lat', '_mc_nbslot'):
                    a = getattr(self, f)
                    pad = np.full_like(a, -1) if f == '_mc_nbslot' \
                        else np.zeros_like(a)
                    setattr(self, f, np.concatenate([a, pad]))
            sl = np.arange(self._mn, self._mn + u)
            self._mn += u
            self._mc_nb[sl, :S] = safe
            self._mc_br[sl, :S] = gc_bearing(clon[:, None], clat[:, None],
                                             nlon.reshape(u, S),
                                             nlat.reshape(u, S))
            self._mc_pass[sl, :S] = slot_ok & (fid >= 0) \
                & ~self.hedge[np.maximum(fid, 0)]
            self._mc_fid[sl, :S] = fid
            self._mc_lon[sl, :S] = nlon.reshape(u, S)
            self._mc_lat[sl, :S] = nlat.reshape(u, S)
            for k, s in zip(fresh, sl):
                self._mslot[k] = s
            slots[miss] = [self._mslot[kb[i]] for i in miss]
        return slots

    def ahead(self, hid, heading):
        """The walk-layer neighbour each animal points at."""
        dev = np.abs(wrap180(self.nb_br[hid] - heading[:, None]))
        dev = np.where(self.nb_id[hid] >= 0, dev, np.inf)
        j = np.argmin(dev, axis=1)
        return self.nb_id[hid, j]

    def random_pos(self, m, patch=None):
        """Random open-cell positions with ex_0511's jitter-and-snap.
        The snap test is the OWNERSHIP rule (motion-cell ancestor),
        the same rule the walker lives by.  `patch` confines the draw
        to one deme — a respawn is an IMMIGRATION, not a world-reset."""
        pool = self.open_ids if patch is None \
            else self.open_ids[self.patch[self.open_ids] == patch]
        ids = pool[RNG.integers(0, len(pool), m)]
        pos_lon = self.lon[ids] + (RNG.random(m) - 0.5) * self.pitch * 0.8
        pos_lat = self.lat[ids] + (RNG.random(m) - 0.5) * self.pitch * 0.8
        full = encode_full(pos_lon, pos_lat)
        mkey = np.asarray(hex9.bin(full, MOTION_LAYER),
                          dtype=np.uint8).reshape(m, 16)
        hid = self.lookup(ancestor(mkey, LAYER))
        bad = (hid < 0) | self.hedge[np.maximum(hid, 0)]
        if bad.any():
            pos_lon[bad] = self.lon[ids[bad]]
            pos_lat[bad] = self.lat[ids[bad]]
            full[bad] = encode_full(pos_lon[bad], pos_lat[bad])
        return pos_lon, pos_lat, full


# ── populations as column banks ──────────────────────────────────────

def species_cfg(cls):
    """Read a species off the ex_0511 class — one source of truth —
    plus this engine's own stamina allele."""
    base = dict(cls.base_genome,
                stamina=FOX_STAMINA if cls.kind == 'F' else BUNNY_STAMINA,
                flee=FLEE_RANGE,       # (foxes carry it unread)
                pack=1.0,              # (bunnies carry it unread)
                eff=FOX_EFFICIENCY if cls.kind == 'F' else 1.0)
    return dict(kind=cls.kind, gestation=cls.gestation,
                cooldown=cls.cooldown_ticks, adult=cls.adult_age,
                pup_cost=cls.pup_cost, milk_total=cls.milk_total,
                milk_rate=cls.milk_rate, territorial=cls.territorial,
                t_range=cls.territory_range, t_push=cls.territory_push,
                efficiency=FOX_EFFICIENCY if cls.kind == 'F'
                else cls.efficiency, base_genome=base,
                old_age=OLD_AGE[cls.kind], senesce=SENESCE[cls.kind],
                # dimorphic ardour: foxes only for now — unconditional
                # bucks DOUBLED rabbit fecundity in the smoke test
                # (male condition was a hidden brake on the warren),
                # and the Allee case this exists for is the fox
                dimorphic=cls.kind == 'F')


def mutate_vec(g):
    for c, (lo, hi, sig) in enumerate(GENES.values()):
        g[:, c] = np.clip(g[:, c] + RNG.normal(0.0, sig, len(g)), lo, hi)
    return g


def crossover_vec(mom, dad):
    """One allele per trait from either parent, then mutation."""
    return mutate_vec(np.where(RNG.random(mom.shape) < 0.5, mom, dad))


class Herd:
    """One species as parallel columns.  Rows stay sorted by uid
    (births append, compaction preserves order) so a uid lookup is a
    searchsorted — object identity without objects."""

    F64 = ('lon', 'lat', 'vlon', 'vlat', 'heading', 'theta', 'speed',
           'stomach', 'milk', 'puff', 'eye_cost', 'threat_lon',
           'threat_lat')
    I32 = ('age', 'pregnant', 'cooldown', 'wander', 'running', 'binky',
           'hexid', 'mslot', 'came_slot', 'came_dir')
    I64 = ('uid', 'target_uid', 'suitor_uid', 'threat_uid')
    I8 = ('target_kind',)
    BOOL = ('female', 'mandown', 'blown', 'alive', 'starved')

    def __init__(self, cfg):
        self.cfg = cfg
        for f in self.F64:
            setattr(self, f, np.zeros(0))
        for f in self.I32:
            setattr(self, f, np.zeros(0, dtype=np.int32))
        for f in self.I64:
            setattr(self, f, np.zeros(0, dtype=np.int64))
        for f in self.I8:
            setattr(self, f, np.zeros(0, dtype=np.int8))
        for f in self.BOOL:
            setattr(self, f, np.zeros(0, dtype=bool))
        self.genome = np.zeros((0, NG))
        self.mate_genome = np.zeros((0, NG))
        self.mkey = np.zeros((0, 16), dtype=np.uint8)

    @property
    def n(self):
        return len(self.uid)

    def _append(self, **cols):
        for f in (self.F64 + self.I32 + self.I64 + self.I8 + self.BOOL
                  + ('genome', 'mate_genome', 'mkey')):
            old = getattr(self, f)
            setattr(self, f, np.concatenate([old, cols[f]]))

    def keep(self, mask):
        for f in (self.F64 + self.I32 + self.I64 + self.I8 + self.BOOL
                  + ('genome', 'mate_genome', 'mkey')):
            setattr(self, f, getattr(self, f)[mask])

    def spawn(self, field, lon, lat, full, genome, female=None, age=None,
              baby=False):
        m = len(lon)
        if m == 0:
            return
        genome = np.asarray(genome, dtype=np.float64).reshape(m, NG)
        mkey = np.asarray(hex9.bin(full, MOTION_LAYER),
                          dtype=np.uint8).reshape(m, 16)
        # residency is OWNERSHIP of the motion cell, never the point
        # bin — at a field-edge cell the two can disagree, and it is
        # the motion cell the animal will walk from
        hexid = field.lookup(ancestor(mkey, LAYER))
        stomach = np.full(m, STOMACH['start'])
        milk = np.zeros(m)
        if baby:
            stomach = STOMACH['baby'] * RNG.uniform(*BELLY_JITTER, m)
            milk = np.full(m, self.cfg['milk_total'])
        self._append(
            lon=lon, lat=lat, vlon=np.array(lon, dtype=np.float64),
            vlat=np.array(lat, dtype=np.float64),
            heading=RNG.random(m) * 360.0,
            theta=np.zeros(m), speed=np.zeros(m), stomach=stomach, milk=milk,
            puff=genome[:, G_STAM].copy(),          # born on fresh legs
            eye_cost=(SENSE_COST * genome[:, G_EYE] ** 2
                      * genome[:, G_ANG] / 180.0
                      + SCENT_COST * genome[:, G_SCENT]
                      + STAMINA_COST * genome[:, G_STAM]),
            threat_lon=np.zeros(m), threat_lat=np.zeros(m),
            age=np.zeros(m, np.int32) if age is None
            else np.asarray(age, np.int32),
            pregnant=np.zeros(m, np.int32), cooldown=np.zeros(m, np.int32),
            wander=np.zeros(m, np.int32), running=np.zeros(m, np.int32),
            binky=np.zeros(m, np.int32),
            hexid=hexid, mslot=np.full(m, -1, np.int32),
            came_slot=np.full(m, -1, np.int32),
            came_dir=np.zeros(m, np.int32),
            uid=np.array([next(_UID) for _ in range(m)], dtype=np.int64),
            target_uid=np.full(m, -1, np.int64),
            suitor_uid=np.full(m, -1, np.int64),
            threat_uid=np.full(m, -1, np.int64),
            target_kind=np.zeros(m, np.int8),
            female=(RNG.random(m) < 0.5) if female is None
            else np.asarray(female, bool),
            mandown=np.zeros(m, bool), blown=np.zeros(m, bool),
            alive=np.ones(m, bool),
            starved=np.zeros(m, bool), genome=genome,
            mate_genome=np.zeros((m, NG)),
            mkey=mkey)

    def uidx(self, uids):
        """uid -> row index, -1 when gone.  Rows are uid-sorted."""
        if self.n == 0:
            return np.full(len(uids), -1, dtype=np.int64)
        i = np.minimum(np.searchsorted(self.uid, uids), self.n - 1)
        return np.where((uids >= 0) & (self.uid[i] == uids), i, -1)

    # ── shared verbs, vectorised ─────────────────────────────────────

    def receptive(self):
        """Ardour is ONE gene, expressed by sex (Ben's catch: a dog
        fox was ignoring a vixen in heat because HIS belly was low —
        bad natural history, and lethal at the n=2 Allee endgame).
        Females gate on condition — they carry the real cost, and a
        lean mother already delivers a smaller litter — while a male
        found by opportunity needs only to be not-starving."""
        cond = self.genome[:, G_ARD] - 0.15
        if self.cfg['dimorphic']:
            cond = np.where(self.female, cond, MALE_MIN)
        return ((self.age >= self.cfg['adult']) & (self.cooldown == 0)
                & (self.pregnant == 0) & ~self.mandown
                & (self.stomach >= cond * STOMACH['max']))

    def keen(self, receptive):
        """SEEKING keeps a condition floor for males too (roaming is
        paid for in stomach), but a much lower one: ardour scaled by
        MALE_ARDOUR rather than taken neat."""
        thr = self.genome[:, G_ARD] - np.where(self.suitor_uid >= 0,
                                               0.16, 0.0)
        if self.cfg['dimorphic']:
            thr = np.where(self.female, thr, thr * MALE_ARDOUR)
        return receptive & (self.stomach >= thr * STOMACH['max'])

    def steer_towards(self, idx, lon, lat, away=False):
        want = gc_bearing(self.lon[idx], self.lat[idx], lon, lat) \
            + (180.0 if away else 0.0)
        self.heading[idx] += np.clip(wrap180(want - self.heading[idx]),
                                     -MAX_TURN, MAX_TURN)

    def do_wander(self, idx):
        fresh = idx[self.wander[idx] == 0]
        self.wander[fresh] = WANDER_RESET
        self.theta[fresh] = (RNG.random(len(fresh)) - 0.5) * 72.0
        self.heading[idx] += self.theta[idx] / WANDER_RESET * 8
        self.wander[idx] -= 1

    def upkeep(self, births):
        """Age, gestation and birth, milk, the sensory bill, starvation.
        Returns the number of litters delivered."""
        self.age += 1
        self.cooldown = np.maximum(0, self.cooldown - 1)
        self.pregnant = np.maximum(0, self.pregnant - 1)
        litters = 0
        due = np.flatnonzero((self.pregnant == 0) & (self.mate_genome[:, 0] > 0))
        if len(due):
            delivered = np.zeros(len(due), dtype=bool)
            for _ in range(LITTER):
                can = (self.stomach[due] - self.cfg['pup_cost']
                       >= STOMACH['starve'])
                idx = due[can]
                if not len(idx):
                    break
                self.stomach[idx] -= self.cfg['pup_cost']
                births.append((self.lon[idx].copy(), self.lat[idx].copy(),
                               crossover_vec(self.genome[idx],
                                             self.mate_genome[idx])))
                delivered |= can
            litters = int(delivered.sum())
            self.mate_genome[due] = 0.0
        drip = np.minimum.reduce([np.full(self.n, self.cfg['milk_rate']),
                                  self.milk, STOMACH['max'] - self.stomach])
        drip = np.maximum(drip, 0.0)
        self.stomach += drip
        self.milk -= drip
        frail = 1.0 + np.maximum(
            0.0, (self.age - self.cfg['old_age']) / self.cfg['senesce'])
        self.stomach -= (WAIT_COST + self.eye_cost) * frail
        starved = self.stomach < STOMACH['starve']
        self.starved |= starved & self.alive
        self.alive &= ~starved
        return litters

    def seen_rows(self, field, idx):
        """(len(idx), H) hexes seen by the SUBSET idx: cached occlusion
        row, cut by each animal's own cone half-angle and eye rings.
        Own hex excluded — every caller treats residency separately.
        Only ever computed for animals that are actually looking."""
        hid = self.hexid[idx]
        field.ensure_rows(np.unique(hid))
        # all (rows × H) arithmetic held to f32: at archipelago scale
        # the f64 upcasts were the second-largest RAM sink
        dev = np.abs(wrap180(field.pair_br[hid]
                             - self.heading[idx, None].astype(np.float32)))
        m = (field.vis[hid]
             & (dev <= self.genome[idx, G_ANG, None].astype(np.float32))
             & (field.pair_d[hid]
                <= self.genome[idx, G_EYE, None].astype(np.float32) + 0.45))
        m[np.arange(len(idx)), hid] = False
        return m

    def sees(self, field, idx, hexcols):
        """(len(idx), len(hexcols)) — the same sight test evaluated
        only at chosen hex columns (e.g. the hexes foxes stand in)."""
        hid = self.hexid[idx]
        field.ensure_rows(np.unique(hid))
        r, c = hid[:, None], hexcols[None, :]
        dev = np.abs(wrap180(field.pair_br[r, c] - self.heading[idx, None]))
        return (field.vis[r, c]
                & (dev <= self.genome[idx, G_ANG, None])
                & (field.pair_d[r, c]
                   <= self.genome[idx, G_EYE, None] + 0.45)) | (r == c)


# ── static bags: grass, roots, carrion ───────────────────────────────

class Bag:
    """Columns for things nailed to the ground."""

    def __init__(self, **proto):
        self.proto = proto
        for f, dt in proto.items():
            setattr(self, f, np.zeros(0, dtype=dt))

    def append(self, **cols):
        for f in self.proto:
            setattr(self, f, np.concatenate([getattr(self, f), cols[f]]))

    def keep(self, mask):
        for f in self.proto:
            setattr(self, f, getattr(self, f)[mask])

    @property
    def n(self):
        return len(getattr(self, next(iter(self.proto))))


def make_grass_bag():
    return Bag(uid=np.int64, lon=np.float64, lat=np.float64,
               hexid=np.int32, fine=U128, size=np.float64,
               root=np.int32)


def uids(m):
    return np.array([next(_UID) for _ in range(m)], dtype=np.int64)


# ── the sim ──────────────────────────────────────────────────────────

class HexenVec:
    """The ex_0511 tick as vectorised phases over the column banks."""

    def __init__(self, layout='scatter', gap=1, demes=3):
        # the trefoil keeps a lighter rock scatter: the walls provide
        # the occlusion drama, the rocks just keep the interiors honest
        self.field = Field(200 if layout == 'scatter' else 20 * demes,
                           layout=layout, gap=gap, demes=demes)
        # start populations scale with the OPEN AREA, so density (and
        # every per-pitch tuning) carries across layouts and sizes
        self.scale = len(self.field.open_ids) / 850.0
        self.fox = Herd(species_cfg(Fox))
        self.bunny = Herd(species_cfg(Bunny))
        self.grass = make_grass_bag()
        self.roots = make_grass_bag()
        self.carrion = Bag(uid=np.int64, lon=np.float64, lat=np.float64,
                           hexid=np.int32, size=np.float64)
        for herd, m in ((self.fox, START_FOX), (self.bunny, START_BUNNY)):
            m = max(2, int(round(m * self.scale)))
            lon, lat, full = self.field.random_pos(m)
            base = np.tile(np.array([herd.cfg['base_genome'][g]
                                     for g in GENES]), (m, 1))
            herd.spawn(self.field, lon, lat, full, base,
                       female=np.arange(m) % 2 == 1,
                       age=herd.cfg['adult'] + RNG.integers(0, 60, m))
        self.sow(self.field.random_pos(
            int(round(START_GRASS * self.scale)))[:2], capped=False)
        self.relict = {'F': None, 'B': None}   # the line, remembered
        self._lastn = {'F': 10 ** 9, 'B': 10 ** 9}
        self.waits = {'F': DEATH_WAIT, 'B': DEATH_WAIT, 'S': DEATH_WAIT}
        self.genocides = {'F': 0, 'B': 0, 'S': 0}
        self.litters = {'F': 0, 'B': 0}
        self.history = []
        self.gene_log = []             # (tick, bunny NG, fox NG) medians
        #                                every 25 ticks — the rises and
        #                                falls, and their RATES, are
        #                                the run's real findings
        self._mate_idx = {}            # herd -> per-row chosen mate index
        self._crowd = {}               # herd -> (n,2) rival push vectors
        self._heat = None              # per-fox index of a keen bunny

    # ── grass ────────────────────────────────────────────────────────

    def sow(self, pts, capped=True):
        """Bind a whole season's seed in one encode; apply BOTH caps
        (fine mosaic, coarse carrying capacity) vectorised: sort the
        batch, rank the contenders per cell, admit while the count
        holds — ex_0511's greedy loop as a cumcount."""
        lon, lat = np.asarray(pts[0]), np.asarray(pts[1])
        if len(lon) == 0:
            return
        full = encode_full(lon, lat)
        hid = self.field.lookup(ancestor(full, LAYER))
        fine = ancestor(full, GRASS_LAYER)
        ok = (hid >= 0) & ~self.field.hedge[np.maximum(hid, 0)]
        if capped and ok.any():
            g = self.grass
            fine_taken = np.sort(g.fine)
            i = np.minimum(np.searchsorted(fine_taken, fine),
                           max(len(fine_taken) - 1, 0))
            if len(fine_taken):
                ok &= fine_taken[i] != fine
            coarse_n = np.bincount(g.hexid, minlength=self.field.H)
            perm = RNG.permutation(len(lon))
            rank = np.zeros(len(lon), dtype=np.int64)
            order = perm[np.argsort(fine[perm], kind='stable')]
            grp = np.concatenate([[True], fine[order][1:] != fine[order][:-1]])
            rank[order] = np.arange(len(lon)) - np.maximum.accumulate(
                np.where(grp, np.arange(len(lon)), 0))
            ok &= rank < GRASS['max_fine']
            order = perm[np.argsort(hid[perm], kind='stable')]
            grp = np.concatenate([[True], hid[order][1:] != hid[order][:-1]])
            rank[order] = np.arange(len(lon)) - np.maximum.accumulate(
                np.where(grp, np.arange(len(lon)), 0))
            ok &= coarse_n[np.maximum(hid, 0)] + rank < GRASS['max_clumps']
        m = int(ok.sum())
        self.grass.append(uid=uids(m), lon=lon[ok], lat=lat[ok],
                          hexid=hid[ok], fine=fine[ok],
                          size=np.full(m, GRASS['initial']),
                          root=np.zeros(m, np.int32))

    def grass_tick(self):
        g = self.grass
        g.size += GRASS['growth']
        seeder = np.flatnonzero(g.size >= GRASS['seed_at'])
        if len(seeder):
            g.size[seeder] -= GRASS['seed_cost']
            src = np.repeat(seeder, GRASS['seeds'])
            b = np.radians(RNG.random(len(src)) * 360.0)
            d = self.field.pitch * (0.6 + RNG.random(len(src)) * 0.7)
            self.sow((g.lon[src] + d * np.sin(b), g.lat[src] + d * np.cos(b)))

    def grass_cleanup(self):
        """Grazed crowns to the root bag; expired roots back into
        daylight if their patch is still open."""
        g = self.grass
        grazed = g.size < GRASS['initial']
        if grazed.any():
            m = int(grazed.sum())
            self.roots.append(uid=g.uid[grazed], lon=g.lon[grazed],
                              lat=g.lat[grazed], hexid=g.hexid[grazed],
                              fine=g.fine[grazed],
                              size=np.zeros(m),
                              root=np.full(m, GRASS['root_wait'], np.int32))
            g.keep(~grazed)
        r = self.roots
        if r.n:
            r.root -= 1
            due = r.root <= 0
            if due.any():
                fine_taken = np.sort(g.fine)
                i = np.minimum(np.searchsorted(fine_taken, r.fine),
                               max(len(fine_taken) - 1, 0))
                free = np.ones(r.n, dtype=bool) if not len(fine_taken) \
                    else fine_taken[i] != r.fine
                coarse_n = np.bincount(g.hexid, minlength=self.field.H)
                revive = due & free & (coarse_n[r.hexid] < GRASS['max_clumps'])
                # within-tick contention: first root per fine cell wins
                if revive.any():
                    ridx = np.flatnonzero(revive)
                    _, first = np.unique(r.fine[ridx], return_index=True)
                    ridx = ridx[first]
                    m = len(ridx)
                    g.append(uid=uids(m), lon=r.lon[ridx],
                             lat=r.lat[ridx], hexid=r.hexid[ridx],
                             fine=r.fine[ridx],
                             size=np.full(m, GRASS['initial']),
                             root=np.zeros(m, np.int32))
                r.keep(~due)           # revived or shaded out, both leave

    # ── boards (scent, territory, heat) ──────────────────────────────

    def boards(self):
        self._mate_idx, self._crowd = {}, {}
        for herd in (self.fox, self.bunny):
            choice = np.full(herd.n, -1, dtype=np.int64)
            rec = herd.receptive()
            cand = np.flatnonzero(rec)
            rows = np.flatnonzero(herd.keen(rec))    # keen ⊆ receptive:
            if len(cand) >= 2 and len(rows):         # only the KEEN look
                # nearest = largest COSINE on unit vectors: one f32
                # matrix product in bounded row-chunks, instead of the
                # six boom²-sized f64 haversine temporaries that were
                # eating RAM by the gigabyte at warren peaks (Ben saw
                # 15 GB) — the scent range becomes a cosine threshold
                xyz = _xyz(herd.lon, herd.lat).astype(np.float32)
                cx = xyz[cand]
                thr = np.cos(np.radians(
                    herd.genome[rows, G_SCENT] * self.field.pitch)
                    ).astype(np.float32)
                for s in range(0, len(rows), 1024):
                    rr = rows[s:s + 1024]
                    cosm = xyz[rr] @ cx.T            # (chunk, cand) f32
                    cosm[herd.female[rr, None]
                         == herd.female[None, cand]] = -2.0
                    cosm[cosm < thr[s:s + 1024, None]] = -2.0
                    j = np.argmax(cosm, axis=1)
                    hit = cosm[np.arange(len(rr)), j] > -1.5
                    choice[rr[hit]] = cand[j[hit]]
            self._mate_idx[id(herd)] = choice

            if herd.cfg['territorial'] and herd.n >= 2:
                d = gc_angle(herd.lon[:, None], herd.lat[:, None],
                             herd.lon[None, :], herd.lat[None, :]) \
                    / self.field.pitch
                np.fill_diagonal(d, np.inf)
                rng_ = herd.cfg['t_range']
                w = np.clip((rng_ - d) / rng_, 0.0, 1.0)
                mate = herd.uidx(herd.suitor_uid)
                has = mate >= 0        # sex is opportune: no push from the
                rows = np.flatnonzero(has)     # animal being courted
                w[rows, mate[rows]] = 0.0
                # ... and loneliness pulls: negative weight on the
                # nearest colleague when it sits beyond the band —
                # scaled by the heritable pack disposition
                pk = herd.genome[:, G_PACK]
                near_j = np.argmin(d, axis=1)
                near_d = d[np.arange(herd.n), near_j]
                lone = near_d > LONELY_AT / pk
                w[lone, near_j[lone]] -= LONELY_PULL * pk[lone]
                br = np.radians(gc_bearing(herd.lon[:, None],
                                           herd.lat[:, None],
                                           herd.lon[None, :],
                                           herd.lat[None, :]))
                push = np.stack([(w * np.sin(br)).sum(axis=1),
                                 (w * np.cos(br)).sum(axis=1)], axis=1)
                push[np.abs(w).max(axis=1) <= 0.01] = 0.0
                self._crowd[id(herd)] = push

        self._heat = np.full(self.fox.n, -1, dtype=np.int64)
        # a DOE in season — females only: with male keenness now cheap
        # and common, the old any-sex reading would have foxes
        # tracking amorous bucks across the field as if in heat
        hot = np.flatnonzero(self.bunny.keen(self.bunny.receptive())
                             & self.bunny.female)
        if len(hot) and self.fox.n:
            d = gc_angle(self.fox.lon[:, None], self.fox.lat[:, None],
                         self.bunny.lon[None, hot],
                         self.bunny.lat[None, hot]) / self.field.pitch
            j = np.argmin(d, axis=1)
            near = d[np.arange(self.fox.n), j] \
                <= HEAT_RANGE * self.fox.genome[:, G_SCENT]
            self._heat[near] = hot[j[near]]

    # ── target picking on the interned grid ──────────────────────────

    @staticmethod
    def _csr(hexids, H, sub=None):
        idx = np.arange(len(hexids)) if sub is None else sub
        order = idx[np.argsort(hexids[idx], kind='stable')]
        counts = np.bincount(hexids[idx], minlength=H)
        starts = np.concatenate([[0], np.cumsum(counts)])
        return order, starts.astype(np.int64), counts

    @staticmethod
    def _pick_in_hex(hexes, csr):
        order, starts, counts = csr
        c = counts[hexes]
        off = np.minimum((RNG.random(len(hexes)) * c).astype(np.int64),
                         np.maximum(c - 1, 0))
        return order[starts[hexes] + off]

    def _pick_hex(self, mask, ahead, hexid, own_first_count=None):
        """ex_0511's pick_from as a Gumbel argmax: own hex pre-empts,
        the hex dead ahead beats the rest, ties break uniformly (the
        2012 rule picked a HEX uniformly, then an occupant)."""
        m, H = mask.shape
        tier = np.zeros((m, H), dtype=np.float32)
        rows = np.arange(m)
        ok_ahead = ahead >= 0
        tier[rows[ok_ahead], ahead[ok_ahead]] += 2.0
        score = np.where(mask, tier + RNG.random((m, H), dtype=np.float32),
                         -np.inf)
        j = np.argmax(score, axis=1)
        got = np.isfinite(score[rows, j])
        choice = np.where(got, j, -1)
        if own_first_count is not None:
            own = own_first_count[hexid] > 0
            choice[own] = hexid[own]
        return choice

    # ── brains ───────────────────────────────────────────────────────

    def bunny_brain(self, births):
        b, f, field = self.bunny, self.fox, self.field
        self.litters['B'] += b.upkeep(births)
        act = b.alive & ~b.mandown
        if not act.any():
            b.speed[:] = 0.0
            return
        b.speed[:] = 0.0

        # flee overrides all: the nearest VISIBLE fox inside range —
        # sight is evaluated at the fox columns only, never a full row
        if f.n:
            calm = act & (b.running == 0)
            ci = np.flatnonzero(calm)
            if len(ci):
                D = gc_angle(b.lon[ci, None], b.lat[ci, None],
                             f.lon[None, :], f.lat[None, :])
                D = np.where(b.sees(field, ci, f.hexid), D, np.inf)
                # motion-tuned vision: a still fox reads as far away
                # (f.speed is last tick's — what the rabbit watched)
                conspic = np.clip(MOTION_VIS + (1.0 - MOTION_VIS)
                                  * f.speed / 0.10, MOTION_VIS, 1.0)
                D = D / conspic[None, :]
                j = np.argmin(D, axis=1)
                near = D[np.arange(len(ci)), j] \
                    <= b.genome[ci, G_FLEE] * field.pitch
                hit = ci[near]
                b.running[hit] = RUN_RESET
                b.suitor_uid[hit] = -1           # romance can wait
                b.threat_uid[hit] = f.uid[j[near]]
                b.threat_lon[hit] = f.lon[j[near]]
                b.threat_lat[hit] = f.lat[j[near]]

        # a mad dash of pure joy
        calm = act & (b.running == 0) & (b.binky == 0) & (b.pregnant == 0) \
            & (b.stomach > BINKY['at'] * STOMACH['max'])
        joy = calm & (RNG.random(b.n) < BINKY['chance'])
        b.binky[joy] = BINKY['ticks']

        run = np.flatnonzero(act & (b.running > 0))
        b.running[run] -= 1
        b.speed[run] = SPD['b_flee']
        ti = f.uidx(b.threat_uid[run])           # a live fox is tracked;
        chase = ti >= 0                          # a dead one leaves only
        b.threat_lon[run[chase]] = f.lon[ti[chase]]   # the memory of where
        b.threat_lat[run[chase]] = f.lat[ti[chase]]   # it last stood
        b.steer_towards(run, b.threat_lon[run], b.threat_lat[run], away=True)

        hop = np.flatnonzero(act & (b.running == 0) & (b.binky > 0))
        b.binky[hop] -= 1
        b.speed[hop] = BINKY['speed']
        b.heading[hop] += RNG.uniform(-70, 70, len(hop))
        b.target_kind[hop] = T_NONE            # forgets its lunch entirely

        busy = np.zeros(b.n, dtype=bool)
        busy[run] = True
        busy[hop] = True
        free = act & ~busy
        free &= ~self.seek_mate(b, free)

        # graze: keep the clump, else own hex, else the seen sward
        tgt = self._resolve(self.grass.uid, b.target_uid,
                            b.target_kind == T_GRASS)
        need = free & (tgt < 0)
        ni = np.flatnonzero(need)
        if len(ni):
            gcnt = np.bincount(self.grass.hexid, minlength=field.H)
            csr = self._csr(self.grass.hexid, field.H)
            for s in range(0, len(ni), 2048):    # bounded (rows × H)
                nn = ni[s:s + 2048]              # peaks at boom scale
                mask = b.seen_rows(field, nn) & (gcnt > 0)[None, :]
                hexes = self._pick_hex(mask, field.ahead(b.hexid[nn],
                                                         b.heading[nn]),
                                       b.hexid[nn], own_first_count=gcnt)
                got = hexes >= 0
                pick = self._pick_in_hex(hexes[got], csr)
                tgt[nn[got]] = pick
                b.target_kind[nn[got]] = T_GRASS
                b.target_uid[nn[got]] = self.grass.uid[pick]
                miss = nn[~got]
                b.target_kind[miss] = T_NONE
                b.speed[miss] = SPD['b_walk']
                b.do_wander(miss)

        chase = np.flatnonzero(free & (tgt >= 0))
        if len(chase):
            t = tgt[chase]
            local = gc_angle(b.lon[chase], b.lat[chase], self.grass.lon[t],
                             self.grass.lat[t]) < BITE_RANGE * field.pitch
            eat, walk = chase[local], chase[~local]
            self._graze(eat, tgt[eat])
            tw = tgt[walk]
            b.speed[walk] = np.where(self.grass.hexid[tw] == b.hexid[walk],
                                     SPD['b_graze'], SPD['b_walk'])
            b.steer_towards(walk, self.grass.lon[tw], self.grass.lat[tw])

    def _graze(self, eaters, tgt):
        b, g = self.bunny, self.grass
        b.speed[eaters] = 0.0
        if not len(eaters):
            return
        room = (STOMACH['max'] - b.stomach[eaters])
        # scarcity: less reward for less grass — the herbivore gets
        # the same functional response the fox has always had
        rate = BUNNY_BITE * g.size[tgt] / (g.size[tgt] + GRASS_HALF)
        req = np.minimum(rate, room)
        tot = np.zeros(g.n)
        np.add.at(tot, tgt, req)
        fac = np.minimum(1.0, g.size / np.maximum(tot, 1e-9))
        got = req * fac[tgt]
        b.stomach[eaters] += got * b.genome[eaters, G_EFF]
        np.add.at(g.size, tgt, -got)
        g.size[tgt] = np.maximum(g.size[tgt], 0.0)
        b.target_kind[eaters[got <= 0.0]] = T_NONE   # bare patch: move on

    def fox_brain(self, births):
        f, b, field = self.fox, self.bunny, self.field
        self.litters['F'] += f.upkeep(births)
        f.speed[:] = 0.0
        act = f.alive & ~f.mandown
        free = act & ~self.seek_mate(f, act)

        # resolve a held target; the tiered search only runs when empty
        t_b = self._resolve(b.uid, f.target_uid, f.target_kind == T_BUNNY)
        t_b[t_b >= 0] = np.where(b.alive[t_b[t_b >= 0]], t_b[t_b >= 0], -1)
        t_c = self._resolve(self.carrion.uid, f.target_uid,
                            f.target_kind == T_CARRION)
        need = free & (t_b < 0) & (t_c < 0)
        ni = np.flatnonzero(need)
        if len(ni):
            seen = f.seen_rows(field, ni)
            live = np.flatnonzero(b.alive)
            bcnt = np.bincount(b.hexid[live], minlength=field.H)
            ccnt = np.bincount(self.carrion.hexid, minlength=field.H)
            bcsr = self._csr(b.hexid, field.H, sub=live)
            ccsr = self._csr(self.carrion.hexid, field.H)
            ahead = field.ahead(f.hexid[ni], f.heading[ni])
            pend = np.ones(len(ni), dtype=bool)

            # 1: prey (else a carcass) already underfoot
            take = pend & (bcnt[f.hexid[ni]] > 0)
            t_b[ni[take]] = self._pick_in_hex(f.hexid[ni[take]], bcsr)
            pend &= ~take
            take = pend & (ccnt[f.hexid[ni]] > 0)
            t_c[ni[take]] = self._pick_in_hex(f.hexid[ni[take]], ccsr)
            pend &= ~take
            # 2: a rabbit in the cone
            sub = np.flatnonzero(pend)
            if len(sub):
                hexes = self._pick_hex(seen[sub] & (bcnt > 0)[None, :],
                                       ahead[sub], f.hexid[ni[sub]])
                got = sub[hexes >= 0]
                t_b[ni[got]] = self._pick_in_hex(hexes[hexes >= 0], bcsr)
                pend[got] = False
            # 3: a doe in season on the wind — followed only when the
            # eyes have nothing (an extra channel, not an override)
            sub = np.flatnonzero(pend)
            if len(sub):
                hot = self._heat[ni[sub]]
                ok = hot >= 0
                ok[ok] = b.alive[hot[ok]]
                t_b[ni[sub[ok]]] = hot[ok]
                pend[sub[ok]] = False
            # 4: a carcass in the cone
            sub = np.flatnonzero(pend)
            if len(sub) and self.carrion.n:
                hexes = self._pick_hex(seen[sub] & (ccnt > 0)[None, :],
                                       ahead[sub], f.hexid[ni[sub]])
                got = sub[hexes >= 0]
                t_c[ni[got]] = self._pick_in_hex(hexes[hexes >= 0], ccsr)
                pend[got] = False

            rest = ni[pend]
            f.speed[rest] = SPD['f_wander']
            f.do_wander(rest)
        f.target_kind[free] = T_NONE
        f.target_kind[free & (t_b >= 0)] = T_BUNNY
        f.target_kind[free & (t_c >= 0)] = T_CARRION
        got_b = free & (t_b >= 0)
        f.target_uid[got_b] = b.uid[t_b[got_b]]
        got_c = free & (t_c >= 0)
        f.target_uid[got_c] = self.carrion.uid[t_c[got_c]]

        # chase / pounce / feed — bunnies first, then the carrion rank
        for tgt, bag_lon, bag_lat, bag_hex, rate, kind in (
                (t_b, b.lon, b.lat, b.hexid, FOX_BITE, T_BUNNY),
                (t_c, self.carrion.lon, self.carrion.lat, self.carrion.hexid,
                 CARRION_BITE, T_CARRION)):
            have = np.flatnonzero(free & (tgt >= 0))
            if not len(have):
                continue
            t = tgt[have]
            local = gc_angle(f.lon[have], f.lat[have], bag_lon[t],
                             bag_lat[t]) < BITE_RANGE * field.pitch
            eat, chase = have[local], have[~local]
            self._devour(eat, tgt[eat], rate, kind)
            tc = tgt[chase]
            f.speed[chase] = np.where(bag_hex[tc] == f.hexid[chase],
                                      SPD['f_pounce'], SPD['f_chase'])
            f.steer_towards(chase, bag_lon[tc], bag_lat[tc])

    def _devour(self, eaters, tgt, rate, kind):
        """Fixed-rate bites, banked at trophic efficiency; contention
        for one carcass shares what is actually there."""
        f = self.fox
        f.speed[eaters] = 0.0
        if not len(eaters):
            return
        eff = f.genome[eaters, G_EFF]          # thrift/greed is heritable
        req = np.minimum(rate, (STOMACH['max'] - f.stomach[eaters]) / eff)
        req = np.maximum(req, 0.0)
        if kind == T_BUNNY:
            b = self.bunny
            avail = b.stomach
            tot = np.zeros(b.n)
            np.add.at(tot, tgt, req)
            fac = np.minimum(1.0, avail / np.maximum(tot, 1e-9))
            got = req * fac[tgt]
            np.add.at(b.stomach, tgt, -got)
            b.mandown[tgt] = True              # the 2012 latch: pinned
            gone = b.stomach < STOMACH['starve']
            b.alive &= ~gone                   # eaten, not starved:
        else:                                  # no corpse of a corpse
            c = self.carrion
            tot = np.zeros(c.n)
            np.add.at(tot, tgt, req)
            fac = np.minimum(1.0, c.size / np.maximum(tot, 1e-9))
            got = req * fac[tgt]
            np.add.at(c.size, tgt, -got)
        f.stomach[eaters] += got * eff
        f.target_kind[eaters[got <= 0.0]] = T_NONE

    @staticmethod
    def _resolve(bag_uid, target_uid, mask):
        """target uids -> row indices in a (uid-sorted) bag, -1 if the
        target is gone or the mask is off."""
        out = np.full(len(target_uid), -1, dtype=np.int64)
        q = np.flatnonzero(mask)
        if len(q) and len(bag_uid):
            i = np.minimum(np.searchsorted(bag_uid, target_uid[q]),
                           len(bag_uid) - 1)
            hit = bag_uid[i] == target_uid[q]
            out[q[hit]] = i[hit]
        return out

    def seek_mate(self, herd, act):
        """The rut, vectorised; courtship itself is a handful of pairs
        and stays a plain loop.  Returns the mask of animals whose
        tick was taken by the urge."""
        rec = herd.receptive()
        keen = herd.keen(rec) & act
        if not keen.any():
            return keen
        s = herd.uidx(herd.suitor_uid)
        valid = s >= 0
        valid[valid] = rec[s[valid]]           # gone or unwilling: re-pick
        s = np.where(valid, s, self._mate_idx[id(herd)])
        ki = np.flatnonzero(keen)
        herd.suitor_uid[ki] = np.where(s[ki] >= 0,
                                       herd.uid[np.maximum(s[ki], 0)], -1)
        roam = np.flatnonzero(keen & (s < 0))
        herd.speed[roam] = SPD['roam']         # range for a partner
        herd.do_wander(roam)

        woo = np.flatnonzero(keen & (s >= 0))
        if len(woo):
            t = s[woo]
            local = gc_angle(herd.lon[woo], herd.lat[woo], herd.lon[t],
                             herd.lat[t]) < BITE_RANGE * self.field.pitch
            court, walk = woo[local], woo[~local]
            herd.speed[walk] = SPD['court']
            herd.steer_towards(walk, herd.lon[s[walk]], herd.lat[s[walk]])
            done = set()
            for i, j in zip(court, s[court]):
                if i in done or j in done:
                    continue
                done.update((int(i), int(j)))
                for m in (i, j):
                    herd.stomach[m] -= MATE_COST
                    herd.cooldown[m] = herd.cfg['cooldown']
                    herd.suitor_uid[m] = -1
                doe, buck = (i, j) if herd.female[i] else (j, i)
                herd.pregnant[doe] = herd.cfg['gestation']
                herd.mate_genome[doe] = herd.genome[buck]
                herd.speed[i] = 0.0    # the partner keeps its own tick
        return keen

    # ── movement: integer steps on the motion lattice ────────────────

    def shun(self, herd):
        """Territorial push as a heading nudge; obstacles get the last
        word later, cell by cell."""
        push = self._crowd.get(id(herd))
        if push is None:
            return
        vx = np.sin(np.radians(herd.heading)) - herd.cfg['t_push'] * push[:, 0]
        vy = np.cos(np.radians(herd.heading)) - herd.cfg['t_push'] * push[:, 1]
        want = np.degrees(np.arctan2(vx, vy))
        nudge = (push != 0).any(axis=1) & ((vx != 0) | (vy != 0))
        herd.heading[nudge] += np.clip(
            wrap180(want[nudge] - herd.heading[nudge]),
            -0.5 * MAX_TURN, 0.5 * MAX_TURN)

    def move(self, herd):
        """Vector intent, grid actuality.

        Each mover carries an IDEAL POINT (vlon, vlat) — its position
        as pure vector motion would have it — advanced exactly
        speed x pitch along the heading every tick.  The walk on the
        motion lattice then CHASES the ideal: each round the mover
        takes the wheel's neighbour of its cell best aligned with the
        bearing to the ideal, provided the neighbour's walk-layer
        ANCESTOR (one cheap cell_ancestor call) is open field, and
        stops within ARRIVE of it.  The sub-cell remainder carries to
        the next tick, so the track is honest to the vector (no
        lattice-axis locking, no side-step) while every realised
        position stays a hex9 cell.  Walls SNAP the ideal back to the
        cell — intent may not tunnel through a hedge — and a step
        forced far off-aim drags the heading its way, so a cornered
        animal rolls around an obstacle instead of clawing at it.  No
        passable neighbour roughly ahead: stop, and turn hard towards
        the most open one — the grid-native veer."""
        field = self.field
        fp = field.pitch / STEPS_PER_PITCH          # fine pitch, degrees
        self.shun(herd)
        herd.heading %= 360.0
        # stamina: sprints drain the reserve, and a blown animal is
        # capped to a winded jog until it has rested back past
        # BLOWN_AT of its reserve (hysteresis: pursue/rest cycles, not
        # flutter at the zero line).  Chases now END, one way or the
        # other — nothing scripts the giving up, the legs just run out
        herd.blown |= herd.puff <= 0.0
        herd.blown &= herd.puff < BLOWN_AT * herd.genome[:, G_STAM]
        herd.speed = np.where((herd.speed > SPRINT_AT) & herd.blown,
                              WINDED_SPEED, herd.speed)
        sprint = herd.speed > SPRINT_AT
        herd.puff = np.where(
            sprint, herd.puff - 1.0,
            np.minimum(herd.puff + PUFF_RECOVER, herd.genome[:, G_STAM]))
        moving = herd.alive & ~herd.mandown & (herd.speed > 0.0)
        mi = np.flatnonzero(moving)
        if not len(mi):
            return

        # THE COARSE PLAN, part 1 — ex_0511's Reynolds lean, one
        # vectorised pass: blocked walk-neighbours push on the heading
        # BEFORE any stepping (closeness × forwardness weights, danger
        # -scaled authority, and the brake), so an animal leans away
        # from a hedge instead of discovering it by headbutt
        brake = np.ones(herd.n)
        blk = ~field.nb_open[herd.hexid[mi]]
        rows = np.flatnonzero(blk.any(axis=1))
        if len(rows):
            ri = mi[rows]
            br6 = field.nb_br[herd.hexid[ri]]
            d6 = gc_angle(herd.lon[ri, None], herd.lat[ri, None],
                          field.nb_lon[herd.hexid[ri]],
                          field.nb_lat[herd.hexid[ri]]) / field.pitch
            dev6 = wrap180(br6 - herd.heading[ri, None])
            w = np.clip((1.25 - d6) / 0.65, 0.0, 1.0) * blk[rows]
            w *= np.clip(0.55 * np.cos(np.radians(dev6)) + 0.55, 0.12, 1.1)
            wmax = w.max(axis=1)
            act = wmax > 0.02
            brad = np.radians(br6)
            vx = np.sin(np.radians(herd.heading[ri])) \
                - 1.4 * (w * np.sin(brad)).sum(axis=1)
            vy = np.cos(np.radians(herd.heading[ri])) \
                - 1.4 * (w * np.cos(brad)).sum(axis=1)
            danger = np.clip(wmax, 0.0, 1.0)
            auth = MAX_TURN * (1.0 + 0.6 * danger)
            want = np.degrees(np.arctan2(vx, vy))
            turn = np.clip(wrap180(want - herd.heading[ri]), -auth, auth)
            turn = np.where((vx == 0.0) & (vy == 0.0), MAX_TURN, turn)
            ai = ri[act]
            herd.heading[ai] = (herd.heading[ai] + turn[act]) % 360.0
            brake[ai] = 1.0 - 0.75 * danger[act]

        x = herd.speed * herd.genome[:, G_DASH] * brake   # pitches/tick
        herd.stomach[mi] -= MOVE_COST * x[mi] / 0.1
        herd.vlon[mi], herd.vlat[mi] = sphere_advance(
            herd.vlon[mi], herd.vlat[mi], herd.heading[mi],
            x[mi] * field.pitch)
        # a mover that keeps getting stopped must not bank unbounded
        # intent: the ideal is kept within a short leash of the cell
        d0 = gc_angle(herd.lon[mi], herd.lat[mi],
                      herd.vlon[mi], herd.vlat[mi])
        leash = (x[mi] * STEPS_PER_PITCH + 3.0) * fp
        far = d0 > leash
        if far.any():
            fi = mi[far]
            herd.vlon[fi], herd.vlat[fi] = sphere_advance(
                herd.lon[fi], herd.lat[fi],
                gc_bearing(herd.lon[fi], herd.lat[fi],
                           herd.vlon[fi], herd.vlat[fi]), leash[far])

        # THE COARSE PLAN, part 2 — jump when clear.  A closed walk-
        # neighbour contests the plan only if it is roughly forward
        # (within 120° of the travel bearing) AND within reach — the
        # segment is short (≤ a third of a pitch), and a hedge whose
        # centroid sits beyond seg + 0.7 pitches cannot be entered
        # this tick (0.7 covers the hex circumradius with margin).
        # Uncontested: the segment is provably hedge-free, so the fine
        # path adds nothing — bin the ideal point directly.  One
        # batched encode for every jumper, at any MOTION_LAYER; the
        # fine lattice is only ever WALKED where the plan is contested.
        seg = np.minimum(d0, leash)
        keep_going = seg > ARRIVE * fp
        want_i = mi[keep_going]
        tb = gc_bearing(herd.lon[want_i], herd.lat[want_i],
                        herd.vlon[want_i], herd.vlat[want_i])
        dev6 = np.abs(wrap180(field.nb_br[herd.hexid[want_i]]
                              - tb[:, None]))
        d6w = gc_angle(herd.lon[want_i, None], herd.lat[want_i, None],
                       field.nb_lon[herd.hexid[want_i]],
                       field.nb_lat[herd.hexid[want_i]]) / field.pitch
        reach = (seg[keep_going] / field.pitch + 0.7)[:, None]
        contested = ((dev6 <= 120.0) & (d6w < reach)
                     & ~field.nb_open[herd.hexid[want_i]]).any(axis=1)
        ji = want_i[~contested]
        walkers = want_i[contested]
        if len(ji):
            full = encode_full(herd.vlon[ji], herd.vlat[ji])
            mkey = np.asarray(hex9.bin(full, MOTION_LAYER),
                              dtype=np.uint8).reshape(-1, 16)
            hid2 = field.lookup(ancestor(mkey, LAYER))
            ok = (hid2 >= 0) & ~field.hedge[np.maximum(hid2, 0)]
            good = ji[ok]
            herd.mkey[good] = mkey[ok]
            glon, glat = decode_keys(mkey[ok])
            herd.lon[good], herd.lat[good] = glon, glat
            herd.hexid[good] = hid2[ok]
            herd.mslot[good] = -1
            herd.came_slot[good] = -1
            walkers = np.concatenate([walkers, ji[~ok]])  # belt/braces

        budget = np.zeros(herd.n, dtype=np.int64)
        budget[walkers] = np.ceil(x[walkers] * STEPS_PER_PITCH
                                  ).astype(np.int64) + 1
        active = walkers
        while len(active):
            m = len(active)
            sl = herd.mslot[active]
            need = np.flatnonzero(sl < 0)
            if len(need):              # new ground: the dict, once —
                got = field.motion_slots(herd.mkey[active[need]])
                sl[need] = got         # and the edge walked to get here
                herd.mslot[active[need]] = got     # is learnt for good
                cs = herd.came_slot[active[need]]
                ok = cs >= 0
                field._mc_nbslot[cs[ok],
                                 herd.came_dir[active[need]][ok]] = got[ok]
            br = field._mc_br[sl]
            to_aim = gc_bearing(herd.lon[active], herd.lat[active],
                                herd.vlon[active], herd.vlat[active])
            dev = np.where(field._mc_pass[sl],
                           np.abs(wrap180(br - to_aim[:, None])), np.inf)
            j = np.argmin(dev, axis=1)
            rows = np.arange(m)
            best = dev[rows, j]
            go = best <= DEV_CAP
            gi, rj = active[go], j[go]
            herd.mkey[gi] = field._mc_nb[sl[go], rj]
            herd.lon[gi] = field._mc_lon[sl[go], rj]
            herd.lat[gi] = field._mc_lat[sl[go], rj]
            herd.hexid[gi] = field._mc_fid[sl[go], rj]
            herd.mslot[gi] = field._mc_nbslot[sl[go], rj]
            herd.came_slot[gi] = sl[go]
            herd.came_dir[gi] = rj
            budget[gi] -= 1
            # sliding along a wall: the step went far off-aim, so the
            # wall wins — heading dragged towards the step taken, and
            # the ideal snapped to the cell (intent shall not tunnel).
            # One slide per tick: the drag needs ticks to act.
            slid = best[go] > WALL_DEV
            si = gi[slid]
            herd.heading[si] += np.clip(
                wrap180(br[rows[go][slid], rj[slid]] - herd.heading[si]),
                -0.75 * MAX_TURN, 0.75 * MAX_TURN)
            herd.vlon[si] = herd.lon[si]
            herd.vlat[si] = herd.lat[si]
            budget[si] = 0
            # the veer: walled ahead — stop, snap intent, and turn
            # towards daylight (unless the ideal is a lateral scrap
            # within a cell or so: then just stop, no drama)
            vi = active[~go]
            budget[vi] = 0
            real = gc_angle(herd.lon[vi], herd.lat[vi],
                            herd.vlon[vi], herd.vlat[vi]) > 1.2 * fp
            vr = vi[real]
            fin = np.isfinite(best[~go][real])
            turn = np.where(fin, np.clip(
                wrap180(br[rows[~go][real], j[~go][real]]
                        - herd.heading[vr]),
                -2 * MAX_TURN, 2 * MAX_TURN), 180.0)
            herd.heading[vr] = (herd.heading[vr] + turn) % 360.0
            herd.vlon[vi] = herd.lon[vi]
            herd.vlat[vi] = herd.lat[vi]
            left = gc_angle(herd.lon[gi], herd.lat[gi],
                            herd.vlon[gi], herd.vlat[gi])
            active = gi[(left > ARRIVE * fp) & (budget[gi] > 0)]

    # ── the tick ─────────────────────────────────────────────────────

    def tick(self):
        f, b, field = self.fox, self.bunny, self.field
        self.boards()
        self.carrion.size -= CARRION_ROT
        self.carrion.keep(self.carrion.size > 1.0)
        self.grass_tick()

        births_b, births_f = [], []
        self.bunny_brain(births_b)
        self.fox_brain(births_f)
        self.move(b)
        self.move(f)

        # the starved leave a carcass; the eaten have already been eaten
        for herd, key in ((b, 'B'), (f, 'F')):
            corpse = ~herd.alive & herd.starved
            m = int(corpse.sum())
            if m:
                self.carrion.append(uid=uids(m), lon=herd.lon[corpse],
                                    lat=herd.lat[corpse],
                                    hexid=herd.hexid[corpse],
                                    size=np.full(m, CARRION[key]))
            n_alive = int(herd.alive.sum())
            if 0 < n_alive < RELICT_MIN[key] <= self._lastn[key]:
                # the relict pool is taken as the line falls THROUGH
                # the viability threshold — while it still carries
                # diversity — never from the last survivor alone
                self.relict[key] = herd.genome[herd.alive].copy()
            if herd.n and n_alive == 0:
                if self.relict[key] is None:   # first-ever collapse
                    self.relict[key] = herd.genome.copy()
                med = np.median(herd.genome, axis=0)
                print(f'tick {len(self.history):6d}: {key} EXTINCT '
                      f'(n was {herd.n}, relict pool '
                      f'{len(self.relict[key])}) — line: '
                      + ' '.join(f'{g}={v:.2f}'
                                 for g, v in zip(GENES, med)))
            if n_alive:
                self._lastn[key] = n_alive
            herd.keep(herd.alive)
        self.grass_cleanup()

        for herd, births in ((b, births_b), (f, births_f)):
            if births:
                lon = np.concatenate([x[0] for x in births])
                lat = np.concatenate([x[1] for x in births])
                gen = np.concatenate([x[2] for x in births])
                herd.spawn(field, lon, lat, encode_full(lon, lat), gen,
                           baby=True)

        # respawn cohorts scale with the world like the founding
        # populations do — 11 foxes split across two demes of a
        # 19-deme archipelago was a rescue pulse tuned for a world
        # a third the size (Ben's n=1 endgame in a full larder:
        # space fixed the prey and DILUTED the predator; the Allee
        # constraint grows with area, so the pulse must too)
        for herd, key, m in ((f, 'F', max(2, int(round(START_FOX
                                                       * self.scale)))),
                             (b, 'B', max(2, int(round(START_BUNNY
                                                       * self.scale)))),
                             (None, 'S', START_GRASS)):
            alive = self.grass.n if herd is None else herd.n
            # a lone fox or doe is FUNCTIONALLY extinct — it can never
            # breed — so the clock starts at n<2 for the sexual
            # species, and the rescue cohort arrives as COMPANY for
            # the survivor rather than after its 2-3k-tick wake
            if alive >= (1 if herd is None else 2):
                continue
            self.waits[key] -= 1
            if self.waits[key] > 0:
                continue
            if herd is None:
                self.sow(field.random_pos(
                    int(round(m * self.scale)))[:2], capped=False)
                print(f'tick {len(self.history):6d}: S respawn — '
                      f'the sward is resown')
            else:
                # LOCAL respawn — but split between the two FARTHEST
                # demes (Ben's catch: a whole cohort into one deme is
                # a predation shock no single warren can absorb — it
                # strips the larder and halves out of the gate).  Two
                # half-cohorts at opposite ends halve the pressure and
                # double the rescue footprint; still an immigration,
                # never a world-reset, carrying RELICT genomes when
                # there are any to carry
                if field.n_demes:
                    a = int(RNG.integers(0, field.n_demes))
                    dd = gc_angle(field.pc_lon[a], field.pc_lat[a],
                                  field.pc_lon, field.pc_lat)
                    z = int(np.argmax(dd))
                    m1 = (m + 1) // 2
                    pa = field.random_pos(m1, patch=a)
                    pz = field.random_pos(m - m1, patch=z)
                    lon = np.concatenate([pa[0], pz[0]])
                    lat = np.concatenate([pa[1], pz[1]])
                    full = np.concatenate([pa[2], pz[2]])
                    where = f'demes {a}+{z}'
                else:
                    lon, lat, full = field.random_pos(m)
                    where = 'the field'
                pool = self.relict[key]
                if pool is not None and len(pool):
                    gen = mutate_vec(
                        pool[RNG.integers(0, len(pool), m)].copy())
                else:
                    gen = np.tile(np.array([herd.cfg['base_genome'][g]
                                            for g in GENES]), (m, 1))
                herd.spawn(field, lon, lat, full, gen,
                           female=np.arange(m) % 2 == 1,
                           age=np.full(m, herd.cfg['adult'], np.int64))
                # immigrants arrive LEAN (the journey is paid for) and
                # OUT OF SYNC (staggered cooldowns) — Ben's catch:
                # full-bellied founders all hit ardour at once and
                # burst a synchronized litter boom into a larder that
                # cannot absorb it.  The founding boom must be earned
                # from the new land, not packed in the wagons.
                herd.stomach[-m:] = STOMACH['max'] \
                    * RNG.uniform(0.45, 0.65, m)
                herd.cooldown[-m:] = RNG.integers(
                    0, herd.cfg['cooldown'] + 1, m).astype(np.int32)
                print(f'tick {len(self.history):6d}: {key} respawn — '
                      f'{m} founders into {where}'
                      + (', relict line' if pool is not None and len(pool)
                         else ', base stock')
                      + (f' (company for {herd.n - m} survivor'
                         + ('s' if herd.n - m > 1 else '') + ')'
                         if herd.n > m else ''))
            self.genocides[key] += 1
            self.waits[key] = DEATH_WAIT

        bm = self.medians(b)
        fm = self.medians(f)
        self.history.append((
            f.n, b.n, float(self.grass.size.sum()) / 1000.0,
            bm[G_EYE], bm[G_DASH], bm[G_ARD], bm[G_SCENT],
            fm[G_EYE], fm[G_DASH], fm[G_ARD], fm[G_SCENT],
            bm[G_STAM], fm[G_STAM]))
        if len(self.history) % 25 == 0:
            self.gene_log.append(np.concatenate(
                [[len(self.history), f.n, b.n], bm, fm]))

    @staticmethod
    def medians(herd):
        if herd.n == 0:
            return np.full(NG, np.nan)
        return np.median(herd.genome, axis=0)


# ── run + render ─────────────────────────────────────────────────────

def main(ticks=3000, frame_every=12, live=False, layout='scatter', gap=1,
         demes=3):
    import matplotlib
    if not live:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    import os
    os.makedirs('output', exist_ok=True)
    sim = HexenVec(layout=layout, gap=gap, demes=demes)
    w = sim.field
    t_start = time.perf_counter()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    pad = w.pitch * 1.2
    for i in range(w.H):
        poly = np.asarray(hex9.cell(w.keys16[i], LAYER))
        fc = '#8f9aa6' if w.hedge[i] else '#fbfbf8'
        ax.add_patch(MplPoly(poly, closed=True, facecolor=fc,
                             edgecolor='#d8d8d8', lw=0.4, zorder=1))
    ax.axhline(0.0, color='#888888', lw=0.7, ls=':', zorder=2)
    ax.set_xlim(w.lon.min() - pad, w.lon.max() + pad)
    ax.set_ylim(w.lat.min() - pad, w.lat.max() + pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    grass_fill = {}
    for i in w.open_ids:
        poly = np.asarray(hex9.cell(w.keys16[i], LAYER))
        patch = MplPoly(poly, closed=True, facecolor='#ffffff',
                        edgecolor='none', zorder=1.5)
        ax.add_patch(patch)
        grass_fill[int(i)] = patch
    # sex by SHAPE not colour (CVD-safe): does filled, bucks open
    b_does, = ax.plot([], [], 'o', mfc='#4a4a4a', mec='#4a4a4a', ms=3.4,
                      zorder=5)
    b_bucks, = ax.plot([], [], 'o', mfc='none', mec='#4a4a4a', ms=3.4,
                       zorder=5)
    expecting, = ax.plot([], [], 'o', mfc='none', mec='#4477aa', ms=8,
                         mew=1.3, zorder=6.5)
    title = ax.set_title('')
    ax.legend(handles=[
        Line2D([], [], marker='o', mfc='none', mec='#4a4a4a', ls='',
               label='bunny ♂ / ♀ (open / filled)'),
        Line2D([], [], marker='^', color='#cc3311', ls='', label='foxes'),
        Line2D([], [], marker='o', mfc='none', mec='#4477aa', ls='',
               label='expecting'),
    ], loc='lower right', fontsize=8)

    quivers = {'F': None, 'B': None}

    def draw_agents():
        for kind, herd, style in (
            ('F', sim.fox, dict(color='#cc3311', scale=1 / (0.80 * w.pitch),
                                width=0.011, headwidth=3.4, headlength=4.4,
                                headaxislength=3.8, zorder=6)),
            ('B', sim.bunny, dict(color='#4a4a4a', scale=1 / (0.42 * w.pitch),
                                  width=0.0042, headwidth=1.6,
                                  headlength=2.2, headaxislength=2.0,
                                  zorder=5.1)),
        ):
            if quivers[kind] is not None:
                quivers[kind].remove()
                quivers[kind] = None
            if herd.n:
                h = np.radians(herd.heading)
                quivers[kind] = ax.quiver(
                    herd.lon, herd.lat, np.sin(h), np.cos(h), pivot='mid',
                    angles='xy', scale_units='xy', minshaft=1, minlength=0,
                    **style)

    frames = None if live else ticks // frame_every

    def step(_frame):
        for _ in range(frame_every):
            sim.tick()
        level = np.zeros(w.H)
        np.add.at(level, sim.grass.hexid, sim.grass.size)
        gold = np.array([0.953, 0.886, 0.663])
        white = np.array([1.0, 1.0, 1.0])
        for i, patch in grass_fill.items():
            a = min(level[i] / (2 * GRASS['seed_at']), 1.0)
            patch.set_facecolor(tuple(white * (1 - a) + gold * a))
        does = sim.bunny.female
        b_does.set_data(sim.bunny.lon[does], sim.bunny.lat[does])
        b_bucks.set_data(sim.bunny.lon[~does], sim.bunny.lat[~does])
        mums_lon = np.concatenate([sim.fox.lon[sim.fox.pregnant > 0],
                                   sim.bunny.lon[sim.bunny.pregnant > 0]])
        mums_lat = np.concatenate([sim.fox.lat[sim.fox.pregnant > 0],
                                   sim.bunny.lat[sim.bunny.pregnant > 0]])
        expecting.set_data(mums_lon, mums_lat)
        draw_agents()
        nf, nb, gs = sim.history[-1][:3]
        title.set_text(f'{len(sim.history):05}; '
                       f'F{nf:03} B{nb:05} G{gs:03.0f}k '
                       f'☠F{sim.genocides["F"]} '
                       f'☠B{sim.genocides["B"]} ☠G{sim.genocides["S"]}')
        return []

    if live:
        anim = FuncAnimation(fig, step, interval=40,  # noqa: F841
                             cache_frame_data=False)
        plt.show()
        print(f'live run ended at tick {len(sim.history)}')
        dump_genes(sim)
        report(sim)
        return

    anim = FuncAnimation(fig, step, frames=frames, blit=False)
    writer = FFMpegWriter(fps=12, codec='h264',
                          extra_args=['-pix_fmt', 'yuv420p'])
    anim.save('output/hexen_vec.mp4', writer=writer, dpi=80)
    dt = time.perf_counter() - t_start
    print(f'wrote output/hexen_vec.mp4 ({frames} frames, {ticks} ticks, '
          f'{ticks / dt:.1f} ticks/s incl. render)')

    h = np.array(sim.history)
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.draw()
    axes[0].imshow(np.asarray(fig.canvas.buffer_rgba()))
    axes[0].axis('off')
    axes[0].set_title('final state')
    t = np.arange(len(h))
    axes[1].plot(t, h[:, 2], color='#c9a227', lw=1.2, label='grass (k-size)')
    axes[1].plot(t, h[:, 1], color='#4a4a4a', lw=1.2, label='bunnies')
    axes[1].plot(t, h[:, 0], color='#cc3311', lw=1.4, label='foxes')
    axes[1].set_xlabel('tick')
    axes[1].set_title(f"populations  (litters "
                      f"{sim.litters['B']}B/{sim.litters['F']}F, genocides "
                      f"{sim.genocides['F']}/{sim.genocides['B']}"
                      f"/{sim.genocides['S']})")
    axes[1].legend(fontsize=9)
    for col, style, lab in ((3, '-', 'bunny eye_k'), (4, '--', 'bunny dash'),
                            (5, ':', 'bunny ardour')):
        axes[2].plot(t, h[:, col], style, color='#4a4a4a', lw=1.2, label=lab)
    for col, style, lab in ((7, '-', 'fox eye_k'), (8, '--', 'fox dash'),
                            (9, ':', 'fox ardour')):
        axes[2].plot(t, h[:, col], style, color='#cc3311', lw=1.2, label=lab)
    axes[2].plot(t, h[:, 11] / 100.0, '-.', color='#4a4a4a', lw=1.2,
                 label='bunny stamina/100')
    axes[2].plot(t, h[:, 12] / 100.0, '-.', color='#cc3311', lw=1.2,
                 label='fox stamina/100')
    axes[2].set_xlabel('tick')
    axes[2].set_title('genome drift (population medians)')
    axes[2].legend(fontsize=8, ncol=2, loc='upper left')
    ax2 = axes[2].twinx()
    ax2.plot(t, h[:, 6], '-', color='#4477aa', lw=1.4, label='bunny scent')
    ax2.plot(t, h[:, 10], '-', color='#994455', lw=1.4, label='fox scent')
    ax2.set_ylabel('scent range (pitches)')
    ax2.legend(fontsize=8, loc='upper right')
    fig2.savefig('output/hexen_vec.png', dpi=140, bbox_inches='tight')
    print('wrote output/hexen_vec.png')
    dump_genes(sim)
    report(sim)


def dump_genes(sim):
    """The genetic record: every gene, both species, full run — as a
    TSV for analysis and a per-gene panel figure where the rises,
    falls and REVERSALS (the arms-race turns) read at a glance."""
    if not sim.gene_log:
        return
    import os
    os.makedirs('output', exist_ok=True)
    import matplotlib.pyplot as plt
    g = np.array(sim.gene_log)
    names = list(GENES)
    with open('output/hexen_genes.tsv', 'w') as fh:
        fh.write('tick\tfoxes\tbunnies\t'
                 + '\t'.join(f'b_{n}' for n in names) + '\t'
                 + '\t'.join(f'f_{n}' for n in names) + '\n')
        for row in g:
            fh.write('\t'.join(f'{v:.4f}' for v in row) + '\n')
    rows = (NG + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.2 * rows),
                             sharex=True)
    axes = axes.ravel()
    t = g[:, 0]
    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(t, g[:, 3 + i], color='#4a4a4a', lw=1.1, label='bunny')
        ax.plot(t, g[:, 3 + NG + i], color='#cc3311', lw=1.1, label='fox')
        ax.set_title(name, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
    for ax in axes[NG:]:
        ax.axis('off')
    axes[max(NG - 3, 0)].set_xlabel('tick')
    fig.tight_layout()
    fig.savefig('output/hexen_genes.png', dpi=120)
    plt.close(fig)
    print('wrote output/hexen_genes.tsv + output/hexen_genes.png')


def report(sim):
    print(f'\n{"":8s}' + ''.join(f'{g:>10s}' for g in GENES) + '   n')
    for herd, lab in ((sim.bunny, 'bunny'), (sim.fox, 'fox')):
        med = HexenVec.medians(herd)
        print(f'{lab:8s}' + ''.join(f'{v:10.2f}' for v in med)
              + f'{herd.n:5d}')
    if (sim.field.patch >= 0).any():
        # the per-deme census: gene-specialisation across the local
        # communities is the whole point of the trefoil, and it is
        # invisible in a global median
        for herd, lab in ((sim.bunny, 'bunny'), (sim.fox, 'fox')):
            where = sim.field.patch[herd.hexid]
            for p in range(sim.field.n_demes):
                sub = herd.genome[where == p]
                if len(sub):
                    med = np.median(sub, axis=0)
                    print(f'{lab:>5s}.{p}  '
                          + ''.join(f'{v:10.2f}' for v in med)
                          + f'{len(sub):5d}')
                else:
                    print(f'{lab:>5s}.{p}  ' + ' ' * 10 * NG + '    0')
    print(f'\ngenocides — foxes {sim.genocides["F"]}, '
          f'bunnies {sim.genocides["B"]}, grass {sim.genocides["S"]}'
          f'   (litters {sim.litters["B"]}B / {sim.litters["F"]}F)')


if __name__ == '__main__':
    LIVE = '--live' in sys.argv
    LAYOUT, DEMES = 'scatter', 3
    for a in sys.argv[1:]:             # --patches (3) or --patches7
        if a.startswith('--patches'):
            LAYOUT = 'trefoil'
            if a[len('--patches'):].isdigit():
                DEMES = int(a[len('--patches'):])
    np.random.seed(1201)                # gcd_cap_rnd (hedges) reads this
    main(ticks=9000, frame_every=3, live=LIVE, layout=LAYOUT, demes=DEMES)
