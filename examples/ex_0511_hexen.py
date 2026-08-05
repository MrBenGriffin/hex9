# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Hexen, revived: the 2012 fox/rabbit/grass simulation on hex9.

A port of Ben's Java Hexen 3.0 (~2012, Swing) onto the symbolic hex9
grid, using the ex_0510 field verbs.  The agent layer keeps the 2012
ecology: steering movers (heading + clamped turn), stomach/starvation
energy budgets, seeding grass clumps, the fox's tiered target search,
the bunny's flee-overrides-graze reflex, the `mandown` latch, and the
genocide-respawn rule.

What 2012 never got (sheer workload — 5/6-day weeks):
  * a world without edges — the Java grid was a TORUS (loopx/loopy);
    here the field sits on the closed octahedron, straddling the
    equator octant seam, and nobody notices;
  * OBSTACLES — hedge hexes are impassable;
  * OCCLUSION — vision is a cone (foxes) or a disk (bunnies) with
    sight lines blocked by hedges: a fox can ambush from behind a
    hedge, and a bunny can graze unseen in its shadow;
  * SEX — reproduction is no longer asexual budding.  An adult on a
    full belly grows a drive, hunts for a mate of its own species
    (the same verbs it hunts food with — a rendezvous problem on the
    hex grid), and a mated female gestates and drops a LITTER OF TWO;
  * a GENOME — and, per Ben, it needs no new body model: the genes
    ARE the existing senses and urge weights.  Each animal inherits
    eye_k (vision rings), eye_angle (field of view), dash (speed
    multiplier) and ardour (the belly fraction at which mating
    outranks eating), one allele per trait from either parent, plus
    mutation.  The cost side is already in the budget: sensing costs
    per tick with eye area, and running costs with speed — so keen
    eyes and fast legs must be paid for in grass or rabbit.  Nothing
    selects the animals; the field does.

2012 -> 2026 dictionary: Hex.petals -> hex9.neighbors; petal(theta) ->
aim; super_petals -> k_ring(2); the own-hex -> ring1 -> ring2 search
-> vision_cone membership by distance; HexGrid -> the wheel.

Run:  python examples/ex_0511_hexen.py
  ->  output/hexen.gif (animation), output/hexen.png (state + curves)
      python examples/ex_0511_hexen.py --live
  ->  real-time window (runs until closed; ~25 fps, 3 ticks/frame)
"""
import sys
from itertools import count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hex9
from ex_0510_fox_and_rabbits import (LAYER, aim, centroids, encode_keys,
                                     encode_keys_multi, gc_angle, gc_bearing,
                                     vision_cone, wrap180, _full_of,
                                     _pitch_of)

hex9.init()
RNG = np.random.default_rng(2012)

# ── world ────────────────────────────────────────────────────────────
CENTRE = (35.0, 0.02)              # (lon, lat): field straddles the seam
FIELD_K = 18                        # field = k-disk of the centre hex

# ── the hierarchy: each process at the grain it deserves ─────────────
# hex9 is not a flat grid, so the sim need not pretend it is.  One
# encode per agent per tick fixes it on the sphere; every layer then
# follows as a re-bin of the same uuid, free.  Grass grows and is
# grazed a layer FINER than the animals walk (the sward keeps texture,
# and a coarse cell that looks bare can still hold green); mates are
# found a layer COARSER (a neighbourhood bucket instead of an
# all-pairs sweep, which is what lets the warren grow).
GRASS_LAYER = LAYER + 1            # 9 fine cells per animal cell
SCENT_LAYER = LAYER - 2            # 81 animal cells per scent bucket

# ── ecology constants (Gang.java / Stayer.java, lightly retuned) ─────
START_FOX, START_BUNNY, START_GRASS = 11, 130, 520
STOMACH = dict(start=4000.0, baby=1000.0, max=5000.0, starve=500.0)
WAIT_COST = 1.0                    # Bodm.waitcost, rescaled to our tick
MOVE_COST = 6.0                    # per unit speed/pitch, per tick
SENSE_COST = 0.05                  # per tick per unit of eye area (genes!)
SCENT_COST = 0.03                  # per tick per pitch of scent range
FOX_BITE = 140.0                   # fox eats bunny (per tick)
CARRION_BITE = 55.0                # a carcass is a SLOW CHEW, not a gulp
#                                    (Ben): dry, tough, and it pins the fox
#                                    in place — a separate dial on the
#                                    scavenging subsidy that leaves live
#                                    predation untouched
BUNNY_BITE = 60.0                  # bunny eats grass (per tick)
GRASS = dict(initial=100.0, growth=6.0, seed_at=2000.0, seed_cost=1100.0,
             seeds=3, max_clumps=1, max_fine=1, root_wait=90)
#   TWO caps, one per grain, which is the hierarchy earning its keep:
#   `max_fine` spreads the sward as a MOSAIC rather than a pile, and
#   `max_clumps` holds the CARRYING CAPACITY at the animal's own grain.
#   Both are needed.  Fine placement alone grew a thick carpet by ~1400
#   ticks (Ben), for a reason worth knowing: a mosaic is HARDER TO
#   GRAZE than a pile — clumps in separate fine cells cost the rabbit a
#   walk between mouthfuls — so intake fell, rabbits thinned, and the
#   sward ran away.  Texture and capacity are different dials.
DEATH_WAIT = 220                   # genocide-respawn mourning (ticks)
CARRION = dict(B=1200.0, F=1800.0)  # body value of a corpse, by species
CARRION_ROT = 10.0                 # decay per tick: a carcass lasts ~2 dozen
#                                    ticks of eating, not 400 — a windfall
#                                    with a deadline, never a standing larder
WANDER_RESET, RUN_RESET = 60, 30   # Mover.wander_reset / running_reset

# ── sex ──────────────────────────────────────────────────────────────
LITTER = 2                         # Ben's rule: two kits per litter
PUP_COST = 1150.0                  # mother's outlay per pup at birth
BELLY_JITTER = (0.75, 1.30)        # newborns vary, so a litter never
#                                    starves in lockstep (Ben's catch)
MATE_COST = 150.0                  # courtship, paid by both
FLEE_RANGE = 1.8                   # pitches: a rabbit bolts only when
#                                    the fox is CLOSE — seeing one three
#                                    fields off and bolting anyway left
#                                    it too harried ever to breed
# r- vs K-strategy lives in the class attributes below (gestation,
# cooldown, adult age): rabbits breed fast and cheap, foxes slowly.
HEAT_RANGE = 0.4                   # a doe in season is smelt at this
#                                    FRACTION of the fox's mate-scent: real
#                                    pressure on fecundity, not omniscience
TERRITORY = 6.0                    # pitches: range over which a territorial
TERRITORY_PUSH = 1.1               # animal shuns its own kind (see
#                                    World.territory_board).  Rabbits get a
#                                    far milder version of the same thing —
#                                    not territory, just elbow room.
BINKY = dict(at=0.78, chance=0.035, ticks=14, speed=0.135)
#   a well-fed rabbit bursts into a mad dash of pure joy.  Ben's idea,
#   and it is DECORATIVE, honestly: an ablation over 3 seeds found no
#   effect on the sward or on boom/bust that could be told from noise.
#   Tuned for visibility rather than for the population curve, because
#   a field of binkying rabbits is its own reward.

# speeds in hex-pitch units per tick (2012 accel gas, rescaled)
SPD = dict(f_wander=0.030, f_chase=0.125, f_pounce=0.160,
           b_graze=0.055, b_walk=0.075, b_flee=0.105,
           court=0.100, roam=0.085)
MAX_TURN = 24.0                    # degrees per tick (steering clamp)
BITE_RANGE = 0.35                  # of a pitch: 'local' in 2012 terms
MAX_EYE_K = 5                      # cache bound; genome may not exceed

# ── the genome: existing senses + urge weights, nothing invented ─────
#      (low, high, mutation sigma) — clipped to [low, high] on birth
GENES = {
    'eye_k':     (1.0, float(MAX_EYE_K), 0.16),   # vision, in rings
    'eye_angle': (35.0, 180.0, 7.0),              # half-angle, degrees
    'dash':      (0.80, 1.30, 0.035),             # speed multiplier
    'ardour':    (0.55, 0.92, 0.030),             # belly frac to mate
    'scent':     (1.0, 20.0, 0.55),               # mate range, pitches
}
# a fox ranges and screams across a territory; a rabbit's world is the
# next few burrows.  Same gene, opposite ends — and free to drift.
FOX_GENOME = dict(eye_k=3.0, eye_angle=55.0, dash=1.0, ardour=0.70,
                  scent=11.0)
BUNNY_GENOME = dict(eye_k=2.0, eye_angle=180.0, dash=1.0, ardour=0.66,
                    scent=3.0)


_UID = count()


def mutate(value, gene):
    lo, hi, sigma = GENES[gene]
    return float(np.clip(value + RNG.normal(0, sigma), lo, hi))


def crossover(a, b):
    """One allele per trait from either parent, then mutation."""
    return {g: mutate(a[g] if RNG.random() < 0.5 else b[g], g)
            for g in GENES}


class World:
    """The field: keys, hedges, residency registry, sight cache."""

    def __init__(self, obstacle_count=100):
        from hhg9.algorithms.pickers import gcd_cap_rnd
        # np.random.seed(42)
        base = Path(__file__).parent
        lon0, lat0 = CENTRE
        self.centre_key = encode_keys(*CENTRE)[0]
        h_lat_lon = gcd_cap_rnd([lat0, lon0], obstacle_count, 20.0)

        self.pitch = _pitch_of(self.centre_key, LAYER)
        disk = np.asarray(hex9.k_disk(_full_of(self.centre_key), LAYER, FIELD_K), dtype=np.uint8)
        self.field = {bytes(r) for r in disk.reshape(-1, 16)}
        h_lat, h_lon = h_lat_lon[:, 0], h_lat_lon[:, 1]
        self.hedges = set(encode_keys(h_lon, h_lat))
        # p = self.pitch
        # t = np.linspace(0, 1, 60)
        # self.hedges = set()
        # for (a_lon, a_lat), (b_lon, b_lat) in [       # two hedges
        #     ((lon0 - 3.4 * p, lat0 + 1.8 * p), (lon0 + 0.4 * p, lat0 + 2.6 * p)),
        #     ((lon0 + 0.8 * p, lat0 - 2.9 * p), (lon0 + 3.4 * p, lat0 - 1.1 * p)),
        # ]:
        #     self.hedges |= set(encode_keys(a_lon + t * (b_lon - a_lon), a_lat + t * (b_lat - a_lat)))
        # self.hedges &= self.field
        self.hedges.discard(self.centre_key)
        self.open = self.field - self.hedges
        lonc, latc = centroids(sorted(self.open))
        self.open_pts = np.stack([lonc, latc], axis=1)
        self.registry = {}
        self.mate_of = {}
        self.crowd_of = {}
        self.prey_of = {}
        self.buckets = {}
        self._ring = {}
        self._sight = {}
        self._cell = {}

    def cell(self, key):
        """(keys, lon, lat, blocked) for a hex AND its neighbours,
        index 0 being the hex itself.  Cached per hex: the lattice and
        the hedges are both static, so this is field furniture, not
        per-animal work.  Feeds obstacle avoidance: bearings and
        distances to the six neighbours are all a steering nudge
        needs.  It does NOT decide containment — decoded centroids are
        not exact cell centres, so a nearest-centroid test disagrees
        with the wheel near cell boundaries and invents obstacles that
        are not there.  Binning stays the wheel's job, batched once
        per tick in Hexen.commit_moves()."""
        hit = self._cell.get(key)
        if hit is None:
            nbs, cnt = hex9.neighbors(_full_of(key)[None, :], LAYER)
            nb = np.asarray(nbs, dtype=np.uint8)[0, :int(np.asarray(cnt)[0])]
            keys = [key] + [bytes(r) for r in nb]
            a = np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(-1, 16)
            lon, lat = hex9.decode(a)
            hit = (keys, np.asarray(lon), np.asarray(lat),
                   np.array([not self.passable(k) for k in keys]))
            self._cell[key] = hit
        return hit

    def sight(self, key):
        """(keys, bearings, distances) unoccluded within MAX_EYE_K.

        Occlusion depends only on the source hex and the (static)
        hedges, never on who is standing there — so it is a property
        of the FIELD and is computed once per hex, lazily.  Each
        animal then applies its own genome (rings, half-angle) as a
        cheap numpy mask."""
        hit = self._sight.get(key)
        if hit is None:
            seen = sorted(vision_cone(key, LAYER, 0.0, 180.0, MAX_EYE_K,
                                      self.hedges) - {key})
            if seen:
                slon, slat = (float(v[0]) for v in centroids([key]))
                lon, lat = centroids(seen)
                hit = (seen, gc_bearing(slon, slat, lon, lat),
                       gc_angle(slon, slat, lon, lat))
            else:
                hit = ([], np.zeros(0), np.zeros(0))
            self._sight[key] = hit
        return hit

    def random_pos(self, m):
        i = RNG.integers(0, len(self.open_pts), m)
        j = (RNG.random((m, 2)) - 0.5) * self.pitch * 0.8
        pos = self.open_pts[i] + j
        # the jitter can cross into a neighbouring hedge; anyone landed
        # inside an obstacle is stuck there for life, so snap them back
        bad = [n for n, k in enumerate(encode_keys(pos[:, 0], pos[:, 1]))
               if not self.passable(k)]
        pos[bad] = self.open_pts[i[bad]]
        return pos

    def rebind(self, movers, statics=()):
        """Re-bin what MOVED; file what did not.

        Grass and carrion are nailed to the ground — their keys were
        fixed when they were sown or died — yet they were being
        re-encoded every tick alongside the animals.  With ~1800
        tussocks to 80 animals that was 96% of the binding work, and
        binding was 43% of the whole tick.  Movers get one batch
        encode (walking layer AND coarse bucket from the same uuid);
        the stationary are simply filed under the key they already
        hold."""
        self.registry = {}
        self.buckets = {}
        if movers:
            got = encode_keys_multi([m.lon for m in movers],
                                    [m.lat for m in movers],
                                    (LAYER, SCENT_LAYER))
            for m, k, ck in zip(movers, got[LAYER], got[SCENT_LAYER]):
                m.key, m.ckey = k, ck
                self.registry.setdefault(k, []).append(m)
                self.buckets.setdefault(ck, []).append(m)
        for st in statics:
            self.registry.setdefault(st.key, []).append(st)


    def scent_board(self, *pops):
        """mate_of[id(animal)] = the nearest keen partner of the other
        sex within that animal's OWN scent range.

        One vectorised all-pairs pass per species per tick.  A COARSE
        -layer bucket index was tried here and MEASURED SLOWER at every
        population from 74 to 1756 agents (~4x), for two reasons worth
        remembering: a bucket must be at least as wide as the nose it
        serves, and a fox's 11-pitch scent already spans a large part
        of the field, so the ring prunes almost nothing; and numpy's
        all-pairs is vectorised where a bucket walk is a Python loop.
        Hierarchical indexing pays when the interaction radius is
        SMALL against the domain — true of grazing, not of fox scent.
        Only the KEEN look; the merely RECEPTIVE are found.  Ranges
        stay asymmetric: a long-nosed fox finds a short-nosed vixen,
        and she need not smell him back."""
        self.mate_of = {}
        for pop in pops:
            cand = [m for m in pop if m.receptive]
            if len(cand) < 2:
                continue
            lon = np.array([m.lon for m in cand])
            lat = np.array([m.lat for m in cand])
            male = np.array([m.sex == 'M' for m in cand])
            d = gc_angle(lon[:, None], lat[:, None],
                         lon[None, :], lat[None, :]) / self.pitch
            d[male[:, None] == male[None, :]] = np.inf   # same sex, self
            for i, m in enumerate(cand):
                if not m.keen:
                    continue                             # not looking
                row = np.where(d[i] > m.genome['scent'], np.inf, d[i])
                j = int(np.argmin(row))
                if np.isfinite(row[j]):
                    self.mate_of[m.uid] = cand[j]

    def territory_board(self, *pops):
        """crowd_of[id(animal)] = the summed push of nearby rivals.

        Ben's formulation: "avoidance of species unless sex is
        opportune" — so the animal we are currently walking towards as
        a mate exerts no push, everyone else does.  Territorial foxes
        that space themselves out stop stripping one locale bare while
        the rabbits boom two fields over; it is the predator-side cure
        for a patchy kill field.  Same pairwise pass as the scent
        board, read with the opposite sign."""
        self.crowd_of = {}
        for pop in pops:
            if len(pop) < 2 or not pop[0].territorial:
                continue
            lon = np.array([m.lon for m in pop])
            lat = np.array([m.lat for m in pop])
            d = gc_angle(lon[:, None], lat[:, None],
                         lon[None, :], lat[None, :]) / self.pitch
            np.fill_diagonal(d, np.inf)
            rng_ = pop[0].territory_range
            wgt = np.clip((rng_ - d) / rng_, 0.0, 1.0)
            for i, m in enumerate(pop):
                mate = self.mate_of.get(m.uid)
                wi = wgt[i]
                if mate is not None:
                    wi = wi.copy()
                    for j, o in enumerate(pop):
                        if o is mate:
                            wi[j] = 0.0          # sex is opportune
                if wi.max() <= 0.01:
                    continue
                br = np.radians(gc_bearing(m.lon, m.lat, lon, lat))
                self.crowd_of[m.uid] = (float((wi * np.sin(br)).sum()),
                                        float((wi * np.cos(br)).sum()))

    def heat_board(self, foxes, prey):
        """prey_of[id(fox)] = the nearest rabbit IN HEAT it can smell.

        Ben's idea, and it earns its keep beyond flavour: `ardour` was
        the one gene with no downside — breeding early was free, so
        the trait could only drift.  A doe in season now carries a
        scent a fox reads at its own range, whatever the hedges and
        whatever the light, so fecundity is paid for in risk and the
        gene finally has two sides."""
        self.prey_of = {}
        hot = [b for b in prey if b.keen]
        if not hot or not foxes:
            return
        lon = np.array([b.lon for b in hot])
        lat = np.array([b.lat for b in hot])
        for f in foxes:
            d = gc_angle(f.lon, f.lat, lon, lat) / self.pitch
            j = int(np.argmin(d))
            if d[j] <= HEAT_RANGE * f.genome['scent']:
                self.prey_of[f.uid] = hot[j]

    def occupants(self, key, kind):
        return [m for m in self.registry.get(key, [])
                if m.kind == kind and not m.dead]

    def passable(self, key):
        return key in self.field and key not in self.hedges


class Grass:
    """Stayer.java: a clump that grows, seeds, and is grazed."""
    kind = 'S'

    def __init__(self, world, lon, lat, key, fine):
        self.lon, self.lat = lon, lat
        self.key, self.fine = key, fine
        self.size = GRASS['initial']
        self.dead = not world.passable(key)
        self.root = 0                  # ticks until a grazed crown resprouts

    @staticmethod
    def sow(world, pts):
        """Bind a whole season's seed in ONE wheel call.

        Binding per seedling cost 43% of the entire tick — a thousand
        single-point encodes a second.  The wheel is a batch instrument
        and every verb here is built that way; sowing had been the one
        place still calling it a blade of grass at a time.  `key` is
        where a rabbit can see it, `fine` is the tussock it occupies,
        and both fall out of the same encode."""
        if len(pts) == 0:
            return []
        lon = np.asarray([p[0] for p in pts], dtype=np.float64)
        lat = np.asarray([p[1] for p in pts], dtype=np.float64)
        got = encode_keys_multi(lon, lat, (LAYER, GRASS_LAYER))
        return [Grass(world, lo, la, k, f) for lo, la, k, f
                in zip(lon, lat, got[LAYER], got[GRASS_LAYER])]

    def takebite(self, wanted):
        bite = min(wanted, self.size)
        self.size -= bite
        if self.size < GRASS['initial']:
            # grazed off, but not gone: the crown survives underground
            self.dead = True
            self.root = GRASS['root_wait']
        return bite

    def resprout(self):
        """A single blade back from the root.

        Grass is a perennial, and grazing takes the leaf, not the
        plant — so a grazed-out sward carries its own recovery in the
        ground rather than waiting for a seed to blow in.  Costs
        nothing to bind: the root keeps the keys the clump was sown
        with, which is the one place a position is guaranteed not to
        have changed."""
        self.size = GRASS['initial']
        self.dead = False
        self.root = 0

    def tick(self, world, clumps_by_key, seedlings):
        self.size += GRASS['growth']
        if self.size >= GRASS['seed_at']:
            self.size -= GRASS['seed_cost']
            for _ in range(GRASS['seeds']):
                b = RNG.random() * 360.0
                d = world.pitch * (0.6 + RNG.random() * 0.7)
                lon = self.lon + d * np.sin(np.radians(b))
                lat = self.lat + d * np.cos(np.radians(b))
                seedlings.append((lon, lat))


class Carrion:
    """A corpse: what starvation leaves behind, for a fox to find.

    Worth BODY mass, not the leftover stomach — an animal that starved
    died with an empty one, but hide and bone still feed a scavenger.
    Rots away over CARRION_ROT ticks, so a carcass is a windfall with
    a deadline rather than a permanent larder."""
    kind = 'C'

    def __init__(self, world, lon, lat, value):
        self.lon, self.lat = float(lon), float(lat)
        self.key = encode_keys(lon, lat)[0]
        self.size = float(value)
        self.dead = False

    def takebite(self, wanted):
        bite = min(wanted, self.size)
        self.size -= bite
        if self.size <= 1.0:
            self.dead = True
        return bite

    def tick(self):
        self.size -= CARRION_ROT
        if self.size <= 1.0:
            self.dead = True


class Mover:
    """Mover.java + sex: heading, speed, stomach, genome, drive."""

    gestation = 80                     # subclasses override all of these
    cooldown_ticks = 60
    adult_age = 100
    pup_cost = PUP_COST                # mother's outlay per pup
    milk_total = 0.0                   # prepaid, dripped after birth
    milk_rate = 10.0
    territorial = False
    territory_range = TERRITORY
    territory_push = TERRITORY_PUSH
    efficiency = 1.0                   # trophic conversion (see Fox)
    base_genome = BUNNY_GENOME

    def __init__(self, world, lon, lat, baby=False, genome=None, sex=None):
        self.lon, self.lat = float(lon), float(lat)
        self.key = None
        self.heading = RNG.random() * 360.0
        self.speed = 0.0
        self.stomach = STOMACH['start']
        self.milk = 0.0
        if baby:
            # a jittered belly plus a prepaid drip of milk: the litter
            # is carried through the window where it can neither hunt
            # nor graze, and siblings no longer expire in unison
            self.stomach = STOMACH['baby'] * RNG.uniform(*BELLY_JITTER)
            self.milk = self.milk_total
        self.genome = dict(genome) if genome else dict(self.base_genome)
        self.sex = sex or ('F' if RNG.random() < 0.5 else 'M')
        self.age = 0
        self.pregnant = 0              # ticks of gestation remaining
        self.mate_genome = None        # the father's, kept at mating
        self.cooldown = 0
        self.target = None             # food
        self.suitor = None             # mate
        self.uid = next(_UID)          # NEVER id(): CPython recycles those
        #                                the moment an animal is freed, so a
        #                                newborn could inherit a dead one's
        #                                mate or territory entry
        self.want = None               # planned step, pending commit
        self.starved = False           # died of hunger => leaves a corpse
        self.binky = 0                 # ticks of joyful nonsense remaining
        self.wander = 0
        self.running = 0
        self.mandown = False           # bitten: pinned (the 2012 latch)
        self.dead = False
        # the price of the sensory genes, charged every tick in
        # upkeep(): eye area in "rings² × field fraction", plus the
        # attention a long nose costs — so a fox's territory-wide
        # scent range is paid for in rabbit
        self.eye_cost = (SENSE_COST * (self.genome['eye_k'] ** 2
                                       * self.genome['eye_angle'] / 180.0)
                         + SCENT_COST * self.genome['scent'])

    def steer_towards(self, lon, lat, away=False):
        want = gc_bearing(self.lon, self.lat, lon, lat) + (180.0 if away else 0)
        self.heading += float(np.clip(wrap180(want - self.heading),
                                      -MAX_TURN, MAX_TURN))

    def do_wander(self):
        if self.wander == 0:
            self.wander = WANDER_RESET
            self.theta = (RNG.random() - 0.5) * 72.0
        self.heading += self.theta / WANDER_RESET * 8
        self.wander -= 1

    def avoid(self, world):
        """Obstacle-avoidance nudge: (turn degrees, speed scale).

        The Reynolds move the 2012 build never got round to.  Instead
        of walking blindly into a hedge and bouncing off at random,
        the animal keeps a feeler arc about its heading and looks TWO
        hexes down it: a blocked cell straight ahead turns hard and
        brakes, one blocked at two hexes' range only leans away.  The
        turn always goes to the OPEN neighbour needing the smallest
        course change, so animals slide along a hedge rather than
        ricochet off it, and the brake means a cornered animal turns
        on the spot instead of grinding into the obstacle.

        All of it reads the cached neighbour table — no encodes, no
        geometry: on a hex lattice the six bearings ARE the feelers."""
        keys, lon, lat, blocked = world.cell(self.key)
        if len(keys) < 2:
            return 0.0, 1.0
        blk = blocked[1:]
        br = gc_bearing(self.lon, self.lat, lon[1:], lat[1:])
        dist = gc_angle(self.lon, self.lat, lon[1:], lat[1:]) / world.pitch
        dev = wrap180(br - self.heading)

        # weight each blocked neighbour by CLOSENESS, then by how far
        # ahead it lies.  Closeness alone is what catches the animal
        # skimming along a hedge: the obstacle is at 90° to its course
        # — invisible to a forward-only feeler — yet one sidestep puts
        # it inside.  The forward term keeps head-on cases dominant.
        w = np.clip((1.25 - dist) / 0.65, 0.0, 1.0) * blk
        w *= np.clip(0.55 * np.cos(np.radians(dev)) + 0.55, 0.12, 1.1)

        if w.max() <= 0.02:                          # two-hex lookahead
            i = int(np.argmin(np.abs(dev)))
            k2, l2, t2, b2 = world.cell(keys[1 + i])
            if len(k2) < 2 or not b2[1:].any():
                return 0.0, 1.0
            d2 = wrap180(gc_bearing(lon[1 + i], lat[1 + i],
                                    l2[1:], t2[1:]) - self.heading)
            if not b2[1:][int(np.argmin(np.abs(d2)))] or not (~blk).any():
                return 0.0, 1.0
            lean = dev[~blk]                         # trouble two hexes
            lean = float(lean[np.argmin(np.abs(lean))])   # off: just lean
            return float(np.clip(0.3 * lean, -MAX_TURN, MAX_TURN)), 0.9

        # steer as VECTORS: present course minus the weighted pull of
        # every obstacle.  Reynolds' way, and it degrades gracefully —
        # several nearby obstacles simply sum into one escape course
        vx = np.sin(np.radians(self.heading)) - 1.4 * float(
            (w * np.sin(np.radians(br))).sum())
        vy = np.cos(np.radians(self.heading)) - 1.4 * float(
            (w * np.cos(np.radians(br))).sum())
        danger = float(np.clip(w.max(), 0.0, 1.0))
        if vx == 0.0 and vy == 0.0:                  # boxed in
            return MAX_TURN, 0.25
        want = np.degrees(np.arctan2(vx, vy))
        # in real danger the nudge OVERRIDES the goal steering (which
        # is still pulling towards food on the far side of the hedge),
        # and the brake buys the turn enough ticks to take effect —
        # slowing is what lets a cornered animal pivot in place
        auth = MAX_TURN * (1.0 + 0.6 * danger)
        return (float(np.clip(wrap180(want - self.heading), -auth, auth)),
                1.0 - 0.75 * danger)

    def shun(self, world):
        """Turn away from rivals — applied BEFORE avoid(), so a fox
        keeping its distance can still never be pushed into a rock:
        obstacles always have the last word on heading."""
        cr = world.crowd_of.get(self.uid)
        if not cr:
            return
        vx = np.sin(np.radians(self.heading)) - self.territory_push * cr[0]
        vy = np.cos(np.radians(self.heading)) - self.territory_push * cr[1]
        if vx == 0.0 and vy == 0.0:
            return
        want = float(np.degrees(np.arctan2(vx, vy)))
        self.heading += float(np.clip(wrap180(want - self.heading),
                                      -0.5 * MAX_TURN, 0.5 * MAX_TURN))

    def move(self, world):
        """PLAN the step; Hexen.commit_moves() binds and applies it.

        Splitting plan from commit lets every animal's destination be
        binned in ONE wheel call per tick rather than one per animal —
        the same batching doctrine as the ex_0510 verbs."""
        self.want = None
        if self.mandown or self.speed == 0.0:
            return
        self.shun(world)
        turn, brake = self.avoid(world)
        self.heading = (self.heading + turn) % 360.0
        d = self.speed * brake * self.genome['dash'] * world.pitch
        self.want = (
            self.lon + d * np.sin(np.radians(self.heading))
            / max(np.cos(np.radians(self.lat)), 0.2),
            self.lat + d * np.cos(np.radians(self.heading)))
        self.stomach -= MOVE_COST * self.speed * brake \
            * self.genome['dash'] / 0.1

    def veer(self, world):
        """Avoidance was overrun: snap towards the open neighbour
        needing the least turn.  A decisive recovery beats the 2012
        random ricochet, and beats a small nudge — which just grinds
        along the obstacle for tick after tick."""
        keys, klon, klat, blocked = world.cell(self.key)
        if len(keys) > 1 and (~blocked[1:]).any():
            turns = wrap180(gc_bearing(self.lon, self.lat, klon[1:],
                                       klat[1:]) - self.heading)[~blocked[1:]]
            self.heading += float(np.clip(
                turns[np.argmin(np.abs(turns))], -2 * MAX_TURN, 2 * MAX_TURN))
        else:
            self.heading += 180.0      # walled in on all six: about-turn
        self.heading %= 360.0

    def local(self, other, world):
        return gc_angle(self.lon, self.lat, other.lon, other.lat) \
            < BITE_RANGE * world.pitch

    def feed(self, target, world, rate):
        """Take at a FIXED rate; bank only `efficiency` of it.

        The earlier form took `rate / efficiency` and banked the full
        rate, so a fox's income was the same at every efficiency and
        the knob only governed how fast prey was shredded — lower
        efficiency made foxes kill FASTER, which is backwards, and it
        welded fox survival to rabbit destruction (hence the knife
        edge Ben mapped at 0.340/0.342).  Taking at a fixed rate also
        gives a real HANDLING TIME — ~36 ticks to eat a rabbit — so a
        fox is occupied while feeding and saturates when prey is
        plentiful: the Holling type II term the system never had."""
        if target is None or target.dead or not self.local(target, world):
            return False
        room = (STOMACH['max'] - self.stomach) / self.efficiency
        bite = target.takebite(min(rate, room))
        self.stomach += bite * self.efficiency
        return bite > 0

    def takebite(self, wanted):
        self.mandown = True
        bite = min(wanted, self.stomach)
        self.stomach -= bite
        if self.stomach < STOMACH['starve']:
            self.dead = True
        return bite

    # ── sex ──────────────────────────────────────────────────────────

    @property
    def receptive(self):
        """Free to mate if a suitor arrives: adult, off cooldown, not
        already carrying, in reasonable condition."""
        return (self.age >= self.adult_age and self.cooldown == 0
                and not self.pregnant and not self.mandown
                and self.stomach >= (self.genome['ardour'] - 0.15)
                * STOMACH['max'])

    @property
    def keen(self):
        """Receptive AND fed past its own ardour threshold, so it goes
        LOOKING.  Only one of a pair need be keen — the other merely
        receptive — because requiring both to peak at the same moment
        made fox litters impossible: a hunting fox sits near its
        threshold, so two were never in rut at once.

        Hysteresis: an animal already walking towards a mate holds its
        keenness at a lower belly, or a long courtship across a
        territory ends in a shrug halfway."""
        thr = self.genome['ardour'] - (0.16 if self.suitor is not None
                                       else 0.0)
        return self.receptive and self.stomach >= thr * STOMACH['max']

    def court(self, other):
        """Meet a keen partner of the other sex: she conceives."""
        for m in (self, other):
            m.stomach -= MATE_COST
            m.cooldown = m.cooldown_ticks
            m.suitor = None
        doe = self if self.sex == 'F' else other
        buck = other if doe is self else self
        doe.pregnant = self.gestation
        doe.mate_genome = dict(buck.genome)

    def give_birth(self, world, births):
        litter = 0
        for _ in range(LITTER):
            if self.stomach - self.pup_cost < STOMACH['starve']:
                break                  # a lean year: fewer kits
            self.stomach -= self.pup_cost
            births.append(type(self)(
                world, self.lon, self.lat, baby=True,
                genome=crossover(self.genome, self.mate_genome)))
            litter += 1
        self.pregnant = 0
        self.mate_genome = None
        return litter

    def upkeep(self, world, births):
        self.age += 1
        self.cooldown = max(0, self.cooldown - 1)
        litter = 0
        if self.pregnant:
            self.pregnant -= 1
            if self.pregnant == 0:
                litter = self.give_birth(world, births)
        if self.milk > 0.0:            # the mother's investment, dripped
            drip = min(self.milk_rate, self.milk,
                       STOMACH['max'] - self.stomach)
            self.stomach += drip
            self.milk -= max(drip, 0.0)
        self.stomach -= WAIT_COST + self.eye_cost
        if self.stomach < STOMACH['starve']:
            self.dead = True
            self.starved = True        # leaves a corpse; an eaten one does not
        return litter

    # ── senses ───────────────────────────────────────────────────────

    def visible_hexes(self, world):
        """The genome's own slice of the field's cached sight lines."""
        keys, br, dist = world.sight(self.key)
        if not keys:
            return []
        m = ((np.abs(wrap180(br - self.heading))
              <= self.genome['eye_angle'])
             & (dist <= (self.genome['eye_k'] + 0.45) * world.pitch))
        return [keys[i] for i in np.flatnonzero(m)]

    def pick_from(self, world, hexes, kind, pred=None):
        """2012 select-from-hexvector: prefer the hex we already head
        towards, else a random occupied one, then a random occupant."""
        pool = {}
        for k in hexes:
            occ = [m for m in world.occupants(k, kind)
                   if pred is None or pred(m)]
            if occ:
                pool[k] = occ
        if not pool:
            return None
        ahead = aim(self.key, LAYER, self.heading)
        keys = list(pool)
        chosen = ahead if ahead in pool else keys[RNG.integers(0, len(keys))]
        occ = pool[chosen]
        return occ[RNG.integers(0, len(occ))]

    def seek_mate(self, world):
        """Returns True if the mating urge took the tick.

        Mate-sensing is by SCENT (see World.scent_board): omni-
        directional, unoccluded — smell drifts round a hedge — and
        ranged by the animal's own `scent` gene rather than its eyes.
        A fox screams across a territory; a rabbit sniffs the next few
        burrows; both pay for the range per tick.  A keen animal
        with nobody in range ROAMS instead of feeding: it is already
        full, so hunting buys it nothing, and roaming is what finds a
        mate.  Roaming burns the belly back below ardour, at which
        point eating resumes — the rut/forage oscillation is emergent,
        not scripted."""
        if not self.keen:
            return False
        mate = self.suitor
        if mate is None or mate.dead or not mate.receptive or mate is self:
            mate = world.mate_of.get(self.uid)
            self.suitor = mate
        if mate is None:
            self.speed = SPD['roam']         # range for a partner
            self.do_wander()
            return True
        if self.local(mate, world):
            self.court(mate)
            self.speed = 0.0
        else:
            self.speed = SPD['court']
            self.steer_towards(mate.lon, mate.lat)
        return True


class Fox(Mover):
    kind = 'F'
    gestation = 105                    # K-strategist: slow, few, dear
    cooldown_ticks = 130
    adult_age = 200
    pup_cost = 1400.0                  # a cub is expensive and long-suckled:
    milk_total = 1500.0                # it cannot catch moving prey for ages
    milk_rate = 12.0
    territorial = True
    # TROPHIC EFFICIENCY: a fox banks only part of what it takes.  At
    # 1.0 a rabbit's whole belly became fox, so a handful of rabbits
    # fed a nation of foxes — the four fox-side improvements stacked
    # into "roaming sharklike foxes" (Ben) with nothing paying for it.
    # Real food chains lose most of the energy at every step.
    # NB: the readings below were taken with the OLD feed(), where the
    # knob governed shredding speed rather than income — they do not
    # transfer.  Retuned against the corrected model; see sweep in the
    # session notes.
    efficiency = 0.55   # retuned against the CORRECTED feed(); now a
    #                     slope, not a cliff.  Sweep (seeds 201 & 4,
    #                     2500 ticks, carrion 55):
    #   eff 0.50  foxM 22.4/22.1  litters 21F/23F   bunM 111/141
    #   eff 0.55  <- chosen: between the two below
    #   eff 0.60  foxM 35.9/27.1  litters 43F/30F   bunM 138/110
    #   eff 0.70  foxM 38.7/40.8  litters 53F/66F   bunM 136/124 (shark-ward)
    # Foxes now BREED at every setting (21-66 litters vs 21 in 7500
    # ticks under the old model) — the infertile decline was the
    # artifact, not the ecology.
    # 0.33 isn't enough to sustain a pop - it's a slow lingering extinction
    # 16k ticks foxes 9, bunnies 473, grass 0   (litters 4990B / 15F)
    #              eye_k eye_angle      dash    ardour     scent   n
    # bunny         1.90    152.02      1.07      0.65      3.17  473
    # fox           2.99     55.00      1.00      0.70     11.00    9
    # 0.340 tick 7542
    #              eye_k eye_angle      dash    ardour     scent   n
    # bunny         1.89    161.80      1.02      0.66      2.52  577
    # fox           2.72     55.00      1.04      0.73     10.87    7
    # genocides — foxes 0, bunnies 0, grass 0   (litters 1948B / 21F)
    # 0.342 at tick 7935 BGen xxx
    #              eye_k eye_angle      dash    ardour     scent   n
    # bunny         2.00    180.00      1.00      0.66      3.00  263
    # fox           2.87     44.98      1.08      0.71     10.85   14
    # genocides — foxes 0, bunnies 2!, grass 0   (litters 1500B / 58F)
    # 0.345 = BGen. xxxx
    # 0.360 = BGen. xxxx

    base_genome = FOX_GENOME

    def tick(self, world, births):
        litter = self.upkeep(world, births)
        if self.dead:
            return litter
        if not self.seek_mate(world):
            seen = self.visible_hexes(world)
            b = self.target
            if b is None or b.dead:                  # tiered search
                own = world.occupants(self.key, 'B') \
                    or world.occupants(self.key, 'C')
                # heat-scent is an EXTRA channel, not an override: a
                # fox takes the rabbit under its nose first, and only
                # follows a distant scent when it can see nothing
                hot = world.prey_of.get(self.uid)     # a doe in season
                b = own[RNG.integers(0, len(own))] if own else \
                    (self.pick_from(world, seen, 'B')
                     or (hot if hot is not None and not hot.dead else None)
                     or self.pick_from(world, seen, 'C'))
                self.target = b
            if b is None:
                self.speed = SPD['f_wander']
                self.do_wander()
            elif self.local(b, world):
                self.speed = 0.0
                rate = FOX_BITE if b.kind == 'B' else CARRION_BITE
                if not self.feed(b, world, rate):
                    self.target = None
            else:
                self.speed = SPD['f_pounce'] if b.key == self.key \
                    else SPD['f_chase']
                self.steer_towards(b.lon, b.lat)
        self.move(world)
        return litter


class Bunny(Mover):
    kind = 'B'
    gestation = 45                     # r-strategist: fast and often
    cooldown_ticks = 35
    adult_age = 90
    # elbow room: mild mutual aversion, wired and ready but OFF — an
    # ablation showed no benefit to the sward and (if anything) deeper
    # grass troughs.  Flip `territorial` to True to try it.
    territorial = False
    territory_range = 2.0
    territory_push = 0.22
    pup_cost = 1050.0                  # a kitten is on grass within days —
    milk_total = 450.0                 # food that does not run away
    milk_rate = 9.0
    base_genome = BUNNY_GENOME

    def tick(self, world, births):
        litter = self.upkeep(world, births)
        if self.dead or self.mandown:
            return litter
        seen = self.visible_hexes(world)
        if self.running == 0:                        # flee overrides all
            foxes = [f for k in [self.key] + list(seen)
                     for f in world.occupants(k, 'F')]
            if foxes:
                d = [gc_angle(self.lon, self.lat, f.lon, f.lat)
                     for f in foxes]
                near = int(np.argmin(d))
                if d[near] <= FLEE_RANGE * world.pitch:
                    self.threat = foxes[near]
                    self.running = RUN_RESET
                    self.suitor = None               # romance can wait
        if self.running == 0 and self.binky == 0 and not self.pregnant \
                and self.stomach > BINKY['at'] * STOMACH['max'] \
                and RNG.random() < BINKY['chance']:
            self.binky = BINKY['ticks']          # a mad dash of pure joy

        if self.running > 0:
            self.running -= 1
            self.speed = SPD['b_flee']
            self.steer_towards(self.threat.lon, self.threat.lat, away=True)
        elif self.binky > 0:
            self.binky -= 1
            self.speed = BINKY['speed']
            self.heading += float(RNG.uniform(-70, 70))
            self.target = None                   # forgets its lunch entirely
        elif not self.seek_mate(world):
            g = self.target
            if g is None or g.dead:
                own = [s for s in world.registry.get(self.key, [])
                       if s.kind == 'S' and not s.dead]
                g = own[RNG.integers(0, len(own))] if own else \
                    self.pick_from(world, seen, 'S')
                self.target = g
            if g is None:
                self.speed = SPD['b_walk']
                self.do_wander()
            elif self.local(g, world):
                self.speed = 0.0
                if not self.feed(g, world, BUNNY_BITE):
                    self.target = None
            else:
                self.speed = SPD['b_graze'] if g.key == self.key \
                    else SPD['b_walk']
                self.steer_towards(g.lon, g.lat)
        self.move(world)
        return litter


class Hexen:
    """Gang.java: the population, its tick, and the respawn rule."""

    def __init__(self):
        self.world = World(200)
        self.foxes = [Fox(self.world, *p)
                      for p in self.world.random_pos(START_FOX)]
        self.bunnies = [Bunny(self.world, *p)
                        for p in self.world.random_pos(START_BUNNY)]
        self.grass = Grass.sow(self.world,
                               self.world.random_pos(START_GRASS))
        self.carrion = []
        self.roots = []
        for i, m in enumerate(self.foxes + self.bunnies):
            m.sex = 'F' if i % 2 else 'M'            # a fair start
            m.age = m.adult_age + int(RNG.integers(0, 60))
        self.waits = {'F': DEATH_WAIT, 'B': DEATH_WAIT, 'S': DEATH_WAIT}
        self.genocides = {'F': 0, 'B': 0, 'S': 0}
        self.litters = {'F': 0, 'B': 0}
        self.history = []

    @staticmethod
    def genome_median(pop):
        """The population's MEDIAN allele per gene — steadier than the
        mean when a boom of one family's kits floods the census."""
        if not pop:
            return {g: np.nan for g in GENES}
        return {g: float(np.median([m.genome[g] for m in pop]))
                for g in GENES}

    def commit_moves(self, movers, w):
        """Bin every planned step in ONE wheel call and apply the legal
        ones; the rest veer.  The wheel is the authority on which hex a
        position falls in — the cached centroids are for steering only."""
        pend = [m for m in movers if m.want is not None and not m.dead]
        if not pend:
            return
        keys = encode_keys([m.want[0] for m in pend],
                           [m.want[1] for m in pend])
        for m, k in zip(pend, keys):
            if w.passable(k):
                m.lon, m.lat = float(m.want[0]), float(m.want[1])
                m.key = k
            else:
                m.veer(w)
            m.want = None

    def tick(self):
        w = self.world
        w.rebind(self.foxes + self.bunnies, self.grass + self.carrion)
        w.scent_board(self.foxes, self.bunnies)
        w.territory_board(self.foxes, self.bunnies)
        w.heat_board(self.foxes, self.bunnies)
        for c in self.carrion:
            c.tick()

        seedlings, clumps, coarse = [], {}, {}
        for s in self.grass:
            clumps.setdefault(s.fine, []).append(s)
            coarse.setdefault(s.key, []).append(s)
        for s in self.grass:
            s.tick(w, clumps, seedlings)
        for s in Grass.sow(w, seedlings):          # one call for the lot
            if s.dead:
                continue
            crowded = (len([c for c in clumps.get(s.fine, [])
                            if not c.dead]) >= GRASS['max_fine']
                       or len([c for c in coarse.get(s.key, [])
                               if not c.dead]) >= GRASS['max_clumps'])
            if not crowded:
                self.grass.append(s)
                clumps.setdefault(s.fine, []).append(s)
                coarse.setdefault(s.key, []).append(s)

        births = []
        for b in self.bunnies:
            self.litters['B'] += 1 if b.tick(w, births) else 0
        for f in self.foxes:
            self.litters['F'] += 1 if f.tick(w, births) else 0
        self.commit_moves(self.bunnies + self.foxes, w)
        self.bunnies += [m for m in births if m.kind == 'B']
        self.foxes += [m for m in births if m.kind == 'F']

        # the starved leave a carcass; the eaten have already been eaten
        self.carrion += [Carrion(w, m.lon, m.lat, CARRION[m.kind])
                         for m in self.bunnies + self.foxes
                         if m.dead and m.starved]
        self.carrion = [c for c in self.carrion if not c.dead]
        self.roots += [g for g in self.grass if g.dead and g.root > 0]
        self.grass = [s for s in self.grass if not s.dead]
        # a crown only comes back into DAYLIGHT — if the neighbours
        # have closed over its patch it stays shaded out, so recovery
        # cannot push the sward past its carrying capacity
        fine_taken, coarse_n = set(), {}
        for g in self.grass:
            fine_taken.add(g.fine)
            coarse_n[g.key] = coarse_n.get(g.key, 0) + 1
        revived, still = [], []
        for r in self.roots:
            r.root -= 1
            if r.root > 0:
                still.append(r)
            elif (r.fine not in fine_taken
                  and coarse_n.get(r.key, 0) < GRASS['max_clumps']):
                r.resprout()
                fine_taken.add(r.fine)
                coarse_n[r.key] = coarse_n.get(r.key, 0) + 1
                revived.append(r)
        self.roots = still                         # the rest are shaded out
        self.grass += revived
        self.bunnies = [b for b in self.bunnies if not b.dead]
        self.foxes = [f for f in self.foxes if not f.dead]

        for pop, k, ctor, n in ((self.foxes, 'F', Fox, START_FOX),
                                (self.bunnies, 'B', Bunny, START_BUNNY),
                                (self.grass, 'S', Grass, START_GRASS)):
            if not pop:
                self.waits[k] -= 1
                if self.waits[k] <= 0:               # genocide-respawn
                    pts = self.world.random_pos(n)
                    new = (Grass.sow(self.world, pts) if k == 'S'
                           else [ctor(self.world, *p) for p in pts])
                    for i, m in enumerate(new):
                        if k != 'S':
                            m.sex = 'F' if i % 2 else 'M'
                            m.age = m.adult_age
                    pop.extend(new)
                    self.genocides[k] += 1
                    self.waits[k] = DEATH_WAIT
        bm = self.genome_median(self.bunnies)
        fm = self.genome_median(self.foxes)
        self.history.append((
            len(self.foxes), len(self.bunnies),
            sum(s.size for s in self.grass) / 1000.0,
            bm['eye_k'], bm['dash'], bm['ardour'], bm['scent'],
            fm['eye_k'], fm['dash'], fm['ardour'], fm['scent']))


# ── run + render ─────────────────────────────────────────────────────

def main(ticks=3000, frame_every=12, live=False):
    import matplotlib
    if not live:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.animation import FuncAnimation, PillowWriter

    sim = Hexen()
    w = sim.world

    fig, ax = plt.subplots(figsize=(8, 8))
    lonc, latc = centroids(sorted(w.field))
    pad = w.pitch * 1.2
    for kk in sorted(w.field):
        poly = np.asarray(hex9.cell(np.frombuffer(kk, dtype=np.uint8), LAYER))
        fc = '#8f9aa6' if kk in w.hedges else '#fbfbf8'
        ax.add_patch(MplPoly(poly, closed=True, facecolor=fc,
                             edgecolor='#d8d8d8', lw=0.4, zorder=1))
    ax.axhline(0.0, color='#888888', lw=0.7, ls=':', zorder=2)
    ax.set_xlim(lonc.min() - pad, lonc.max() + pad)
    ax.set_ylim(latc.min() - pad, latc.max() + pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    grass_fill = {}
    for kk in sorted(w.open):
        poly = np.asarray(hex9.cell(np.frombuffer(kk, dtype=np.uint8), LAYER))
        patch = MplPoly(poly, closed=True, facecolor='#ffffff',
                        edgecolor='none', zorder=1.5)
        ax.add_patch(patch)
        grass_fill[kk] = patch
    cone_dots, = ax.plot([], [], 'h', color='#dce9f7', ms=9, zorder=2.5)
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
        """Heading-oriented markers: fox = red dart along its heading,
        bunny = body dot + a small nose tick.  Quivers are rebuilt per
        frame (resizing a Quiver in place is fragile)."""
        for kind, movers, style in (
            ('F', sim.foxes, dict(color='#cc3311', scale=1 / (0.80 * w.pitch),
                                  width=0.011, headwidth=3.4, headlength=4.4,
                                  headaxislength=3.8, zorder=6)),
            ('B', sim.bunnies, dict(color='#4a4a4a', scale=1 / (0.42 * w.pitch),
                                    width=0.0042, headwidth=1.6,
                                    headlength=2.2, headaxislength=2.0,
                                    zorder=5.1)),
        ):
            if quivers[kind] is not None:
                quivers[kind].remove()
                quivers[kind] = None
            if movers:
                h = np.radians([m.heading for m in movers])
                quivers[kind] = ax.quiver(
                    [m.lon for m in movers], [m.lat for m in movers],
                    np.sin(h), np.cos(h), pivot='mid', angles='xy',
                    scale_units='xy', minshaft=1, minlength=0, **style)

    frames = None if live else ticks // frame_every

    def step(_frame):
        for _ in range(frame_every):
            sim.tick()
        level = {}
        for s in sim.grass:
            level[s.key] = level.get(s.key, 0.0) + s.size
        gold = np.array([0.953, 0.886, 0.663])
        white = np.array([1.0, 1.0, 1.0])
        for kk, patch in grass_fill.items():
            a = min(level.get(kk, 0.0) / (2 * GRASS['seed_at']), 1.0)
            patch.set_facecolor(tuple(white * (1 - a) + gold * a))
        # if sim.foxes:                       # fox[0]'s occluded view
        #     f0 = sim.foxes[0]
        #     seen = sorted(set(f0.visible_hexes(w)) - {f0.key}) if f0.key \
        #         else []
        #     cone_dots.set_data(*(centroids(seen) if seen else ([], [])))
        does = [b for b in sim.bunnies if b.sex == 'F']
        bucks = [b for b in sim.bunnies if b.sex == 'M']
        b_does.set_data([b.lon for b in does], [b.lat for b in does])
        b_bucks.set_data([b.lon for b in bucks], [b.lat for b in bucks])
        mums = [m for m in sim.foxes + sim.bunnies if m.pregnant]
        expecting.set_data([m.lon for m in mums], [m.lat for m in mums])
        draw_agents()
        nf, nb, gs = sim.history[-1][:3]
        title.set_text(f'Hexen (2012 -> hex9)   tick {len(sim.history)}   '
                       f'foxes {nf}  bunnies {nb}  grass {gs:.0f}k   '
                       f'expecting {len(mums)}  litters '
                       f'{sim.litters["B"]}B/{sim.litters["F"]}F  '
                       f'genocides F{sim.genocides["F"]}'
                       f'/B{sim.genocides["B"]}/G{sim.genocides["S"]}')
        return []

    if live:
        # a real-time window: the sim runs forever at ~25 fps until
        # the window is closed (matplotlib drives the tick loop)
        anim = FuncAnimation(fig, step, interval=40,  # noqa: F841
                             cache_frame_data=False)
        plt.show()
        print(f'live run ended at tick {len(sim.history)}')
        report(sim)
        return

    anim = FuncAnimation(fig, step, frames=frames, blit=False)
    anim.save('output/hexen.gif', writer=PillowWriter(fps=12), dpi=80)
    print(f'wrote output/hexen.gif ({frames} frames, {ticks} ticks)')

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
    axes[2].set_xlabel('tick')
    axes[2].set_title('genome drift (population medians)')
    axes[2].legend(fontsize=8, ncol=2, loc='upper left')
    ax2 = axes[2].twinx()               # scent lives on a wider scale
    ax2.plot(t, h[:, 6], '-', color='#4477aa', lw=1.4, label='bunny scent')
    ax2.plot(t, h[:, 10], '-', color='#994455', lw=1.4, label='fox scent')
    ax2.set_ylabel('scent range (pitches)')
    ax2.legend(fontsize=8, loc='upper right')
    fig2.savefig('output/hexen.png', dpi=140, bbox_inches='tight')
    print('wrote output/hexen.png')
    report(sim)


def report(sim):
    """Median genome per species and the per-entity genocide tally —
    the two things Ben wanted visible at the end of a run."""
    print(f'\n{"":8s}' + ''.join(f'{g:>10s}' for g in GENES) + '   n')
    for pop, lab in ((sim.bunnies, 'bunny'), (sim.foxes, 'fox')):
        med = sim.genome_median(pop)
        print(f'{lab:8s}' + ''.join(f'{med[g]:10.2f}' for g in GENES)
              + f'{len(pop):5d}')
    print(f'\ngenocides — foxes {sim.genocides["F"]}, '
          f'bunnies {sim.genocides["B"]}, grass {sim.genocides["S"]}'
          f'   (litters {sim.litters["B"]}B / {sim.litters["F"]}F)')


if __name__ == '__main__':
    LIVE = '--live' in sys.argv
    SEED = '--seed' in sys.argv
    np.random.seed(SEED if SEED else 4007)
    main(ticks=6000, frame_every=3 if LIVE else 10, live=LIVE)
