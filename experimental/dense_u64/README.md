# Dense (mixed-radix) 64-bit H9 packing — experimental

`dense_pack.py` packs an H9 address in its **true mixed radix** (root base-12,
region digits base-9, tail in the low nibble) instead of one hex nibble per
digit. This reclaims the ~0.83 bits/digit that nibble packing wastes and lets a
single `uint64` reach far deeper than the production `'ua'` format.

Measured round-trip resolution (uniform lat/lon grid, worst cell):

| depth | layer | max error | note |
|------:|------:|----------:|------|
| 13 | L13 | ~4.3 m  | old nibble `'ua'` |
| 14 | L14 | ~1.4 m  | current nibble `'ua'` (single-nibble tail) |
| 15 | L15 | ~0.48 m | dense default |
| 16 | L16 | ~0.16 m | |
| 17 | L17 | ~0.05 m | densest single u64 (`12·9¹⁷·16 < 2⁶⁴`) |

Run `python experimental/dense_u64/dense_pack.py` to reproduce.

## The cost: readability and truncatability

A nibble-packed address is a string of hierarchical hex digits — you can read
the hierarchy by eye and get a parent cell by chopping characters. A dense u64
is a single opaque integer: no per-digit hex, and **no cheap truncation to a
parent**. That matters for the bin question below.

## Two findings from wiring this up (see `../../` library)

### 1. Does the single-nibble tail obviate the key/address (k/a) distinction?

**Only for *canonical* (coalesced) addresses — not for the raw uint64 path.**

The reversible tail is `(p_mo<<3)|(p_c2<<1)|r_mo`; the key tail drops `p_mo`
(bit 3). They coincide iff `p_mo == 0`. Measured over 3600 points:

- **Canonical / UUID path** (`h9_enc`, which coalesces): `p_mo == 0` for
  **100%** of points; tail values ∈ {0..5}. So on the UUID path the reversible
  and key tails are byte-identical — **address ≡ bin**, and the k/a distinction
  is already moot (this is what `tail.py` documents).
- **Raw uint64 path** (`hex_pack` / `'ua'`, which does **not** coalesce):
  `p_mo == 1` for **~50%** of points. So `'ua'` and a true key differ for half
  of all cells — the distinction is real here.
- Separately, the production **key u64 branch looks degenerate**: `hex_pack`'s
  key path packs `tail_ids >> 4`, but `tail_pack_key` returns a *low*-nibble
  value, so the packed key tail is **always 0** (`'uk'` ends in `0` for every
  point). Either intentional (body-only key) or a regression from the
  single-nibble tail migration — worth confirming before any unification.

**Conclusion:** to *remove* the k/a distinction for uint64, the uint64
representative must be encoded **canonically** (coalesce like `_coalesce_bin`),
so `p_mo ≡ 0` and address ≡ bin. The tail change alone does not do it.

### 2. Bin vs address: "is this an L[i] bin or an address at an L[i] position?"

- **UUID (128-bit): already solved** the way you describe. `_coalesce_bin` fills
  nibbles `L+1..30` with `0xF` and `h9_layer` returns `UUID_DEPTH − count(0xF)`.
  `4FFF…F2`-style sentinels are exactly the mechanism.
- **uint64 (`hex_pack`): NOT solved.** A truncated `'ua5'` packs the tail
  adjacent to the body (good) but pads with **zeros, not `0xF`**, e.g.
  `9223379000000000`. A trailing real digit of `0` is then indistinguishable
  from padding, so the layer is **not recoverable** — it is not a self-describing
  L[i] bin.

To make the uint64 a proper L[i] pack, pad with `0xF` like the UUID and read
`L = 14 − count(0xF)`. This costs nothing extra (sentinels replace the zeros).

### The tension

You cannot have **both** the dense squash **and** `0xF`-sentinel L[i] bins in
the same 64 bits: the squash spends the bit-slack that the sentinels need. So
the choice for uint64 is:

- **Dense** → maximum point resolution (L15–L17), opaque, address-only (no cheap
  bins/truncation); or
- **Sentinelled nibble** → self-describing L[i] bins + readability, capped at L14.

The 128-bit UUID already gives the sentinelled-bin behaviour at high depth, so
the uint64's best distinct role is probably the **dense, deep point address**
for GIS that need a 64-bit key — which is why this prototype lives here.
