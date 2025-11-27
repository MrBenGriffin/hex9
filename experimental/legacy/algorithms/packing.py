"""
Part of the H9 project
This is to do with uint64 packing of hex-digits, region-digits, and other address purposes.
"""
import numpy as np


def u64_hx_words_p(engine, octant, mode0, regions, depth):
    """Generate u64 addresses"""
    N = octant.shape[0]
    total_nibbles = 1 + depth
    W = (total_nibbles + 15) // 16
    words = np.zeros((N, W), dtype=np.uint64)
    context_initial = np.array([0, 1, 2], dtype=np.uint8)

    # This is the octant header (mode can be calculated from octant).
    # X hdr = ((octant.astype(np.uint64) << 1) | mode0.astype(np.uint64)) & np.uint64(0xF)
    hdr = octant.astype(np.uint64) & np.uint64(0xF)
    words[:, 0] |= (hdr << np.uint64(60))

    # these are states - change over each iteration.
    mode = mode0.copy()
    context = np.broadcast_to(np.asarray(context_initial, dtype=np.uint8), (N, 3)).copy()

    # helper
    def slot_for(pos_):
        """Move the new set of values up to the right position."""
        if pos_ < 15:
            w_ = 0
            shift_ = np.uint64((14 - pos_) * 4)
        else:
            pos2 = pos_ - 15
            w_ = 1 + (pos2 // 16)
            shift_ = np.uint64((15 - (pos2 % 16)) * 4)
        return w_, shift_

    # stream digits
    for pos in range(depth):
        region_id = regions[:, pos + 1]
        digit, context, mode = engine.ugc_hex_context(region_id, context, mode)
        w, shift = slot_for(pos)
        words[:, w] |= (digit.astype(np.uint64) & np.uint64(0xF)) << shift

    return words, context, mode


def u64_hx_words(octant, mode0, regions, depth):
    """Generate u64 addresses (MSB-first within each 64-bit word)."""
    N = octant.shape[0]

    # pack header (1 nibble) + depth digits
    num_digits = int(depth)
    total_nibbles = 1 + num_digits
    W = (total_nibbles + 15) // 16
    words = np.zeros((N, W), dtype=np.uint64)

    context_initial = np.array([0, 1, 2], dtype=np.uint8)

    # Header nibble (layer 0): octant at the MSB nibble of word 0
    hdr = octant.astype(np.uint64) & np.uint64(0xF)
    words[:, 0] |= (hdr << np.uint64(60))

    # rolling state
    mode = mode0.copy().astype(np.uint8, copy=False)
    context = np.broadcast_to(context_initial, (N, 3)).copy()

    # Map logical digit position -> (word index, bit shift) in MSB-first nibble order
    def slot_for(pos_):
        """pos_ = 0.. num_digits-1 → where to place the nibble."""
        if pos_ < 15:
            w_ = 0
            shift_ = np.uint64((14 - pos_) * 4)  # header is at nibble 15
        else:
            pos2 = pos_ - 15
            w_ = 1 + (pos2 // 16)
            shift_ = np.uint64((15 - (pos2 % 16)) * 4)
        return w_, shift_

    # Stream exactly depth-1 digits; drive state identically to legacy
    for pos in range(num_digits):
        region_id = regions[:, pos + 1]  # same indexing as legacy
        digit, context, mode = ugc_hex_context(region_id, context, mode)
        w, shift = slot_for(pos)
        words[:, w] |= (digit.astype(np.uint64) & np.uint64(0xF)) << shift

    return words, context.copy(), mode.copy()


def u64_pack(values, depth_to_use=None):
    """Generate u64 addresses (MSB-first within each 64-bit word)."""
    val_count = values.shape[0]
    max_depth = values.shape[1]
    req_depth = depth_to_use or max_depth
    nibbles = min(req_depth, max_depth)
    w_count = (nibbles + 15) // 16
    words = np.zeros((val_count, w_count), dtype=np.uint64)

    def slot_for(pos_):
        """
        Map logical digit position -> (word index, bit shift) in MSB-first nibble order,
        **including** the MS nibble (no header exclusion).
        """
        if pos_ < 16:
            w_ = 0
            shift_ = np.uint64((15 - pos_) * 4)  # pos 0 → shift 60 (MS nibble)
        else:
            pos2 = pos_ - 16
            w_ = 1 + (pos2 // 16)
            shift_ = np.uint64((15 - (pos2 % 16)) * 4)
        return w_, shift_

    # Stream exactly depth-1 digits; drive state identically to legacy
    for pos in range(nibbles):
        digit = values[:, pos]  # same indexing as legacy
        w, shift = slot_for(pos)
        words[:, w] |= (digit.astype(np.uint64) & np.uint64(0xF)) << shift
    return words


def u64_layers(words: np.ndarray, positions: np.ndarray = None) -> np.ndarray:
    """
    Generic nibble extractor for BCD-packed uint64 words.
    - words:     (N, W) uint64, MS word first
    - positions: array-like of nibble indices, where within each word:
                   0 = MS nibble,
                   1 = next nibble (i.e., immediately below MS),
                   ...,
                   15 = LS nibble.
                 Globally, positions increments across words: 16 = MS nibble of word 1, etc.
                 If None, we return **all nibbles** including the MS header nibble at position 0.
    Returns: (N, len(positions)) uint8, raw nibble values (0..15)
    """
    W = words.shape[1]
    max_pos = W * 16

    # By default, extract **all nibbles** including the MS header nibble at position 0.
    if positions is None:
        positions = np.arange(0, max_pos, dtype=np.uint64)
    else:
        positions = np.asarray(positions, dtype=int)

    if (positions < 0).any() or (positions >= max_pos).any():
        bad = positions[(positions < 0) | (positions >= max_pos)]
        raise IndexError(f"positions {bad} out of range [0, {max_pos - 1}]")

    # Word index for each requested nibble
    w = positions // 16

    # Nibble index within the word (0..15), where 0=MS nibble of the word
    # Correct off-by-one: for positions % 16 == 0, nibble should be 0 (MS);
    # for 1 → 1 (next below MS); ...; for 15 → 15 (LS).
    nibble = (positions % 16).astype(np.uint64)

    # Bit shifts to align nibble to LSB; MS nibble is at bit 60, LS nibble at bit 0
    sh = (np.uint64(60) - (nibble * np.uint64(4))).astype(np.uint64)

    # Gather, shift, and mask
    gathered = words[:, w]
    return ((gathered >> sh) & np.uint64(0xF)).astype(np.uint8)


if __name__ == "__main__":
    rx = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 3, 3, 1, 1, 2, 9]])
    xc = u64_pack(rx)
    cv = u64_layers(xc)
    print(f'xc: {xc[0][0]:0x}')
    print("nibbles:", cv[0, :8])
