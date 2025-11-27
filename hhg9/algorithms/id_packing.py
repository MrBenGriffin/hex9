# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
In `id_packing.py`, we take sparse values and generate packing/unpacking arrays for them.
This is used for efficiency where we can use ids as indices for vectorisation and broadcasting
This method packs bit fields, so if sizes are not powers of two there will be some unused IDs between.
"""
import numpy as np
from typing import Sequence


def ceil_log2(x: int) -> int:
    """Return an optimal shift for a given dimension"""
    return 0 if x <= 1 else int(np.ceil(np.log2(x)))


def select_dtype(max_id: int) -> int:
    """Find the best dtype for an array"""
    ranges = np.array([2**8-1, 2**16-1, 2**32-1, 2**64-1], dtype=np.uint64)
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    return dtypes[np.searchsorted(ranges, max_id)]


def compose_luts(
    sizes: Sequence[int],
    *,
    fill: int | None = None,
    return_mask: bool = False,
    dtype: np.dtype | None = None,
):
    """
    General N-axis bit-packing.
    Given sizes = [S0, S1, ..., Sk-1], pack a coordinate (i0, i1, ..., ik-1) into an
    integer ID using bit fields:
        ID = sum( i_a << shift[a] ) over a=0...k-1
    where shift[a] = sum_{b>a} bits[b] and bits[b] = ceil(log2(Sb)).

    Parameters
    ----------
    sizes : sequence of int
        Axis sizes [S0, S1, ..., Sk-1].
    fill : int or None, keyword-only, default None
        If provided, fill value for **unused** decode rows (IDs that are not
        representable because sizes are not powers of two). If None, unused
        rows remain zero. Ensure `fill` fits into the selected dtype.
    return_mask : bool, default False
        If True, also return a boolean mask `used` of shape (max_id+1,)
        indicating which IDs are valid (i.e. correspond to some coordinate).
    dtype : np.dtype or None
        If provided, force the dtype of the packed IDs (`encode`) and of the
        decode table. Otherwise the dtype is chosen to fit `max_id`.

    Returns
    -------
    encode : ndarray, shape=tuple(sizes), dtype=uint{8|16|32|64}
        For every coordinate, the packed ID.
    decode : ndarray, shape=(max_id+1, k), dtype matches encode
        For every ID in 0...max_id, the coordinate (i0...ik-1). Unused IDs
        (if any due to non power-of-two packing) are set to `fill` if given,
        else zeros.
    shifts : np.ndarray, shape=(k,), int
        Bit shift per axis (most-significant first).
    bits : np.ndarray, shape=(k,), int
        Bit width per axis.
    used : np.ndarray, shape=(max_id+1,), bool  (only if return_mask=True)
        True where the ID corresponds to a valid coordinate.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.ndim != 1 or np.any(sizes <= 0):
        raise ValueError("sizes must be a 1D sequence of positive integers")

    k = sizes.size
    bits = np.array([ceil_log2(int(s)) for s in sizes], dtype=np.int64)
    shifts = np.zeros(k, dtype=np.int64)
    if k > 1:
        shifts[:-1] = np.cumsum(bits[::-1])[::-1][1:]

    # build broadcasting index grids
    idx_grids = np.meshgrid(*[np.arange(int(s), dtype=np.uint64) for s in sizes], indexing="ij")
    # pack
    enc = np.zeros(sizes.tolist(), dtype=np.uint64)
    for a in range(k):
        if bits[a] == 0:  # size 1 axis contributes nothing
            continue
        enc |= (idx_grids[a] << np.uint64(shifts[a]))

    # choose compact dtype if possible
    max_id = int(enc.max())
    auto_dtype = select_dtype(max_id)
    d_type = dtype if dtype is not None else auto_dtype
    enc = enc.astype(d_type, copy=False)
    dec_dtype = d_type

    # decode table (dense 0...max_id). Note: if any size is not a power of two,
    # some ids in 0...max_id are unused; their rows will remain zeros.
    size_ids = max_id + 1
    dec = np.zeros((size_ids, k), dtype=dec_dtype)

    # Fill decode by scattering from the coordinate arrays
    flat_ids = enc.reshape(-1).astype(np.int64, copy=False)

    # Track which IDs are actually representable
    used = np.zeros(size_ids, dtype=bool)
    used[flat_ids] = True

    # build flattened coordinates per axis
    for a in range(k):
        # broadcast grid, flattened
        coord_a = np.broadcast_to(
            np.arange(int(sizes[a]), dtype=dec_dtype).reshape(
                ([-1] + [1]*(k-1)) if a == 0 else ([1]*a + [-1] + [1]*(k-a-1))
            ),
            sizes
        ).reshape(-1)
        dec[flat_ids, a] = coord_a

    # Optionally fill unused rows with a sentinel value
    if fill is not None:
        fill_val = np.asarray(fill, dtype=dec_dtype)
        dec[~used, :] = fill_val

    enc.setflags(write=False)
    dec.setflags(write=False)
    if return_mask:
        return enc, dec, shifts, bits, used
    return enc, dec, shifts, bits
