"""
Part of the H9 project - Preparation 0001
Load a CSV file and store it as numpy.
In this case, we are loading the Meta General Population csv.
https://data.humdata.org/organization/meta
This is 33k data points, with fractional population counts.
longitude,latitude,_general_2020
Last Tested 12 August 2025 √
"""
import numpy as np
from pathlib import Path


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          keep_zeros: bool = True,
          force: bool = False,
          **_):
    """Convert `<src_dir>/<file>_general_2020.csv` → `<dst_dir>/<file>_lon_lat_pop.npy`.
    - Skips work if the output is newer than the input (unless `force=True`).
    - Drops rows with NaNs.
    - Drops zero-population rows by default (set `keep_zeros=True` to retain them).
    """
    src_path = Path(src_dir) / f"{file}_general_2020.csv"
    out_path = Path(dst_dir) / f"{file}_lat_lon_pop.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {src_path}")

    if out_path.exists() and not force and out_path.stat().st_mtime >= src_path.stat().st_mtime:
        print(f"[pr0001] up-to-date: {out_path}")
        return out_path

    # Load CSV (longitude,latitude,_general_2020)
    def quoted_float(s):
        """Return float from string `s`, honouring bytes, stripping whitespace."""
        if isinstance(s, bytes):            # s may be bytes.
            s = s.decode('utf-8', 'ignore')
        s = s.strip().strip('"')
        return float(s) if s else 0  # Return 0 for empty strings.

    data = np.genfromtxt(
        src_path,
        delimiter=',',
        skip_header=1,
        autostrip=True,
        usecols=(0, 1, 2),
        converters={0: quoted_float, 1: quoted_float, 2: quoted_float},
        dtype=np.float64)
    if data.ndim == 1:  # single row
        data = data.reshape(1, -1)

    # Drop rows with NaNs (some datasets have sparse/noisy cells)
    if data.size and data.shape[1] >= 3:
        mask = np.all(np.isfinite(data[:, :3]), axis=1)
        data = data[mask]

    # Drop zero-pop rows unless explicitly kept
    if data.size and data.shape[1] >= 3 and not keep_zeros:
        nz = data[:, 2] > 0
        removed = int((~nz).sum())
        if removed:
            print(f"[pr0001] filtered out {removed} zero-pop rows")
        data = data[nz]

    records = data.shape[0]
    print(f"[pr0001] Converted {src_path} → {records} rows. Writing {out_path}")
    np.save(out_path, data)
    return out_path


if __name__ == '__main__':
    stage("gbr")  # default behaviour filters zeros
