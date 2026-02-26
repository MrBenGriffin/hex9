"""Shared helpers for boundary handling across pipeline stages.
- YAML-driven named presets (per dataset id)
- Signature/tag generation for filenames
- needs_run: up-to-date (re)build check
- resolve_bounds: explicit → preset → data-driven with padding
- apply_bounds: filter (lat,lon) by bbox
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Default location; override by passing your own loader if needed
PRESET_FILE = Path("presets.yaml")


# ----------------------------
# I/O and up-to-date checking
# ----------------------------
def needs_run(inputs: list[Path | str], outputs: list[Path | str]) -> bool:
    """Return True if any output is missing or older than the newest input.
    Accepts either `Path` or string paths.
    """
    in_paths = [Path(p) for p in inputs]
    out_paths = [Path(p) for p in outputs]

    if any(not p.exists() for p in out_paths):
        return True

    newest_in = max((p.stat().st_mtime for p in in_paths if p.exists()), default=0.0)
    oldest_out = min((p.stat().st_mtime for p in out_paths if p.exists()), default=float("inf"))

    # If there are no existing outputs yet, we should run
    if oldest_out == float("inf"):
        return True

    return oldest_out < newest_in


# ---------------
# Preset handling
# ---------------
def load_presets(preset_file: Path | None = None) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    """Load per-dataset bounds presets from YAML, or return empty mapping if missing.

    YAML structure:
      dataset_id:
        preset_name: [min_lat, min_lon, max_lat, max_lon]
    """
    pf = preset_file or PRESET_FILE
    data = {}
    if yaml is None or not pf.exists():
        return data

    with pf.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# -----------------
# Bounds utilities
# -----------------
def _sig_from_bounds(bounds: Optional[Tuple[float, float, float, float]], preset: Optional[str]) -> str:
    """Create a short signature string for filenames based on bounds or preset name."""
    if preset:
        return preset.replace("/", "-")
    if not bounds:
        return "all"
    mi_la, mi_lo, ma_la, ma_lo = bounds
    return f"bbox_{mi_la:.2f}_{mi_lo:.2f}_{ma_la:.2f}_{ma_lo:.2f}"


def resolve_bounds(
    dataset: str,
    presets: Dict[str, Dict[str, Tuple[float, float, float, float]]],
    *,
    preset: Optional[str] = None,
    explicit: Optional[Tuple[float, float, float, float]] = None,
    data_lat_lon: Optional[np.ndarray] = None,
    pad_frac: float = 0.025,
) -> tuple[Tuple[float, float, float, float], str]:
    """Resolve bounds (min_lat, min_lon, max_lat, max_lon) and signature tag.

    Priority:
      1) explicit bounds if provided
      2) preset bounds for the given dataset
      3) data-driven min/max from `data_lat_lon` with symmetric padding (pad_frac of max span)

    Returns:
      (bounds, signature_tag)
    """
    bounds: Optional[Tuple[float, float, float, float]] = None

    if explicit is not None:
        bounds = tuple(map(float, explicit))  # type: ignore
        sig = _sig_from_bounds(bounds, preset)
        return bounds, sig

    if preset:
        ds_presets = presets.get(dataset, {})
        if preset not in ds_presets:
            raise ValueError(
                f"Unknown preset '{preset}' for dataset '{dataset}'. Known: {sorted(ds_presets)}"
            )
        bounds = ds_presets[preset]
        sig = _sig_from_bounds(bounds, preset)
        return bounds, sig

    if data_lat_lon is None:
        raise ValueError("resolve_bounds: need either explicit/preset bounds or data_lat_lon for min/max")

    # Data-driven min/max with padding
    min_lat, min_lon = np.min(data_lat_lon, axis=0)
    max_lat, max_lon = np.max(data_lat_lon, axis=0)
    lat_span = float(max_lat - min_lat)
    lon_span = float(max_lon - min_lon)
    max_span = max(lat_span, lon_span)
    pad = max_span * float(pad_frac)
    bounds = (float(min_lat - pad), float(min_lon - pad), float(max_lat + pad), float(max_lon + pad))
    # Use a stable tag for auto-computed bounds
    sig = "derived"
    return bounds, sig


def apply_bounds(lat_lon: np.ndarray, bounds: Tuple[float, float, float, float] | None) -> np.ndarray:
    """Filter rows by (min_lat, min_lon, max_lat, max_lon) bounds if provided.
    Returns the filtered lat_lon (hex_layer,2).
    """
    if not bounds:
        return lat_lon
    min_lat, min_lon, max_lat, max_lon = bounds
    mask = (
        (lat_lon[:, 0] >= min_lat) & (lat_lon[:, 0] <= max_lat) &
        (lat_lon[:, 1] >= min_lon) & (lat_lon[:, 1] <= max_lon)
    )
    return lat_lon[mask]
