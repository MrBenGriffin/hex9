# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Natural Earth layer resolution, with download-on-demand.

Natural Earth vector data is in the public domain (see NOTICE.md), so it is
fetched rather than committed: the repository stays light and every checkout
gets the same official, versioned bundle.

Layers are resolved to ``examples/src/landmasses`` by default and downloaded
as the official ``.zip`` archives, which carry the complete shapefile — the
``.dbf`` attributes and the ``.prj`` that declares EPSG:4326. A bare ``.shp``
loads with ``crs=None``, leaving the CRS an assumption of the calling code.

Typical use::

    from hhg9.geo.naturalearth import ne_layer
    land = load_shape_polygons(ne_layer('land', '50m'))

Downloading asks first when attached to a terminal. Pass ``prompt=False`` to
fetch unattended, or set ``consent=False`` to fail instead of downloading.

Note: the OpenStreetMap land-polygon extracts from osmdata.openstreetmap.de
are deliberately NOT handled here. They are ODbL — share-alike and
attribution-required — unlike everything else this repository ships, so they
stay a local, opt-in extra rather than something a script fetches for you.
"""

import sys
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

#: Official archive host. Layers live at {SCALE}_{CATEGORY}/ne_{SCALE}_{NAME}.zip
NE_BASE = 'https://naturalearth.s3.amazonaws.com'

#: Default destination — shared by the examples, git-ignored.
NE_DIR = Path(__file__).resolve().parents[2] / 'examples' / 'src' / 'landmasses'

#: Category per layer. Natural Earth splits its downloads this way.
NE_CATEGORY = {
    'land': 'physical',
    'ocean': 'physical',
    'lakes': 'physical',
    'minor_islands': 'physical',
    'coastline': 'physical',
    'rivers_lake_centerlines': 'physical',
    'admin_0_countries': 'cultural',
    'populated_places': 'cultural',
}

NE_SCALES = ('10m', '50m', '110m')


def ne_url(name: str, scale: str = '50m') -> str:
    """Return the official archive URL for a Natural Earth layer."""
    if scale not in NE_SCALES:
        raise ValueError(f'scale must be one of {NE_SCALES}, got {scale!r}')
    category = NE_CATEGORY.get(name)
    if category is None:
        raise ValueError(f'unknown Natural Earth layer {name!r}; '
                         f'known: {sorted(NE_CATEGORY)}')
    return f'{NE_BASE}/{scale}_{category}/ne_{scale}_{name}.zip'


def _ask(stem: str, url: str) -> bool:
    """Ask before reaching the network. Assume no when not on a terminal."""
    if not (sys.stdin and sys.stdin.isatty()):
        print(f'{stem} is missing and stdin is not a terminal — not downloading.\n'
              f'  Fetch it with: ne_layer(..., prompt=False)\n  Source: {url}',
              file=sys.stderr)
        return False
    reply = input(f'{stem} is not present.\n  Download from {url} ? [y/N] ')
    return reply.strip().lower() in ('y', 'yes')


def ne_fetch(name: str, scale: str = '50m', dest: Path | None = None,
             prompt: bool = True) -> Path:
    """Download and extract one Natural Earth layer. Returns the .shp path."""
    dest = Path(dest) if dest is not None else NE_DIR
    stem = f'ne_{scale}_{name}'
    url = ne_url(name, scale)
    if prompt and not _ask(stem, url):
        raise FileNotFoundError(
            f'{stem} not present in {dest} and download declined. '
            f'Fetch it manually from {url} and extract it there.')

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f'{stem}.zip'
    print(f'Downloading {url}')
    try:
        with urlopen(url) as response, open(archive, 'wb') as fp:
            shutil.copyfileobj(response, fp)
    except (URLError, HTTPError) as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f'could not download {url}: {exc}') from exc

    # Extract only this layer's own members, and flatten: some archives nest
    # the shapefile in a directory, and callers expect dest/<stem>.shp.
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            leaf = Path(member.filename).name
            if member.is_dir() or not leaf.startswith(stem):
                continue
            with zf.open(member) as src, open(dest / leaf, 'wb') as fp:
                shutil.copyfileobj(src, fp)
    archive.unlink(missing_ok=True)

    shp = dest / f'{stem}.shp'
    if not shp.exists():
        raise RuntimeError(f'{url} did not yield {shp.name}')
    return shp


def ne_layer(name: str, scale: str = '50m', dest: Path | None = None,
             prompt: bool = True, consent: bool = True) -> Path:
    """Return the local .shp path for a Natural Earth layer, fetching if absent.

    Args:
        name: Layer name, e.g. 'land', 'ocean', 'lakes', 'minor_islands'.
        scale: One of '10m', '50m', '110m'.
        dest: Destination directory (default ``examples/src/landmasses``).
        prompt: Ask before downloading. Ignored when the layer is present.
        consent: If False, raise rather than download a missing layer.

    Returns:
        Path to the ``.shp``. Its sibling ``.dbf``/``.prj`` are alongside it,
        so the loaded CRS is declared rather than assumed.
    """
    dest = Path(dest) if dest is not None else NE_DIR
    shp = dest / f'ne_{scale}_{name}.shp'
    if shp.exists():
        return shp
    if not consent:
        raise FileNotFoundError(
            f'{shp} not present (consent=False). Source: {ne_url(name, scale)}')
    return ne_fetch(name, scale, dest=dest, prompt=prompt)
