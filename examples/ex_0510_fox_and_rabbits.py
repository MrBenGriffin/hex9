# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Fox and rabbits: simulation verbs on the symbolic hex9 grid.

Field primitives built on the libhex9 wheel's validated neighbour
algebra (hex9.k_disk / hex9.neighbors — exact integer seam crossing,
no geometry sampled).  The API is CELL-FIRST: every verb takes and
returns hex KEYS (layer-scoped bin uuids as bytes); positions appear
only when the sim binds an agent to the grid.  Internals are
vectorised — one encode/bin round per verb, not per cell.

  vision_cone(src_key, layer, bearing, half_angle, k, obstacles)
      source hex + bearing + cone angle + limit (k rings) -> the
      visible hex keys.  Cells whose line of sight from the source
      centroid passes through an obstacle hex are occluded (the
      obstacle itself is visible — you can see the hedge, just not
      through it).  Sight lines are great-circle, sampled at a third
      of the (self-calibrated) hex pitch and binned in one call.
  aim(src_key, layer, bearing)
      choose a bearing -> the neighbour hex it points at.
  walk_to(src_key, dest_key, layer, obstacles)
      choose a destination hex -> the shortest hex path (A* over
      hex9.neighbors, obstacle hexes impassable), as a key list.

Moving-target pursuit is deliberately absent: that is the sim's loop
(re-aim / re-walk per tick), not a grid verb.

The demo field straddles the equator octant seam on purpose, and a
hedge stands in the fox's cone: rabbits in its shadow are safe, and
the walk detours around it.  The seam changes nothing — it is a
property of the addressing, not the field.

Run:  python examples/ex_0510_fox_and_rabbits.py   -> output/fox_rabbits.png
"""
import heapq

import numpy as np

import hex9

LAYER = 8

hex9.init()


# ── spherical helpers ────────────────────────────────────────────────

def gc_bearing(lon1, lat1, lon2, lat2):
    """Initial great-circle bearing, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.subtract(lon2, lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def gc_angle(lon1, lat1, lon2, lat2):
    """Central angle, degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.subtract(lon2, lon1))
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _xyz(lon, lat):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def _lonlat(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])),
            np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))))


# ── key plumbing (full uuid in, bin KEY out — walk by re-encode) ─────

def encode_keys(lon, lat, layer=LAYER):
    """Positions -> hex keys, one batch call.  The sim's bind verb."""
    u = hex9.encode(np.atleast_1d(np.asarray(lon, dtype=np.float64)),
                    np.atleast_1d(np.asarray(lat, dtype=np.float64)))
    b = hex9.bin(np.asarray(u, dtype=np.uint8).reshape(-1, 16), layer)
    return [bytes(r) for r in np.asarray(b, dtype=np.uint8).reshape(-1, 16)]


def encode_keys_multi(lon, lat, layers):
    """Positions -> {layer: [keys]} for SEVERAL layers at once.

    The hierarchy leverage: binning is a re-bin of the SAME full uuid
    at each depth, so a coarse cell costs no extra encode — one call
    fixes a point on the sphere and every resolution follows.  A sim
    can therefore run its processes at whatever grain each deserves
    (fine forage, coarse neighbourhoods) off a single bind."""
    u = np.asarray(
        hex9.encode(np.atleast_1d(np.asarray(lon, dtype=np.float64)),
                    np.atleast_1d(np.asarray(lat, dtype=np.float64))),
        dtype=np.uint8).reshape(-1, 16)
    out = {}
    for layer in layers:
        b = np.asarray(hex9.bin(u, layer), dtype=np.uint8).reshape(-1, 16)
        out[layer] = [bytes(r) for r in b]
    return out


def centroids(keys):
    """Keys -> (lon, lat) arrays, one batch call."""
    a = np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(-1, 16)
    lon, lat = hex9.decode(a)
    return np.asarray(lon), np.asarray(lat)


def _full_of(key):
    lon, lat = centroids([key])
    u = hex9.encode(lon, lat)
    return np.asarray(u, dtype=np.uint8).reshape(16)


def _pitch_of(key, layer):
    """Median centroid step to the neighbours — the local hex pitch."""
    nbs, cnt = hex9.neighbors(_full_of(key)[None, :], layer)
    nb = np.asarray(nbs, dtype=np.uint8)[0, :int(np.asarray(cnt)[0])]
    lon0, lat0 = centroids([key])
    nlon, nlat = hex9.decode(nb)
    return float(np.median(gc_angle(lon0, lat0, nlon, nlat)))


# ── the three simulation verbs ───────────────────────────────────────

def vision_cone(src_key, layer, bearing, half_angle, k,
                obstacles=frozenset()):
    """Visible hex keys from src_key: within k rings, within
    half_angle of bearing, and not occluded by an obstacle hex.
    The source hex is always visible."""
    u = _full_of(src_key)
    slon, slat = (float(v[0]) for v in centroids([src_key]))
    disk = np.asarray(hex9.k_disk(u, layer, k), dtype=np.uint8).reshape(-1, 16)
    keys = [bytes(r) for r in disk]
    dlon, dlat = hex9.decode(disk)
    br = gc_bearing(slon, slat, dlon, dlat)
    dist = gc_angle(slon, slat, dlon, dlat)
    in_cone = np.abs(wrap180(br - bearing)) <= half_angle

    cand = [i for i in range(len(keys))
            if keys[i] != src_key and in_cone[i]]
    seen = {src_key}
    if not cand:
        return seen
    if not obstacles:
        seen.update(keys[i] for i in cand)
        return seen

    # occlusion: sample every sight line at pitch/3, bin all samples
    # in ONE encode/bin round, then test obstacle membership per line
    pitch = _pitch_of(src_key, layer)
    a = _xyz(slon, slat)
    ns = np.maximum(2, np.ceil(dist[cand] / (pitch / 3)).astype(int))
    pts, owner = [], []
    for j, i in enumerate(cand):
        t = (np.arange(ns[j]) + 0.5) / ns[j]
        seg = a[None, :] * (1 - t[:, None]) + _xyz(dlon[i], dlat[i]) * t[:, None]
        pts.append(seg)
        owner.extend([j] * ns[j])
    slon_s, slat_s = _lonlat(np.concatenate(pts))
    skeys = encode_keys(slon_s, slat_s, layer)
    blocked = np.zeros(len(cand), dtype=bool)
    for kk, j in zip(skeys, owner):
        if kk in obstacles and kk != src_key and kk != keys[cand[j]]:
            blocked[j] = True
    seen.update(keys[i] for j, i in enumerate(cand) if not blocked[j])
    return seen


def aim(src_key, layer, bearing):
    """The neighbour hex whose centroid bearing best matches."""
    nbs, cnt = hex9.neighbors(_full_of(src_key)[None, :], layer)
    nb = np.asarray(nbs, dtype=np.uint8)[0, :int(np.asarray(cnt)[0])]
    slon, slat = centroids([src_key])
    nlon, nlat = hex9.decode(nb)
    br = gc_bearing(slon, slat, nlon, nlat)
    return bytes(nb[int(np.argmin(np.abs(wrap180(br - bearing))))])


def walk_to(src_key, dest_key, layer, obstacles=frozenset(),
            max_expand=5000):
    """Shortest hex path src -> dest (A*; obstacle hexes impassable).
    Returns the key list (first = src, last = dest), or None."""
    dlon, dlat = (float(v[0]) for v in centroids([dest_key]))
    pitch = _pitch_of(src_key, layer)

    def h(key):
        lon, lat = centroids([key])
        return float(gc_angle(lon[0], lat[0], dlon, dlat)) / pitch * 0.95

    openq = [(h(src_key), 0, src_key)]
    came, g = {src_key: None}, {src_key: 0}
    for _ in range(max_expand):
        if not openq:
            break
        _, gc, cur = heapq.heappop(openq)
        if cur == dest_key:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if gc > g[cur]:
            continue
        nbs, cnt = hex9.neighbors(_full_of(cur)[None, :], layer)
        nb = np.asarray(nbs, dtype=np.uint8)[0, :int(np.asarray(cnt)[0])]
        for row in nb:
            kk = bytes(row)
            if kk in obstacles and kk != dest_key:
                continue
            if kk not in g or g[cur] + 1 < g[kk]:
                g[kk] = g[cur] + 1
                came[kk] = cur
                heapq.heappush(openq, (g[kk] + h(kk), g[kk], kk))
    return None


# ── demo: one fox, forty rabbits, a hedge, a field over the seam ─────

def main():
    rng = np.random.default_rng(7)
    fox = (35.02, -0.11)                    # (lon, lat) — south of the seam
    bearing, half_angle, k = 15.0, 32.0, 6

    fox_key = encode_keys(*fox)[0]
    n = 40
    r_lon = rng.uniform(34.62, 35.42, n)
    r_lat = rng.uniform(-0.38, 0.42, n)
    r_keys = encode_keys(r_lon, r_lat)

    # a hedge: hexes along a line across the middle of the cone
    t = np.linspace(0, 1, 40)
    hedge = set(encode_keys(34.94 + t * (35.20 - 34.94),
                            0.235 + t * (0.185 - 0.235)))
    hedge.discard(fox_key)

    cone = vision_cone(fox_key, LAYER, bearing, half_angle, k, hedge)
    open_cone = vision_cone(fox_key, LAYER, bearing, half_angle, k)
    shadow = open_cone - cone
    seen = [i for i in range(n) if r_keys[i] in cone and r_keys[i] not in hedge]
    safe = [i for i in range(n) if r_keys[i] in shadow]

    aim_key = aim(fox_key, LAYER, bearing)
    target = max(seen, key=lambda i: gc_angle(*fox, r_lon[i], r_lat[i]),
                 default=None)
    path = (walk_to(fox_key, r_keys[target], LAYER, hedge)
            if target is not None else None)

    _, clat = centroids(sorted(cone))
    print(f'cone: {len(cone)} hexes visible, {len(shadow)} in hedge shadow '
          f'({int(np.sum(clat < 0))} visible south of the seam)')
    print(f'fox sees {len(seen)}/{n} rabbits; '
          f'{len(safe)} rabbit(s) hidden by the hedge')
    if path:
        print(f'walk to farthest visible rabbit: {len(path) - 1} steps '
              f'(detours around the hedge)')

    render(fox, bearing, half_angle, k, fox_key, cone, shadow, hedge,
           aim_key, path or [], r_lon, r_lat, seen, safe)


def render(fox, bearing, half_angle, k, fox_key, cone, shadow, hedge,
           aim_key, path, r_lon, r_lat, seen, safe):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly, Patch

    fig, ax = plt.subplots(figsize=(9, 8))

    def outline(key, **kw):
        poly = np.asarray(hex9.cell(np.frombuffer(key, dtype=np.uint8), LAYER))
        ax.add_patch(MplPoly(poly, closed=True, **kw))

    disk = np.asarray(hex9.k_disk(_full_of(fox_key), LAYER, k),
                      dtype=np.uint8).reshape(-1, 16)
    path_set = set(path)
    for kk in {bytes(r) for r in disk} | hedge:
        if kk in path_set:
            fc, ec, lw = '#bcd4ee', '#999999', 0.5      # walked: pale blue
        elif kk in hedge:
            fc, ec, lw = '#8f9aa6', '#6b7683', 0.5      # hedge: slate grey
        elif kk in shadow:
            fc, ec, lw = '#e8e8e8', '#c9a227', 0.7      # occluded: gold edge
        elif kk in cone:
            fc, ec, lw = '#f3e2a9', '#999999', 0.5      # seen: pale gold
        else:
            fc, ec, lw = '#f5f5f5', '#bbbbbb', 0.4      # in range, unseen
        outline(kk, facecolor=fc, edgecolor=ec, lw=lw, zorder=1)
    outline(aim_key, facecolor='none', edgecolor='#1f4e9c',
            lw=2.2, ls='--', zorder=4)

    ax.axhline(0.0, color='#666666', lw=0.8, ls=':', zorder=2)
    ax.annotate('octant seam (equator)', (34.6, 0.005), fontsize=8,
                color='#666666')

    reach = 6.5 * 0.12
    for a in (bearing - half_angle, bearing + half_angle):
        ax.plot([fox[0], fox[0] + reach * np.sin(np.radians(a))],
                [fox[1], fox[1] + reach * np.cos(np.radians(a))],
                color='#1f4e9c', lw=1.2, zorder=3)
    ax.annotate('', xy=(fox[0] + 0.35 * reach * np.sin(np.radians(bearing)),
                        fox[1] + 0.35 * reach * np.cos(np.radians(bearing))),
                xytext=fox,
                arrowprops=dict(arrowstyle='-|>', color='#1f4e9c', lw=1.6),
                zorder=5)

    hid = [i for i in range(len(r_lon)) if i not in seen and i not in safe]
    ax.plot(r_lon[hid], r_lat[hid], 'o', mfc='none', mec='#888888',
            ms=5, zorder=5, label=f'rabbits unseen ({len(hid)})')
    ax.plot(r_lon[safe], r_lat[safe], 's', mfc='white', mec='#4a4a4a',
            ms=6, zorder=6, label=f'rabbits in shadow ({len(safe)})')
    ax.plot(r_lon[seen], r_lat[seen], 'o', color='#b8860b', ms=6,
            zorder=6, label=f'rabbits seen ({len(seen)})')
    ax.plot(*fox, marker='^', color='#12325e', ms=13, zorder=7,
            label='fox')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='#8f9aa6', edgecolor='#6b7683'))
    labels.append('hedge')

    ax.set_aspect('equal')
    ax.set_xlim(34.58, 35.46)
    ax.set_ylim(-0.42, 0.46)
    ax.set_title(f'vision_cone(bearing {bearing:.0f}°, ±{half_angle:.0f}°, '
                 f'k={k}, hedge) · aim (dashed) · walk_to (blue) — '
                 f'layer {LAYER}')
    ax.legend(handles, labels, loc='lower right', fontsize=9)
    fig.savefig('output/fox_rabbits.png', dpi=200, bbox_inches='tight')
    print('wrote output/fox_rabbits.png')


if __name__ == '__main__':
    main()
