import json
from collections import defaultdict

def room_label(tags, way_id):
    # Prefer alt_name (e.g. "Room 4") then name, else way id.
    alt = tags.get("alt_name")
    name = tags.get("name")
    if alt and name:
        return f"{alt} — {name}"
    if alt:
        return alt
    if name:
        return name
    return f"way:{way_id}"

def build_room_adjacency(overpass_json, min_shared_nodes=2):
    elements = overpass_json["elements"]

    rooms = []
    for el in elements:
        if el.get("type") == "way":
            tags = el.get("tags", {})
            if tags.get("indoor") == "room":
                rooms.append({
                    "id": el["id"],
                    "label": room_label(tags, el["id"]),
                    "nodes": set(el["nodes"]),
                    "tags": tags,
                })

    # pairwise compare: O(n^2) fine for hundreds of rooms
    adj = defaultdict(set)
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            shared = rooms[i]["nodes"] & rooms[j]["nodes"]
            if len(shared) >= min_shared_nodes:
                a = rooms[i]["label"]
                b = rooms[j]["label"]
                adj[a].add(b)
                adj[b].add(a)

    # make JSON-friendly (sorted lists)
    return {k: sorted(v) for k, v in sorted(adj.items(), key=lambda kv: kv[0])}


if __name__ == '__main__':
    data = json.loads(open("bm.json","r",encoding="utf-8").read())
    adj = build_room_adjacency(data, min_shared_nodes=2)
    print(json.dumps(adj, indent=2, ensure_ascii=False))
