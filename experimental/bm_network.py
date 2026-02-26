# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import networkx as nx
import json


def run(nodes, edges):
    g = nx.Graph()

    # nodes
    for node_id, meta in nodes.items():
        g.add_node(node_id, **meta)

    # edges (pairs default kind="door"; triples override)
    for e in edges:
        if len(e) == 2:
            a, b = e
            kind = "open"
        else:
            a, b, kind = e
        g.add_edge(a, b, kind=kind)

    # sanity checks
    unknown = [(a, b) for a, b, *_ in edges if a not in g.nodes or b not in g.nodes]
    if unknown:
        raise ValueError(f"unknown node in edges: {unknown[:5]}")

    # components
    comps = list(nx.connected_components(g))
    comps.sort(key=len, reverse=True)
    print("components:", [len(c) for c in comps])



    # example: shortest path with a penalty for stairs/lifts
    # kind_cost = {"door": 1.0, "open": 1.0, "step": 3.0, "lift": 2.0}
    # for u, v, d in g.edges(data=True):
    #     d["weight"] = kind_cost.get(d.get("kind", "open"), 1.0)
    #
    # path = nx.shortest_path(g, "0e000s", "0e001n", weight="weight")
    # print(path)


if __name__ == '__main__':
    data = json.loads(open("bm_net.json","r",encoding="utf-8").read())
    nodes = data["nodes"]
    edges = data["edges"]
    run(nodes, edges)
