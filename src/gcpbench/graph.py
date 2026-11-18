"""Graph container, I/O and the five vertex features.

Vertices are relabelled 0..n-1 in ascending order of their original
integer labels, so "lowest vertex id wins ties" means the same thing as in
the original code (which broke ties by the original label).

Feature definitions follow Table 2 of the manuscript:
    sat(v)    distinct colours among coloured neighbours        dynamic
    deg(v)    number of incident edges                          static
    ucn(v)    neighbours not yet coloured                       dynamic
    cc(v)     local clustering coefficient (networkx definition) static
    twohop(v) vertices at distance exactly 2                    static
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Graph:
    n: int
    adj: list                         # list[list[int]], sorted neighbour lists
    labels: list = field(default_factory=list)   # original label of each vertex
    name: str = ""
    _feats: dict | None = None
    _nx: nx.Graph | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_networkx(cls, G: nx.Graph, name: str = "") -> "Graph":
        labels = sorted(G.nodes())
        index = {lab: i for i, lab in enumerate(labels)}
        adj = [sorted(index[u] for u in G.neighbors(lab) if u != lab) for lab in labels]
        g = cls(n=len(labels), adj=adj, labels=labels, name=name)
        g._nx = G
        return g

    @classmethod
    def from_edges(cls, n: int, edges, name: str = "") -> "Graph":
        sets = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                sets[u].add(v)
                sets[v].add(u)
        return cls(n=n, adj=[sorted(s) for s in sets], labels=list(range(n)), name=name)

    @property
    def m(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def edges(self):
        for u, nb in enumerate(self.adj):
            for v in nb:
                if u < v:
                    yield u, v

    def to_networkx(self) -> nx.Graph:
        if self._nx is None or self._nx.number_of_nodes() != self.n:
            G = nx.Graph()
            G.add_nodes_from(range(self.n))
            G.add_edges_from(self.edges())
            self._nx = G
            self.labels = list(range(self.n))
        return self._nx

    def csr(self):
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(a) for a in self.adj])
        indices = np.fromiter((v for a in self.adj for v in a), dtype=np.int64,
                              count=int(indptr[-1]))
        return indptr, indices

    def sha256(self) -> str:
        """Hash of the canonical form (n, sorted edge list); label-independent of file format."""
        h = hashlib.sha256()
        h.update(f"n={self.n}\n".encode())
        for u, v in self.edges():
            h.update(f"{u} {v}\n".encode())
        return h.hexdigest()

    # --------------------------------------------------------------- features
    def static_features(self):
        """(deg, cc, twohop) as Python lists, computed exactly as the original code."""
        if self._feats is None:
            G = self._nx
            deg = [len(a) for a in self.adj]
            if G is not None and G.number_of_nodes() == self.n:
                ccd = nx.clustering(G)
                cc = [ccd[lab] for lab in self.labels]
            else:
                ccd = nx.clustering(self.to_networkx())
                cc = [ccd[i] for i in range(self.n)]
            twohop = []
            for v in range(self.n):
                nb = set(self.adj[v])
                two = set()
                for u in self.adj[v]:
                    two.update(self.adj[u])
                two -= nb
                two.discard(v)
                twohop.append(len(two))
            self._feats = {"deg": deg, "cc": cc, "twohop": twohop}
        f = self._feats
        return f["deg"], f["cc"], f["twohop"]

    def feature_arrays(self):
        deg, cc, th = self.static_features()
        return (np.asarray(deg, dtype=np.float64), np.asarray(cc, dtype=np.float64),
                np.asarray(th, dtype=np.float64))


# ---------------------------------------------------------------------- I/O

def read_edgelist(path: str) -> Graph:
    """Legacy format written by nx.write_edgelist (isolated vertices are absent)."""
    G = nx.read_edgelist(path, nodetype=int)
    return Graph.from_networkx(G, name=os.path.basename(path))


def read_dimacs(path: str) -> Graph:
    """DIMACS .col format ('p edge n m' / 'e u v', 1-based)."""
    n = None
    edges = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            t = line.split()
            if t[0] == "p":
                n = int(t[2])
            elif t[0] == "e":
                edges.append((int(t[1]) - 1, int(t[2]) - 1))
    if n is None:
        raise ValueError(f"{path}: no 'p' line")
    return Graph.from_edges(n, edges, name=os.path.basename(path))


def write_dimacs(g: Graph, path: str, comment: str = "") -> None:
    with open(path, "w") as f:
        for line in comment.splitlines():
            f.write(f"c {line}\n")
        f.write(f"p edge {g.n} {g.m}\n")
        for u, v in g.edges():
            f.write(f"e {u + 1} {v + 1}\n")


def read_graph(path: str) -> Graph:
    if path.endswith(".col") or path.endswith(".col.b"):
        return read_dimacs(path)
    return read_edgelist(path)


def verify_coloring(g: Graph, colors) -> bool:
    if any(c is None or c < 0 for c in colors):
        return False
    return all(colors[u] != colors[v] for u, v in g.edges())


def n_colors(colors) -> int:
    return len(set(colors)) if len(colors) else 0
