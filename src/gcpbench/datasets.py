"""Dataset registry (plan S0.2, A1, A2, D1, D2, D6).

Sets
----
VAL      The 140 instances of the submitted paper (Table 4), relabelled as the
         validation set. All development decisions (candidate selection,
         failure instances for refinement) use VAL only.
         Either regenerate them (``--set VAL``, master seed 42, identical to the
         original generator) or register the authors' original files with
         ``register-dir`` -- strongly preferred if they still exist, because the
         paper's instances may not have come from seed 42.
TEST-A   Fresh 140 instances, same families/scales/parameters/counts as Table 4,
         new master seed; checked to be hash-disjoint from VAL.
TEST-B   Balanced design: 20 instances per family x scale cell (180), one
         vertex range per scale for all families, and parameters chosen to
         match the mean degree across families within a scale (answers R1.1).
PLANTED  Graphs with a planted k-colouring and a planted k-clique, so
         chi = k exactly by construction (D2).
DIMACS   Standard DIMACS/COLOR02 instances (D1), read from .col files placed in
         data/dimacs/ (see scripts/fetch_dimacs.py); bounds are recorded with
         source and retrieval date in data/dimacs/bounds.csv.

Every instance records generator, parameters, master and instance seeds and
the SHA-256 of its canonical edge list. ``lock`` writes
registry/datasets.lock.json; ``verify`` regenerates/re-reads and checks every
hash, so TEST/EXT sets cannot change silently after the protocol freeze.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random

import networkx as nx

from .graph import Graph, write_dimacs, read_graph

# ----------------------------------------------------------------- specs
# Exactly the original generator (data/generate_graphs.py of the submission).
LEGACY_SPEC = [
    ("ER", "small",  (30, 120),    lambda rng: {"p": rng.choice([0.10, 0.20])}, 20),
    ("ER", "medium", (500, 1000),  lambda rng: {"p": 0.05},                     20),
    ("ER", "large",  (2000, 5000), lambda rng: {"p": rng.choice([0.01, 0.02])}, 10),
    ("BA", "small",  (30, 120),    lambda rng: {"m": rng.choice([2, 4])}, 20),
    ("BA", "medium", (500, 1000),  lambda rng: {"m": 4},                  20),
    ("BA", "large",  (2000, 5000), lambda rng: {"m": 4},                  10),
    ("WS", "small",  (120, 120),   lambda rng: {"k": 8,  "p": 0.1}, 10),
    ("WS", "medium", (500, 1000),  lambda rng: {"k": 20, "p": 0.1}, 20),
    ("WS", "large",  (2000, 5000), lambda rng: {"k": 20, "p": 0.1}, 10),
]

# TEST-B: one vertex range per scale for every family; target mean degree per scale.
TESTB_SCALES = {"small": ((30, 120), 8), "medium": ((500, 1000), 20), "large": ((2000, 5000), 20)}
TESTB_PER_CELL = 20

# PLANTED: (n, k, mean degree) grid, replicated.
PLANTED_GRID = [(n, k, 4 * k) for n in (500, 2000, 5000) for k in (5, 10, 20)]
PLANTED_REPS = 4

MASTER_SEEDS = {"VAL": 42, "TEST-A": 20261118, "TEST-B": 20261119, "PLANTED": 20261120}


def _make(family: str, n: int, params: dict, gseed: int) -> nx.Graph:
    if family == "ER":
        return nx.erdos_renyi_graph(n, params["p"], seed=gseed)
    if family == "BA":
        return nx.barabasi_albert_graph(n, min(params["m"], n - 1), seed=gseed)
    if family == "WS":
        k = min(params["k"], n - 1 if (n - 1) % 2 == 0 else n - 2)
        return nx.watts_strogatz_graph(n, k, params["p"], seed=gseed)
    raise ValueError(family)


def _legacy_instances(master_seed: int):
    """Same RNG call sequence as the original generator: n, params, then graph seed."""
    rng = random.Random(master_seed)
    for family, scale, (lo, hi), pfn, count in LEGACY_SPEC:
        for idx in range(count):
            n = rng.randint(lo, hi) if lo != hi else lo
            params = pfn(rng)
            gseed = rng.randint(0, 2**31)
            yield family, scale, idx, n, params, gseed


def _testb_instances(master_seed: int):
    rng = random.Random(master_seed)
    for scale, ((lo, hi), d) in TESTB_SCALES.items():
        for family in ("ER", "BA", "WS"):
            for idx in range(TESTB_PER_CELL):
                n = rng.randint(lo, hi)
                if family == "ER":
                    params = {"p": round(d / (n - 1), 6)}
                elif family == "BA":
                    params = {"m": d // 2}
                else:
                    params = {"k": d, "p": 0.1}
                gseed = rng.randint(0, 2**31)
                yield family, scale, idx, n, params, gseed


def planted_graph(n: int, k: int, mean_degree: float, seed: int) -> tuple[nx.Graph, list]:
    """Random k-partite graph (equal classes, inter-class edge prob. p) plus one
    planted k-clique, so that chi = k. Returns (graph, hidden colouring)."""
    rng = random.Random(seed)
    cls = [i % k for i in range(n)]
    rng.shuffle(cls)
    p = mean_degree / (n * (k - 1) / k)
    G = nx.fast_gnp_random_graph(n, min(p, 1.0), seed=rng.randint(0, 2**31))
    G.remove_edges_from([(u, v) for u, v in list(G.edges()) if cls[u] == cls[v]])
    reps = [rng.choice([v for v in range(n) if cls[v] == c]) for c in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            G.add_edge(reps[i], reps[j])
    return G, cls


# ------------------------------------------------------------- generation

def generate(set_name: str, out_dir: str, master_seed: int | None = None,
             legacy_edgelist: bool = False) -> list[dict]:
    seed = MASTER_SEEDS[set_name] if master_seed is None else master_seed
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    if set_name in ("VAL", "TEST-A"):
        it = (("legacy",) + t for t in _legacy_instances(seed))
    elif set_name == "TEST-B":
        it = (("testb",) + t for t in _testb_instances(seed))
    elif set_name == "PLANTED":
        it = _planted_iter(seed)
    else:
        raise ValueError(set_name)

    for kind, family, scale, idx, n, params, gseed in it:
        hidden = None
        if family == "PLANTED":
            G, hidden = planted_graph(n, params["k"], params["d"], gseed)
            gen = "planted_graph"
        else:
            G = _make(family, n, params, gseed)
            gen = {"ER": "erdos_renyi_graph", "BA": "barabasi_albert_graph",
                   "WS": "watts_strogatz_graph"}[family]
        stem = f"{family}_{scale}_{idx:03d}"
        if legacy_edgelist:
            fname = stem + ".edgelist"
            nx.write_edgelist(G, os.path.join(out_dir, fname), data=False)
            g = Graph.from_networkx(nx.read_edgelist(os.path.join(out_dir, fname), nodetype=int))
        else:
            fname = stem + ".col"
            g = Graph.from_networkx(G)
            write_dimacs(g, os.path.join(out_dir, fname),
                         comment=f"set={set_name} family={family} scale={scale} params={params} seed={gseed}")
        row = {
            "set": set_name, "instance_id": fname, "family": family, "scale": scale,
            "n_vertices": g.n, "n_edges": g.m, "n_generated": n,
            "mean_degree": round(2 * g.m / g.n, 4) if g.n else 0.0,
            "params": json.dumps(params, sort_keys=True), "generator": f"networkx.{gen}",
            "networkx_version": nx.__version__, "master_seed": seed, "instance_seed": gseed,
            "sha256": g.sha256(),
            "chi_planted": params.get("k", "") if family == "PLANTED" else "",
        }
        rows.append(row)
    _write_manifest(out_dir, rows)
    return rows


def _planted_iter(seed: int):
    rng = random.Random(seed)
    idx = 0
    for n, k, d in PLANTED_GRID:
        for _ in range(PLANTED_REPS):
            yield ("planted", "PLANTED", f"n{n}_k{k}", idx, n, {"k": k, "d": d}, rng.randint(0, 2**31))
            idx += 1


def register_dir(set_name: str, in_dir: str, meta_csv: str | None = None) -> list[dict]:
    """Register existing instance files (e.g. the paper's original 140) as a set."""
    meta = {}
    if meta_csv and os.path.exists(meta_csv):
        with open(meta_csv) as f:
            meta = {r["instance_id"]: r for r in csv.DictReader(f)}
    rows = []
    for fname in sorted(os.listdir(in_dir)):
        if not (fname.endswith(".edgelist") or fname.endswith(".col")):
            continue
        g = read_graph(os.path.join(in_dir, fname))
        fam, scale = (fname.split("_") + ["?", "?"])[:2]
        m = meta.get(fname, {})
        rows.append({
            "set": set_name, "instance_id": fname, "family": m.get("family", fam),
            "scale": m.get("scale", scale), "n_vertices": g.n, "n_edges": g.m,
            "n_generated": m.get("n_vertices", g.n),
            "mean_degree": round(2 * g.m / g.n, 4) if g.n else 0.0,
            "params": m.get("params", ""), "generator": "registered-file",
            "networkx_version": "", "master_seed": "", "instance_seed": "",
            "sha256": g.sha256(), "chi_planted": "",
        })
    _write_manifest(in_dir, rows)
    return rows


def _write_manifest(out_dir: str, rows: list[dict]) -> None:
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_manifest(set_dir: str, scale: str | None = None, family: str | None = None) -> list[dict]:
    with open(os.path.join(set_dir, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))
    if scale:
        rows = [r for r in rows if r["scale"] == scale]
    if family:
        rows = [r for r in rows if r["family"] == family]
    return rows


def load_set(set_dir: str, scale: str | None = None, family: str | None = None):
    """Yield (manifest_row, Graph) for each instance of a set."""
    for row in load_manifest(set_dir, scale, family):
        g = read_graph(os.path.join(set_dir, row["instance_id"]))
        g.name = row["instance_id"]
        yield row, g


# ------------------------------------------------------------------- lock

def lock(set_dirs: dict, lock_path: str) -> dict:
    out = {"networkx_version": nx.__version__, "sets": {}}
    for name, d in set_dirs.items():
        rows = load_manifest(d)
        items = sorted((r["instance_id"], r["sha256"]) for r in rows)
        h = hashlib.sha256("".join(f"{a}:{b}\n" for a, b in items).encode()).hexdigest()
        out["sets"][name] = {"n_instances": len(items), "set_sha256": h,
                             "instances": dict(items)}
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def verify(set_dirs: dict, lock_path: str) -> list[str]:
    """Re-read every instance file and compare hashes with the lock file."""
    with open(lock_path) as f:
        locked = json.load(f)
    problems = []
    for name, d in set_dirs.items():
        ref = locked["sets"].get(name)
        if ref is None:
            problems.append(f"{name}: not in lock file")
            continue
        for row, g in load_set(d):
            want = ref["instances"].get(row["instance_id"])
            got = g.sha256()
            if want != got:
                problems.append(f"{name}/{row['instance_id']}: hash {got[:12]} != locked {str(want)[:12]}")
        if len(load_manifest(d)) != ref["n_instances"]:
            problems.append(f"{name}: instance count changed")
    return problems


def check_disjoint(dir_a: str, dir_b: str) -> int:
    a = {r["sha256"] for r in load_manifest(dir_a)}
    b = {r["sha256"] for r in load_manifest(dir_b)}
    return len(a & b)


# ----------------------------------------------------------------- DIMACS

# Classes requested in plan E-D1. Bounds are NOT hard-coded: fill
# data/dimacs/bounds.csv from a published table, with source and date.
DIMACS_INSTANCES = {
    "DSJC": ["DSJC125.1", "DSJC125.5", "DSJC125.9", "DSJC250.1", "DSJC250.5", "DSJC250.9",
             "DSJC500.1", "DSJC500.5", "DSJC500.9", "DSJC1000.1", "DSJC1000.5", "DSJC1000.9"],
    "le450": [f"le450_{k}{s}" for k in (5, 15, 25) for s in "abcd"],
    "flat": ["flat300_20_0", "flat300_26_0", "flat300_28_0",
             "flat1000_50_0", "flat1000_60_0", "flat1000_76_0"],
    "queen": ["queen5_5", "queen6_6", "queen7_7", "queen8_8", "queen8_12", "queen9_9",
              "queen10_10", "queen11_11", "queen12_12", "queen13_13", "queen14_14",
              "queen15_15", "queen16_16"],
    "myciel": [f"myciel{i}" for i in range(3, 8)],
    "school": ["school1", "school1_nsh"],
    "latin_square": ["latin_square_10"],
    "register": ["fpsol2.i.1", "fpsol2.i.2", "fpsol2.i.3", "inithx.i.1", "inithx.i.2",
                 "inithx.i.3", "mulsol.i.1", "mulsol.i.2", "mulsol.i.3", "mulsol.i.4",
                 "mulsol.i.5", "zeroin.i.1", "zeroin.i.2", "zeroin.i.3"],
}


def register_dimacs(dimacs_dir: str) -> list[dict]:
    rows = []
    cls_of = {name: c for c, names in DIMACS_INSTANCES.items() for name in names}
    for fname in sorted(os.listdir(dimacs_dir)):
        if not fname.endswith(".col"):
            continue
        g = read_graph(os.path.join(dimacs_dir, fname))
        stem = fname[:-4]
        rows.append({
            "set": "DIMACS", "instance_id": fname, "family": cls_of.get(stem, "other"),
            "scale": "small" if g.n <= 150 else "medium" if g.n <= 1000 else "large",
            "n_vertices": g.n, "n_edges": g.m, "n_generated": g.n,
            "mean_degree": round(2 * g.m / g.n, 4), "params": "", "generator": "DIMACS file",
            "networkx_version": "", "master_seed": "", "instance_seed": "",
            "sha256": g.sha256(), "chi_planted": "",
        })
    if rows:
        _write_manifest(dimacs_dir, rows)
    return rows
