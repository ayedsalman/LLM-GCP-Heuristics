# Algorithms and complexity (revision plan E3, E4; answers R1.6)

Notation: n vertices, m edges, Δ maximum degree, k colours used, R GPU rounds.
Features (Table 2): sat(v), deg(v), ucn(v) dynamic/static as stated; cc(v),
twohop(v) static. Ties are broken by the **lowest vertex id** everywhere.

## Algorithm 1: sequential greedy colouring with a scoring formula

```
Input : graph G = (V, E), scoring formula s(sat, deg, ucn, cc, twohop)
Output: proper colouring c : V -> {0, 1, ...}

 1  precompute deg(v), cc(v), twohop(v) for all v                       -- static features
 2  for v in V: sat(v) <- 0; ucn(v) <- deg(v); C(v) <- {}; key(v) <- s(0, deg(v), deg(v), cc(v), twohop(v))
 3  H <- max-heap of (key(v), -v) for all v                               -- lowest id wins ties
 4  while H not empty:
 5      (k, v) <- pop(H)
 6      if v coloured or k != key(v): continue                            -- lazy deletion
 7      c(v) <- min{ j >= 0 : j not in C(v) }                             -- first fit
 8      for u in N(v) with u uncoloured:
 9          ucn(u) <- ucn(u) - 1
10          if c(v) not in C(u): C(u) <- C(u) + {c(v)}; sat(u) <- sat(u) + 1
11          key(u) <- s(sat(u), deg(u), ucn(u), cc(u), twohop(u)); push(H, (key(u), -u))
12  return c
```

Validation before admission (Stage 2, revised): the formula text is parsed
into an AST and accepted only if it is a sum of `constant x feature` terms
(no feature products, rand(), max/min, conditionals, other functions or
unknown identifiers): `src/gcpbench/validator.py`. On violation the model is
re-queried (at most 3 times) and the rejection is logged.

DSATUR is the special case s = 10000*sat + deg (the paper's baseline), or the
lexicographic key (sat, deg). Brélaz's original rule is (sat, ucn).

## Algorithm 2: GPU local-maxima colouring (one round per loop iteration)

```
Input : CSR graph, priority p, seed
 1  r(v) <- splitmix64(seed XOR v) / 2^64 for all v                       -- declared random term
 2  while some vertex is uncoloured:                                      -- one round
 3      parallel for uncoloured v:                                        -- score kernel
 4          sat(v) <- |{c(u) : u in N(v), u coloured}|; ucn(v) <- |{u in N(v) uncoloured}|
 5          score(v) <- p(sat, deg, ucn, cc, twohop, r)(v)
 6      parallel for uncoloured v:                                        -- local-maximum kernel
 7          win(v) <- for all uncoloured u in N(v): score(u) < score(v) or (score(u) = score(v) and u > v)
 8      parallel for v with win(v):                                       -- assignment kernel
 9          c(v) <- min{ j : j not in {c(u) : u in N(v), u coloured before this round} }
```

Winners are pairwise non-adjacent (line 7 is a strict total order on each
edge), so line 9 has no write conflicts. At least the globally best
uncoloured vertex wins each round, so R <= n.

## Complexity (verified empirically by `experiments/e_e_runtime.py`)

| Part | Statement |
|---|---|
| deg | O(n + m) |
| cc | O(Σ_v deg(v)²) with neighbour marking (O(m·a(G)) with arboricity-ordered triangle listing) |
| twohop | O(Σ_v Σ_{u∈N(v)} deg(u)) = O(Σ_u deg(u)²) with a marking array |
| Online phase, naive (submitted code) | each step rescores every uncoloured vertex: O(m) work for sat/ucn + O(n) argmax per step ⇒ **O(n(n + m))** |
| Online phase, incremental (Algorithm 1) | each edge triggers at most two updates and two heap pushes ⇒ ≤ n + 2m heap entries ⇒ **O((n + m) log n)**; independent of which features or bilinear static terms the formula uses |
| First-fit colour choice | O(sat(v)) per vertex with the per-vertex colour set ⇒ O(m) total |
| RLF (Leighton, corrected) | O(n²) array scans in our implementation (O(n·m) classical bound) |
| Smallest-last | O(m log n) with a lazy heap (O(n + m) with bucket queues) |
| GPU, per round | O(n_u + m_u) work over uncoloured vertices and incident edges, × ⌈k/64⌉ for the colour bitmasks; depth O(Δ·⌈k/64⌉) per kernel |
| GPU, total | O(R·(n + m)·⌈k/64⌉) work. For random priorities R is O(log n / log log n) in expectation on bounded-degree graphs (Jones and Plassmann). For dynamic saturation-based priorities no such bound is known, so R is reported empirically (`e_f_mean_rounds.csv`). Worst case R = n |
| Memory | CSR O(n + m); O(n) for scores, colours, flags, features; heap O(n + m) |

The submitted paper timed the naive O(n(n + m)) loop for the LLM formulas
and a different implementation for DSATUR. Timing every method through
Algorithm 1 removes that implementation artefact; `e_e_runtime_summary.csv`
reports both settings side by side.
