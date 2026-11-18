# LLM-GCP-Heuristics

Code, benchmark sets and generated heuristic formulas for **"LLM-Aided
Heuristics for the Graph Coloring Problem"** (AlMubaraki, Ahmad and Salman;
submitted to *Computation*, MDPI).

Version 2 is the **revision infrastructure** that answers the reviewers'
comments (26 comments from three reviewers). It has disjoint validation and
test sets, multi-seed generation, control baselines, strict validation of the
formulas, exact optimality gaps, an equal-footing runtime comparison and a GPU
kernel ablation. `docs/REVISION_MAP.md` maps every item of the revision plan
to code and outputs. `results/revision/SUMMARY.md` gives the results that
could be produced without LLM API calls or a GPU.

## Layout

```
PROTOCOL.md                  frozen protocol (selection rule, margins, seeds, hypotheses)
registry/                    datasets.lock.json (SHA-256 of every instance), protocol_freeze.json,
                             candidates.jsonl (every candidate formula + provenance), models.json
data/generate_graphs.py      VAL / TEST-A / TEST-B / PLANTED sets; DIMACS registration; lock/verify
src/heuristics/              ORIGINAL submitted implementation (unchanged; the reference)
src/gcpbench/
  validator.py               AST validator for the linear-combination constraint (C1)
  formulas.py                Table-6 formulas, compiled with their exact operation order
  greedy.py                  Algorithm 1: incremental greedy, O((n+m) log n), bit-identical to the original
  classical.py               DSATUR variants, IDO, Welsh-Powell, smallest-last, First-Fit, RLF (fixed), RLF_legacy
  fastlinear.py              numba engine for control/tuning searches
  parallel.py                Algorithm 2: exact CPU emulator of the CUDA kernel, Jones-Plassmann, Luby
  bounds.py                  exact chi (CP-SAT), clique lower bounds, TabuCol upper bounds
  stats.py                   paired tests, bootstrap CIs, Holm, TOST, Friedman/Nemenyi, across-seed tests
  datasets.py, evaluate.py, timing.py
  llm/                       prompts (verbatim), clients, parser, registry, pipeline (E-B1, E-B3, E-H), CUDA audit (E-C)
src/cuda/parallel_coloring.cu  revised GPU driver (one kernel, swappable priorities), see src/cuda/README.md
experiments/                 e0_equivalence, e_a_main, e_b2_controls, e_d_bounds, e_e_runtime,
                             e_f_ablation, e_g_characterise, run_llm (API experiments), make_summary
analysis/statistical_analysis.py  original Friedman/Nemenyi CLI (now with computed q_alpha and --paired TOST)
docs/                        ALGORITHMS.md (pseudocode, complexity), INTERVENTION_LOG.md, REVISION_MAP.md
tests/                       pytest suite (includes a host build of the CUDA source checked against the emulator)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. benchmark sets (VAL reproduces the submitted generator byte-for-byte)
python data/generate_graphs.py --all
python data/generate_graphs.py --verify          # hashes must match registry/datasets.lock.json
python scripts/freeze_protocol.py --check        # PROTOCOL.md unchanged since the freeze

# 2. experiments that need neither API keys nor a GPU
python experiments/e0_equivalence.py             # new evaluator == original code
python experiments/e_a_main.py --workers 2       # VAL vs TEST-A, paired/TOST, Friedman table, baselines
python experiments/e_b2_controls.py              # random / handcrafted / CMA-ES controls, perturbation
python experiments/e_d_bounds.py                 # exact chi (small), bounds (medium/large), gaps
python experiments/e_e_runtime.py                # equal-footing runtime, scaling, Pareto
python experiments/e_f_ablation.py               # GPU priority ablation (emulator)
python experiments/e_g_characterise.py           # effective weights, ordering similarity, relatedness
python experiments/make_summary.py               # results/revision/SUMMARY.md

# 3. experiments that need API keys (see registry/models.json)
python experiments/run_llm.py zero-shot --model LLaMA --seeds 0-19
python experiments/run_llm.py refine    --model LLaMA --seeds 0-19 --arms T,G,R,N
python experiments/run_llm.py cuda-audit --model Claude --samples 20    # needs nvcc + GPU

# 4. GPU
nvcc -O3 --fmad=false -std=c++17 src/cuda/parallel_coloring.cu -o parallel_coloring
python scripts/verify_cuda.py --bin ./parallel_coloring

pytest -q
```

Every experiment writes CSVs to `results/revision/` (committed) and caches
per-instance colour counts in `results/cache/` (not committed).

## Defects found in the submitted code (fixed in v2; originals kept for reproduction)

1. **RLF.** `rlf_coloring` grows each colour class by
   `len(N(v) & current_class)`. This key is always 0 for every candidate, so
   the choice falls back to set-iteration order. It is kept as `RLF_legacy`;
   `RLF` implements Leighton's rule.
2. **CUDA kernel.** Stale `is_local_max` flags recoloured vertices in later
   rounds, racing with adjacent vertices coloured in the same round. Colours
   were capped at 64. `float` scores lost the degree term of `1e6*sat + deg`
   once sat ≥ 17. The timed loop copied the full colour array every round.
3. **Validation.** The linear-combination constraint was not enforced; the
   headline DeepSeek H1 (`0.6 sat + 0.4 deg·cc`) is bilinear.
4. **Runtime comparison.** The LLM formulas ran through an O(n(n+m)) rescoring
   loop, while DSATUR is cheap in any priority-queue implementation.

## Citation

```bibtex
@article{almubaraki2026llmgcp,
  title   = {LLM-Aided Heuristics for the Graph Coloring Problem},
  author  = {AlMubaraki, Zenah Khaled and Ahmad, Imtiaz and Salman, Ayed},
  journal = {Computation},
  year    = {2026},
  note    = {under revision}
}
```

## License

MIT, see [LICENSE](LICENSE).
