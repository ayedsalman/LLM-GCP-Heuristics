# LLM-GCP-Heuristics

Code, benchmark instances, and generated heuristic formulas for
**"Sequential and GPU Parallel Heuristics for the Graph Coloring
Problem"** (AlMubaraki & Ahmad, 2026).

This repository implements the LLM-generated vertex-ordering heuristics
described in the paper, the dual CPU/GPU benchmarking pipeline, and the
statistical significance testing (Friedman test with Nemenyi post-hoc
correction) used throughout Section 4.

## Repository structure

```
.
├── analysis/
│   └── statistical_analysis.py   # Friedman + Nemenyi, CD diagrams (Section 3.9)
├── data/
│   └── generate_graphs.py        # 140-instance ER/BA/WS benchmark suite (Table 4)
├── src/
│   ├── heuristics/
│   │   └── sequential_heuristics.py  # All formulas from the paper's formulas table
│   ├── cuda/
│   │   └── parallel_coloring.cu      # CUDA local-maxima parallel coloring kernel
│   └── run_benchmark.py          # End-to-end sequential benchmark runner
├── results/                      # Output CSVs / CD diagrams land here (gitignored)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate the 140-instance synthetic benchmark suite
python data/generate_graphs.py --out-dir data/graphs --seed 42

# 2. Run all sequential heuristics on one scale
cd src
python run_benchmark.py --graphs-dir ../data/graphs --scale medium \
    --out ../results/seq_medium.csv
cd ..

# 3. Statistical significance testing (Friedman + Nemenyi)
python analysis/statistical_analysis.py \
    --input results/seq_medium.csv --alpha 0.05 --baseline DSATUR \
    --cd-plot results/seq_medium_cd.png
```

This reproduces the methodology of Section 4.1.4 / 4.2.3 of the paper.
Note that `--seed` controls the random graph generation; exact color
counts and rank values will differ slightly instance-by-instance from
the paper's reported numbers (which were generated with the paper's
own seed and, for the parallel results, executed on an NVIDIA T4 GPU),
but the qualitative pattern (DSATUR/LLaMA/Claude/GPT-4o/DeepSeek
clustering near the top, GPT-5.1/RLF/First-Fit forming a weaker tier)
should reproduce, and the Nemenyi critical difference for a given
(*k* algorithms, *N* instances) pair is deterministic and will match
the paper's reported CD exactly regardless of the specific graphs
generated.

## What's implemented vs. illustrative

- **`analysis/statistical_analysis.py`** — fully implemented and
  tested. Verified to reproduce the paper's exact Nemenyi CD values
  (e.g. CD=1.485 for k=8 algorithms, N=50 instances, matching the
  paper's small-scale sequential test) since CD depends only on *k*
  and *N*, not on the specific data.
- **`data/generate_graphs.py`** — fully implemented and tested;
  generates exactly 140 instances matching the paper's Table 4
  parameters (ER/BA/WS × small/medium/large).
- **`src/heuristics/sequential_heuristics.py`** — fully implemented
  and tested; every formula is transcribed directly from the paper's
  formulas table and verified to produce valid colorings.
- **`src/run_benchmark.py`** — fully implemented and tested end-to-end
  against the two files above.
- **`src/cuda/parallel_coloring.cu`** — implements the local-maxima
  parallel coloring methodology described in Section 3.6 (Score
  Kernel → parallel reduction → simultaneous color assignment via
  CSR representation, `cudaEvent_t` timing). The default kernel
  computes the DSATUR-equivalent (saturation-only) score; comments in
  the file show exactly where to substitute the LLM-generated
  formulas (GPT-5.1, Claude, DeepSeek) from the paper's Table 4.7.
  **This file could not be compiled or run in the environment used to
  prepare this repository (no CUDA toolkit / GPU available)** — please
  compile with `nvcc -O3 parallel_coloring.cu -o parallel_coloring` on
  a CUDA-capable machine and verify before relying on it.

## Statistical analysis usage

`analysis/statistical_analysis.py` accepts either:

- **Wide CSV**: one row per graph instance, one column per algorithm,
  values = color count (or any other per-instance metric).
- **Long CSV**: columns `instance_id, algorithm, value`.

```python
from analysis.statistical_analysis import run_friedman_nemenyi, load_results

df = load_results("results/seq_medium.csv")
report = run_friedman_nemenyi(df, alpha=0.05, baseline="DSATUR")
print(report.summary())
```

## Citation

If you use this code, please cite:

```bibtex
@article{almubaraki2026llmgcp,
  title   = {Sequential and GPU Parallel Heuristics for the Graph Coloring Problem},
  author  = {AlMubaraki, Zenah Khaled and Ahmad, Imtiaz},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
