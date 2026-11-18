# GPU pipeline (revised)

`parallel_coloring.cu` is one fixed local-maxima kernel with swappable
priority functions (experiment E-F). Its colourings are bit-identical to the
CPU emulator `src/gcpbench/parallel.py`, so the ablation can be run without
a GPU and the GPU run is checked against it.

```bash
nvcc -O3 --fmad=false -std=c++17 parallel_coloring.cu -o parallel_coloring
./parallel_coloring ../../data/VAL/ER_medium_000.edgelist --priority GPT51 --reps 30 --header
python ../../scripts/verify_cuda.py --bin ./parallel_coloring --set-dir ../../data/VAL
```

`--fmad=false` is required: fused multiply-add changes double rounding and
breaks bit-equality with the emulator for the scaled priorities.

Timing hygiene (plan S0.6, E6): the driver prints kernel-only time (events
around the round loop; it includes one 4-byte device-to-host read per round
and launch overhead) and end-to-end time (allocation, host-to-device copies,
loop, colour copy-back), as median and IQR over `--reps` after `--warmup`
runs, with device name, driver and runtime versions. Colab T4 sessions vary:
repeat the campaign in at least two sessions and record `nvidia-smi -q -d CLOCK`.

## Changes from the submitted version

1. `is_local_max` is cleared for coloured vertices. Before, a stale flag made
   vertices coloured in an earlier round re-enter `assign_colors`. They were
   then recoloured, racing with an adjacent vertex coloured in the same round.
2. No 64-colour cap (saturation and first-fit use 64-bit chunk masks).
3. Scores are `double`. In `float`, `1e6*sat + deg` loses the degree term once
   `sat >= 17`.
4. The timed loop reads a 4-byte counter instead of copying the full colour array.
5. `rand()` is a declared, seeded per-vertex term `r(v)` (plan C5).
6. Real graph loading (`.edgelist`, DIMACS `.col`), host verification, rounds
   `R`, repetitions, and device metadata.

The four LLM kernels evaluated in the paper should be archived unchanged under
`src/cuda/llm_kernels/<model>/{original,repaired}.cu` with the diff of every
human correction (plan C3, F4), and diffed against each other.
