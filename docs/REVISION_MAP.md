# Revision plan → code and outputs

Status key: **done** = implemented and run here; **ready** = implemented and
tested (mock/emulator), needs API keys, a GPU or a longer budget;
**authors** = needs the authors' records or writing.

| Plan item | Comment(s) | Code | Output (results/revision/) | Status |
|---|---|---|---|---|
| S0.1 fast evaluator, tie-break | all | `gcpbench/greedy.py` | `e0_equivalence_summary.csv` | done |
| S0.2 dataset registry + lock | R2.1, R3.4, R1.1 | `gcpbench/datasets.py`, `data/generate_graphs.py`, `registry/datasets.lock.json` | manifests in `data/*/manifest.csv` | done (DIMACS: authors download) |
| S0.3 candidate registry | R2.2, R2.3 | `gcpbench/llm/registry.py`, `registry/candidates.jsonl` | — | ready; 16 Table-6 formulas seeded; 25 unprinted candidates: authors |
| S0.4 statistics library | R1.7, R3.2 | `gcpbench/stats.py` | — | done |
| S0.5 protocol freeze | all | `PROTOCOL.md`, `scripts/freeze_protocol.py` | `registry/protocol_freeze.json` | done |
| S0.6 timing hygiene | R1.11 | `gcpbench/timing.py` | `e_e_platform.json` | done (CPU); GPU: authors |
| A1–A3, A6–A8 VAL/TEST, frozen rule | R2.1, R3.4 | `e_a_main.py`, `PROTOCOL.md` §2 | `paired_vs_dsatur.csv`, `test_table_friedman.csv` | done for the Table-6 formulas |
| A4 all 37 candidates on TEST | R2.1 | `scripts/seed_registry.py --add` | — | authors (25 formulas not in paper) |
| A5 selection optimism | R2.1, R3.4 | `e_a_main.py` | `e_a_optimism.csv` | done |
| B1 20-seed zero-shot | R2.2, R3.8 | `llm/pipeline.py`, `run_llm.py zero-shot` | `results/llm/e_b1_zero_shot.csv` | ready |
| B2 random/handcrafted controls | R2.2, R3.5 | `e_b2_controls.py` | `e_b2_summary.csv`, `e_b2_llm_vs_random.csv` | done |
| B3 tuned upper bound | R3.5, R1.2 | `e_b2_controls.py` (CMA-ES) | `e_b2_arms.csv` | done at reduced budget (see `e_b2_budget.json`) |
| B4/B5 uniform refinement + ablation arms | R1.3, R3.8 | `llm/pipeline.refine`, `run_llm.py refine` | `results/llm/e_b3_refine.csv` | ready |
| B6 TOST instead of "indistinguishable" | R3.2 | `stats.tost_paired` | `paired_vs_dsatur.csv` | done |
| B7 snapshots, dates | R2.2 | `llm/clients.py`, `registry/models.json` | registry fields | ready |
| B8 automated structural explanation | R1.2 | `llm/explain.py` | — | ready |
| C1 AST validator + re-query | R2.3, R3.6 | `validator.py`, `llm/pipeline.query` | `strict_linearity_table6.json` | done |
| C2 three-tier CUDA reporting | R2.3, R3.6, R1.9 | `llm/cuda_audit.py`, `run_llm.py cuda-audit` | `results/llm/e_c_cuda_summary.csv` | ready (needs nvcc + GPU) |
| C3 intervention log | R1.9 | `docs/INTERVENTION_LOG.md` | — | authors |
| C4 strictly linear subset | R2.3 | `e_a_main.py` | `paired_vs_dsatur.csv` (DeepSeek_H1A row) | done |
| C5 rand() declared + seeded | R2.3 | `parallel.py`, `parallel_coloring.cu` | `e_f_rand_seed_variance.csv` | done |
| C6 GenAI statement | R1.9 | — | — | authors (writing) |
| D1 DIMACS/COLOR02 | R2.4, R3.3 | `datasets.register_dimacs`, `DIMACS_INSTANCES` | — | ready (download blocked here; bounds must be cited) |
| D2 planted-chi graphs | R2.4 | `datasets.planted_graph` | PLANTED set | done |
| D3 exact chi (small) | R2.4 | `bounds.exact_chi`, `e_d_bounds.py` | `bounds.csv`, `gaps_by_family_scale.csv` | done |
| D4 LB/UB medium, large | R2.4 | `bounds.clique_lb`, `bounds.tabucol_ub` | `bounds.csv` | done at reduced budget; full budget: authors |
| D5 per-family gaps | R2.4 | `e_d_bounds.gaps` | `gaps_by_family_scale.csv` | done |
| D6 balanced TEST-B | R1.1 | `datasets.TESTB_*` | TEST-B set, `seq_means_by_set_scale_family.csv` | done |
| D7 every test in one table | R1.7 | `stats.test_table_row` | `test_table_friedman.csv` | done |
| D8 large-scale narrative | R1.5 | `e_a_main.py` | `seq_means_by_set_scale_family.csv` | done (numbers) / authors (text) |
| D9 complete runtime columns | R1.11 | `e_e_runtime.py` | `e_e_runtime_summary.csv` | done (CPU) |
| E1 equal-footing implementation | R1.11, R3.2 | `greedy.py`, `classical.py` | `e0_equivalence_summary.csv` | done |
| E2 timing campaign | R1.11 | `e_e_runtime.py` | `e_e_runtime_summary.csv` | done at reduced reps |
| E3 pseudocode | R1.6 | `docs/ALGORITHMS.md` | — | done |
| E4 complexity + log-log fits | R1.6 | `docs/ALGORITHMS.md`, `e_e_runtime.py` | `e_e_scaling_fits.csv` | done |
| E5 Pareto plot | R1.11, R2.5 | `e_e_runtime.pareto` | `fig_pareto_colors_runtime.png`, `e_e_pareto.csv` | done |
| E6/E7 GPU kernel/e2e time, rounds | R3.7 | `parallel_coloring.cu`, `parallel.py` | `e_f_mean_rounds.csv` | rounds done; GPU timing: authors |
| F1 per-instance identity | R1.8, R3.7 | `e_f_ablation.py` | `e_f_identity_matrix.csv` | done (emulator) |
| F2 kernel ablation + negative controls | R1.8, R3.7 | `e_f_ablation.py` | `e_f_mean_colors.csv` | done (emulator) |
| F3 harder instances | R3.7 | `e_f_ablation.py` (dense ER, PLANTED) | `e_f_mean_colors.csv` | done (emulator) |
| F4 diff the four kernels | R3.7 | `scripts/verify_cuda.py` | — | authors (original kernels not in repo) |
| G1 effective weights | R1.10 | `e_g_characterise.py` | `g1_effective_weights.csv` | done |
| G2 ordering similarity | R1.10, R1.4 | `e_g_characterise.py` | `g2_ordering_similarity.csv` | done |
| G3 relatedness | R1.2 | `e_g_characterise.py` | `fig_g3_relatedness.png` | done for Table-6 lineages |
| G4 rationale table | R1.10 | parser keeps `rationale` per candidate | registry | ready (needs B1 run) |
| G5 perturbation / headroom | R1.2 | `e_b2_controls.py` | `g5_perturbation.csv` | done |
| G6 improved classical baselines | R1.4 | `classical.py` | all tables | done |
| G7 DSATUR as a point of the family | R1.4 | `e_g_characterise.py` | `g7_dsatur_direction.csv` | done |
| G8 feedback influence | R1.2 | `pipeline.refine` (`shown` + per-instance cache) | — | ready |
| H1–H6 reframing, length | R1.12, R3.1, R3.9, R2.5 | `results/revision/SUMMARY.md` (evidence) | — | authors (writing); E-H code ready |
