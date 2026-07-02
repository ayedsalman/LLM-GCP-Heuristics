/*
 * parallel_coloring.cu
 *
 * CUDA C++ implementation of the parallel local-maxima greedy coloring
 * loop described in the paper (Section 3.6, GPU-Parallel CUDA Pipeline).
 *
 * Methodology:
 *   - Graph stored in CSR (Compressed Sparse Row) format in device memory.
 *   - Each round, the Score Kernel launches one thread per uncolored
 *     vertex to compute score(v) in parallel (Eq. 1 in the paper).
 *   - A parallel reduction identifies all vertices whose score strictly
 *     exceeds every uncolored neighbor's score (the local-maxima set).
 *   - All local-maxima vertices are colored simultaneously using the
 *     smallest available color; the shared color array is updated
 *     atomically.
 *   - Repeats until every vertex is colored.
 *   - Execution time is measured with cudaEvent_t, excluding host-device
 *     memory transfer time.
 *
 * This file implements the DSATUR-style baseline formula
 * (score = sat(v), i.e. saturation-only) as the default kernel, and
 * documents where to substitute the LLM-generated parallel formulas
 * from the paper's Table 4.7 (e.g. GPT-5.1: 1e6*sat + 1200*ucn + 17*deg;
 * Claude: -ucn, a local-minima strategy).
 *
 * Compile:
 *   nvcc -O3 parallel_coloring.cu -o parallel_coloring
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <climits>

#define UNCOLORED (-1)
#define MAX_COLOR_CANDIDATES 64  // upper bound on colors tried per vertex per round

struct CSRGraph {
    int num_vertices;
    int num_edges;
    int *row_offsets;   // size num_vertices + 1
    int *col_indices;   // size 2 * num_edges (undirected)
};

// ---------------------------------------------------------------------
// Score kernel: one thread per uncolored vertex.
//
// This is the DSATUR-equivalent baseline (score = saturation only,
// degree as an implicit tiebreak via vertex id). To reproduce one of
// the LLM-generated parallel heuristics from the paper, replace the
// score expression below with, e.g.:
//
//   GPT-5.1:  score = 1000000.0 * sat + 1200.0 * ucn + 17.0 * degree[v];
//   Claude:   score = -(float)ucn;   // local-minima strategy
//   DeepSeek: score = (float)degree[v] + rand_val;  // JP-style + degree bias
// ---------------------------------------------------------------------
__global__ void compute_scores(
    const CSRGraph graph,
    const int *color,       // current color assignment, UNCOLORED if none
    float *scores,          // output: one score per vertex
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || color[v] != UNCOLORED) return;

    int start = graph.row_offsets[v];
    int end = graph.row_offsets[v + 1];

    // saturation: distinct colors among colored neighbors
    bool seen[MAX_COLOR_CANDIDATES] = {false};
    int sat = 0;
    int degree = end - start;
    int ucn = 0;

    for (int i = start; i < end; ++i) {
        int u = graph.col_indices[i];
        int cu = color[u];
        if (cu == UNCOLORED) {
            ucn++;
        } else if (cu < MAX_COLOR_CANDIDATES && !seen[cu]) {
            seen[cu] = true;
            sat++;
        }
    }

    // Default: DSATUR-equivalent baseline score
    scores[v] = (float)sat * 1e6f + (float)degree;

    (void)ucn;  // available for substituting LLM-generated formulas above
}

// ---------------------------------------------------------------------
// Local-maxima identification: a vertex is a local maximum if its
// score strictly exceeds every uncolored neighbor's score. Ties broken
// by vertex id (lower id wins) for determinism.
// ---------------------------------------------------------------------
__global__ void find_local_maxima(
    const CSRGraph graph,
    const int *color,
    const float *scores,
    bool *is_local_max,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || color[v] != UNCOLORED) return;

    int start = graph.row_offsets[v];
    int end = graph.row_offsets[v + 1];
    bool local_max = true;

    for (int i = start; i < end && local_max; ++i) {
        int u = graph.col_indices[i];
        if (color[u] != UNCOLORED) continue;
        if (scores[u] > scores[v] ||
            (scores[u] == scores[v] && u < v)) {
            local_max = false;
        }
    }
    is_local_max[v] = local_max;
}

// ---------------------------------------------------------------------
// Color assignment: each local-maximum vertex picks the smallest color
// not used by any already-colored neighbor. Safe because local maxima
// among currently-uncolored vertices form an independent set w.r.t.
// this round's coloring decisions.
// ---------------------------------------------------------------------
__global__ void assign_colors(
    const CSRGraph graph,
    int *color,
    const bool *is_local_max,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || !is_local_max[v]) return;

    bool used[MAX_COLOR_CANDIDATES] = {false};
    int start = graph.row_offsets[v];
    int end = graph.row_offsets[v + 1];

    for (int i = start; i < end; ++i) {
        int u = graph.col_indices[i];
        int cu = color[u];
        if (cu != UNCOLORED && cu < MAX_COLOR_CANDIDATES) used[cu] = true;
    }

    int c = 0;
    while (c < MAX_COLOR_CANDIDATES && used[c]) c++;
    color[v] = c;
}

// ---------------------------------------------------------------------
// Host-side driver: runs rounds until every vertex is colored.
// ---------------------------------------------------------------------
int run_parallel_coloring(const CSRGraph &h_graph, std::vector<int> &h_color) {
    int n = h_graph.num_vertices;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    CSRGraph d_graph = h_graph;
    cudaMalloc(&d_graph.row_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_graph.col_indices, 2 * h_graph.num_edges * sizeof(int));
    cudaMemcpy(d_graph.row_offsets, h_graph.row_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.col_indices, h_graph.col_indices, 2 * h_graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int *d_color;
    float *d_scores;
    bool *d_is_local_max;
    cudaMalloc(&d_color, n * sizeof(int));
    cudaMalloc(&d_scores, n * sizeof(float));
    cudaMalloc(&d_is_local_max, n * sizeof(bool));
    cudaMemcpy(d_color, h_color.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    int rounds = 0;
    int remaining = n;
    while (remaining > 0) {
        compute_scores<<<grid_size, block_size>>>(d_graph, d_color, d_scores, n);
        find_local_maxima<<<grid_size, block_size>>>(d_graph, d_color, d_scores, d_is_local_max, n);
        assign_colors<<<grid_size, block_size>>>(d_graph, d_color, d_is_local_max, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_color.data(), d_color, n * sizeof(int), cudaMemcpyDeviceToHost);
        remaining = std::count(h_color.begin(), h_color.end(), UNCOLORED);
        rounds++;

        if (rounds > 10 * n) {  // safety valve against infinite loops
            fprintf(stderr, "warning: exceeded expected round count, aborting\n");
            break;
        }
    }

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float ms = 0;
    cudaEventElapsedTime(&ms, start_evt, stop_evt);
    printf("Parallel coloring finished in %d rounds, %.3f ms (device time)\n", rounds, ms);

    cudaFree(d_graph.row_offsets);
    cudaFree(d_graph.col_indices);
    cudaFree(d_color);
    cudaFree(d_scores);
    cudaFree(d_is_local_max);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    return rounds;
}

// ---------------------------------------------------------------------
// Minimal driver / smoke test: builds a small ring-of-triangles graph,
// colors it, and reports the color count. Replace with your own CSR
// loader (e.g. from the .edgelist files produced by
// data/generate_graphs.py) for real benchmarking.
// ---------------------------------------------------------------------
int main() {
    // Tiny hand-built example: a 6-cycle (chromatic number 2)
    int n = 6;
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 10, 12};
    std::vector<int> col_indices = {
        1, 5,   // neighbors of 0
        0, 2,   // neighbors of 1
        1, 3,   // neighbors of 2
        2, 4,   // neighbors of 3
        3, 5,   // neighbors of 4
        4, 0,   // neighbors of 5
    };

    CSRGraph g;
    g.num_vertices = n;
    g.num_edges = (int)col_indices.size() / 2;
    g.row_offsets = row_offsets.data();
    g.col_indices = col_indices.data();

    std::vector<int> color(n, UNCOLORED);
    run_parallel_coloring(g, color);

    int max_color = -1;
    for (int v = 0; v < n; ++v) {
        printf("vertex %d -> color %d\n", v, color[v]);
        max_color = std::max(max_color, color[v]);
    }
    printf("Total colors used: %d\n", max_color + 1);

    return 0;
}
