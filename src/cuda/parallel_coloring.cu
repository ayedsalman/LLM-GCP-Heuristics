/*
 * parallel_coloring.cu  --  revised GPU local-maxima colouring driver
 * (revision plan C5, E6, E7, F1-F4; Algorithm 2 of the revised manuscript)
 *
 * One fixed kernel, swappable priority functions, so the effect of the
 * priority can be measured in isolation (experiment E-F). The colouring
 * produced is bit-identical to the CPU emulator src/gcpbench/parallel.py,
 * which is the reference used to verify this file (--dump + compare).
 *
 * Round structure
 *   score kernel   : one thread per uncoloured v computes sat, ucn and the
 *                    priority score (double precision);
 *   local maxima   : v wins if every uncoloured neighbour u has
 *                    score(u) < score(v), or equal score and u > v;
 *   assign kernel  : winners take the smallest colour unused by neighbours
 *                    coloured in earlier rounds (winners are independent).
 *
 * Fixes relative to the submitted repository version
 *   1. is_local_max is cleared for already-coloured vertices; previously a
 *      stale flag made coloured vertices recolour in later rounds, racing
 *      with an adjacent vertex coloured in the same round.
 *   2. No 64-colour cap: saturation and first-fit use 64-bit chunk masks up
 *      to the current number of colours.
 *   3. Scores are double: in float, 1e6*sat + deg loses the degree term once
 *      sat >= 17 (2^24 / 1e6).
 *   4. The timed loop no longer copies the whole colour array to the host
 *      every round; a device counter (4 bytes) is read instead.
 *   5. rand() is a declared, seeded per-vertex term r(v) = splitmix64(seed^v)
 *      mapped to [0,1), fixed for the run (plan C5).
 *   6. Reads real graphs (.edgelist from data/generate_graphs.py, or DIMACS .col),
 *      verifies the colouring on the host, reports rounds, kernel-only and
 *      end-to-end time over repetitions, and the device/driver/runtime.
 *
 * Build (no FMA contraction, so double results match the CPU emulator):
 *   nvcc -O3 --fmad=false -std=c++17 parallel_coloring.cu -o parallel_coloring
 *
 * Usage
 *   ./parallel_coloring GRAPH --priority NAME [--seed S] [--reps 30] [--warmup 3]
 *                       [--dump colors.txt] [--coef ws,wd,wu,wc,wt,wr]
 *   NAME in: random degree neg_ucn sat sat1e6_deg GPT4o GPT51 Claude DeepSeek
 *            neg_deg neg_sat linear
 *   Output: one CSV line
 *   graph,priority,seed,n,m,colors,rounds,valid,kernel_ms_median,kernel_ms_q1,
 *   kernel_ms_q3,e2e_ms_median,device,driver,runtime
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define UNCOLORED (-1)

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err__ = (x);                                                   \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),   \
              __FILE__, __LINE__);                                             \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

// Must match PRIORITY_IDS in src/gcpbench/parallel.py (dict insertion order).
enum Priority {
  P_RANDOM = 0, P_DEGREE, P_NEG_UCN, P_SAT, P_SAT1E6_DEG, P_GPT4O, P_GPT51,
  P_CLAUDE, P_DEEPSEEK, P_NEG_DEG, P_NEG_SAT, P_LINEAR
};

static const char *PRIORITY_NAMES[] = {
  "random", "degree", "neg_ucn", "sat", "sat1e6_deg", "GPT4o", "GPT51",
  "Claude", "DeepSeek", "neg_deg", "neg_sat", "linear"};

struct Coef { double ws, wd, wu, wc, wt, wr; };

// ------------------------------------------------------------------ hashing
__host__ __device__ inline double splitmix_unit(uint64_t seed, uint64_t v) {
  uint64_t z = (seed ^ v) + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

// Operation order identical to PRIORITIES in parallel.py (compile with --fmad=false).
__device__ inline double device_score(int p, double s, double d, double u, double c,
                                      double t, double r, Coef k) {
  switch (p) {
    case P_RANDOM:     return r;
    case P_DEGREE:     return d;
    case P_NEG_UCN:    return -u;
    case P_SAT:        return s;
    case P_SAT1E6_DEG: return 1e6 * s + d;
    case P_GPT4O:      return 1e6 * s + r * c + d;
    case P_GPT51:      return 1e6 * s + 1200.0 * u + 17.0 * d;
    case P_CLAUDE:     return -u;
    case P_DEEPSEEK:   return d + r;
    case P_NEG_DEG:    return -d;
    case P_NEG_SAT:    return -s;
    default:           return k.ws * s + k.wd * d + k.wu * u + k.wc * c + k.wt * t + k.wr * r;
  }
}

// ------------------------------------------------------------------ kernels
__global__ void score_kernel(int n, const int *__restrict__ rowp, const int *__restrict__ col,
                             const int *__restrict__ color, const double *__restrict__ cc,
                             const double *__restrict__ twohop, const double *__restrict__ rnd,
                             int ncolors, int prio, Coef k, double *__restrict__ score) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n || color[v] != UNCOLORED) return;
  int b = rowp[v], e = rowp[v + 1];
  int ucn = 0, sat = 0;
  for (int i = b; i < e; ++i) ucn += (color[col[i]] == UNCOLORED);
  for (int base = 0; base < ncolors; base += 64) {
    unsigned long long mask = 0ULL;
    for (int i = b; i < e; ++i) {
      int cu = color[col[i]];
      if (cu >= base && cu < base + 64) mask |= 1ULL << (cu - base);
    }
    sat += __popcll(mask);
  }
  score[v] = device_score(prio, (double)sat, (double)(e - b), (double)ucn, cc[v], twohop[v],
                          rnd[v], k);
}

__global__ void local_max_kernel(int n, const int *__restrict__ rowp, const int *__restrict__ col,
                                 const int *__restrict__ color, const double *__restrict__ score,
                                 unsigned char *__restrict__ is_max) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  if (color[v] != UNCOLORED) { is_max[v] = 0; return; }   // fix 1
  double sv = score[v];
  unsigned char win = 1;
  for (int i = rowp[v]; i < rowp[v + 1] && win; ++i) {
    int u = col[i];
    if (color[u] != UNCOLORED) continue;
    double su = score[u];
    if (su > sv || (su == sv && u < v)) win = 0;
  }
  is_max[v] = win;
}

__global__ void assign_kernel(int n, const int *__restrict__ rowp, const int *__restrict__ col,
                              int *__restrict__ color, const unsigned char *__restrict__ is_max,
                              int ncolors, int *__restrict__ n_colored, int *__restrict__ max_color) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n || !is_max[v]) return;
  int b = rowp[v], e = rowp[v + 1];
  int chosen = -1;
  for (int base = 0; chosen < 0; base += 64) {
    unsigned long long mask = 0ULL;
    if (base < ncolors) {
      for (int i = b; i < e; ++i) {
        int cu = color[col[i]];   // neighbours coloured this round are never adjacent winners
        if (cu >= base && cu < base + 64) mask |= 1ULL << (cu - base);
      }
    }
    if (~mask) chosen = base + __ffsll((long long)~mask) - 1;
  }
  color[v] = chosen;
  atomicAdd(n_colored, 1);
  atomicMax(max_color, chosen);
}

// --------------------------------------------------------------------- host
struct Host {
  int n = 0;
  long long m = 0;
  std::vector<int> rowp, col;
  std::vector<double> cc, twohop;
};

static Host load_graph(const std::string &path) {
  std::ifstream in(path);
  if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  std::vector<std::pair<long long, long long>> edges;
  bool dimacs = path.size() > 4 && path.substr(path.size() - 4) == ".col";
  long long declared_n = -1;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    if (dimacs) {
      std::string t; ss >> t;
      if (t == "p") { std::string fmt; ss >> fmt >> declared_n; }
      else if (t == "e") { long long a, b; ss >> a >> b; edges.push_back({a - 1, b - 1}); }
    } else {
      long long a, b;
      if (ss >> a >> b) edges.push_back({a, b});
    }
  }
  // relabel: ascending original labels -> 0..n-1 (same as gcpbench.graph.Graph)
  std::vector<long long> labels;
  if (dimacs) { for (long long i = 0; i < declared_n; ++i) labels.push_back(i); }
  else {
    for (auto &e : edges) { labels.push_back(e.first); labels.push_back(e.second); }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  }
  std::map<long long, int> idx;
  for (size_t i = 0; i < labels.size(); ++i) idx[labels[i]] = (int)i;
  Host g; g.n = (int)labels.size();
  std::vector<std::vector<int>> adj(g.n);
  for (auto &e : edges) {
    int a = idx[e.first], b = idx[e.second];
    if (a == b) continue;
    adj[a].push_back(b); adj[b].push_back(a);
  }
  g.rowp.assign(g.n + 1, 0);
  for (int v = 0; v < g.n; ++v) {
    std::sort(adj[v].begin(), adj[v].end());
    adj[v].erase(std::unique(adj[v].begin(), adj[v].end()), adj[v].end());
    g.rowp[v + 1] = g.rowp[v] + (int)adj[v].size();
  }
  g.col.reserve(g.rowp[g.n]);
  for (int v = 0; v < g.n; ++v) g.col.insert(g.col.end(), adj[v].begin(), adj[v].end());
  g.m = g.rowp[g.n] / 2;
  // static features exactly as networkx / gcpbench: cc = t / (d (d-1)), t = 2 * triangles
  g.cc.assign(g.n, 0.0); g.twohop.assign(g.n, 0.0);
  std::vector<int> mark(g.n, -1), mark2(g.n, -1);
  for (int v = 0; v < g.n; ++v) {
    for (int i = g.rowp[v]; i < g.rowp[v + 1]; ++i) mark[g.col[i]] = v;
    long long t = 0; int two = 0;
    for (int i = g.rowp[v]; i < g.rowp[v + 1]; ++i) {
      int w = g.col[i];
      for (int j = g.rowp[w]; j < g.rowp[w + 1]; ++j) {
        int x = g.col[j];
        if (mark[x] == v && x != w) ++t;
        if (x != v && mark[x] != v && mark2[x] != v) { mark2[x] = v; ++two; }
      }
    }
    long long d = g.rowp[v + 1] - g.rowp[v];
    g.cc[v] = (t == 0) ? 0.0 : (double)t / (double)(d * (d - 1));
    g.twohop[v] = (double)two;
  }
  return g;
}

struct RunResult { std::vector<int> color; int rounds; float kernel_ms; float e2e_ms; };

static RunResult run_once(const Host &g, int prio, Coef k, uint64_t seed) {
  int n = g.n, bs = 256, gs = (n + bs - 1) / bs;
  std::vector<double> rnd(n);
  for (int v = 0; v < n; ++v) rnd[v] = splitmix_unit(seed, (uint64_t)v);

  cudaEvent_t e0, e1, k0, k1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventRecord(e0));

  int *d_rowp, *d_col, *d_color, *d_cnt, *d_maxc;
  double *d_cc, *d_two, *d_rnd, *d_score;
  unsigned char *d_max;
  CUDA_CHECK(cudaMalloc(&d_rowp, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col, std::max<size_t>(1, g.col.size()) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_color, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cc, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_two, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_rnd, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_score, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_max, n));
  CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_maxc, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_rowp, g.rowp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  if (!g.col.empty())
    CUDA_CHECK(cudaMemcpy(d_col, g.col.data(), g.col.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cc, g.cc.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_two, g.twohop.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rnd, rnd.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_color, 0xFF, n * sizeof(int)));   // all UNCOLORED (-1)
  CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));
  int neg1 = -1;
  CUDA_CHECK(cudaMemcpy(d_maxc, &neg1, sizeof(int), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaEventRecord(k0));
  int colored = 0, maxc = -1, rounds = 0;
  while (colored < n) {
    ++rounds;
    int ncolors = maxc + 1;
    score_kernel<<<gs, bs>>>(n, d_rowp, d_col, d_color, d_cc, d_two, d_rnd, ncolors, prio, k, d_score);
    local_max_kernel<<<gs, bs>>>(n, d_rowp, d_col, d_color, d_score, d_max);
    assign_kernel<<<gs, bs>>>(n, d_rowp, d_col, d_color, d_max, ncolors, d_cnt, d_maxc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&colored, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&maxc, d_maxc, sizeof(int), cudaMemcpyDeviceToHost));
    if (rounds > 10 * n + 10) { fprintf(stderr, "no progress; aborting\n"); exit(3); }
  }
  CUDA_CHECK(cudaEventRecord(k1));

  RunResult res;
  res.color.resize(n);
  CUDA_CHECK(cudaMemcpy(res.color.data(), d_color, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&res.kernel_ms, k0, k1));
  CUDA_CHECK(cudaEventElapsedTime(&res.e2e_ms, e0, e1));
  res.rounds = rounds;

  cudaFree(d_rowp); cudaFree(d_col); cudaFree(d_color); cudaFree(d_cc); cudaFree(d_two);
  cudaFree(d_rnd); cudaFree(d_score); cudaFree(d_max); cudaFree(d_cnt); cudaFree(d_maxc);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(k0); cudaEventDestroy(k1);
  return res;
}

static bool verify(const Host &g, const std::vector<int> &c) {
  for (int v = 0; v < g.n; ++v) {
    if (c[v] < 0) return false;
    for (int i = g.rowp[v]; i < g.rowp[v + 1]; ++i) if (c[g.col[i]] == c[v]) return false;
  }
  return true;
}

static double quantile(std::vector<float> x, double q) {
  std::sort(x.begin(), x.end());
  double pos = q * (x.size() - 1);
  size_t lo = (size_t)pos;
  size_t hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (pos - lo) * (x[hi] - x[lo]);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s GRAPH --priority NAME [--seed S] [--reps R] [--warmup W] "
                    "[--dump FILE] [--coef ws,wd,wu,wc,wt,wr] [--header]\n", argv[0]);
    return 1;
  }
  std::string path = argv[1], pname = "sat1e6_deg", dump;
  uint64_t seed = 0; int reps = 30, warmup = 3; bool header = false;
  Coef k{0, 0, 0, 0, 0, 0};
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--priority") pname = argv[++i];
    else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
    else if (a == "--reps") reps = atoi(argv[++i]);
    else if (a == "--warmup") warmup = atoi(argv[++i]);
    else if (a == "--dump") dump = argv[++i];
    else if (a == "--header") header = true;
    else if (a == "--coef") {
      if (sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf,%lf", &k.ws, &k.wd, &k.wu, &k.wc, &k.wt, &k.wr) != 6) {
        fprintf(stderr, "--coef needs 6 comma-separated numbers\n"); return 1;
      }
    }
  }
  int prio = -1;
  for (int p = 0; p <= P_LINEAR; ++p) if (pname == PRIORITY_NAMES[p]) prio = p;
  if (prio < 0) { fprintf(stderr, "unknown priority %s\n", pname.c_str()); return 1; }

  Host g = load_graph(path);
  cudaDeviceProp prop{}; int dev = 0, drv = 0, rt = 0;
  CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);

  for (int w = 0; w < warmup; ++w) run_once(g, prio, k, seed);
  std::vector<float> kms, ems;
  RunResult last;
  for (int r = 0; r < std::max(1, reps); ++r) {
    last = run_once(g, prio, k, seed);
    kms.push_back(last.kernel_ms); ems.push_back(last.e2e_ms);
  }
  bool ok = verify(g, last.color);
  int ncol = 0; for (int c : last.color) ncol = std::max(ncol, c + 1);
  if (!dump.empty()) {
    FILE *f = fopen(dump.c_str(), "w");
    for (int c : last.color) fprintf(f, "%d\n", c);
    fclose(f);
  }
  if (header)
    printf("graph,priority,seed,n,m,colors,rounds,valid,kernel_ms_median,kernel_ms_q1,kernel_ms_q3,"
           "e2e_ms_median,device,driver,runtime\n");
  printf("%s,%s,%llu,%d,%lld,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%s,%d,%d\n", path.c_str(), pname.c_str(),
         (unsigned long long)seed, g.n, g.m, ncol, last.rounds, ok ? 1 : 0, quantile(kms, 0.5),
         quantile(kms, 0.25), quantile(kms, 0.75), quantile(ems, 0.5), prop.name, drv, rt);
  return ok ? 0 : 4;
}
