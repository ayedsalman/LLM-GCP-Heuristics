"""Timing hygiene (plan S0.6, E2).

* single-thread BLAS/OpenMP (set before numpy is imported: use
  ``python -X importtime`` free shell env, or ``scripts/pin_env.sh``),
* CPU affinity pinned to one core where the OS allows it,
* warm-up runs (also triggers numba compilation), then >= 30 timed
  repetitions; report median and inter-quartile range,
* platform record stored next to every timing table.

GPU timing hygiene (device, driver, CUDA version, clocks, >= 2 Colab
sessions) is handled by the CUDA driver's CSV output; see src/cuda/README.md.
"""

from __future__ import annotations

import os
import platform
import statistics
import time

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def pin(cpu: int | None = 0) -> dict:
    for k in THREAD_ENV:
        os.environ.setdefault(k, "1")
    pinned = None
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
            pinned = cpu
        except OSError:
            pinned = None
    return {"pinned_cpu": pinned, **{k: os.environ.get(k) for k in THREAD_ENV}}


def time_fn(fn, reps: int = 30, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    xs.sort()
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
    return {"median_s": statistics.median(xs), "q1_s": q[0], "q3_s": q[2],
            "iqr_s": q[2] - q[0], "min_s": xs[0], "reps": reps, "warmup": warmup}


def platform_info() -> dict:
    import networkx
    import numpy
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import numba
        nb = numba.__version__
    except ImportError:
        nb = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "numpy": numpy.__version__, "networkx": networkx.__version__, "numba": nb,
            "cpu": cpu, "machine": platform.machine(), "os": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
