"""Verify the compiled CUDA driver against the CPU emulator (plan F4, C2).

    nvcc -O3 --fmad=false -std=c++17 src/cuda/parallel_coloring.cu -o parallel_coloring
    python scripts/verify_cuda.py --bin ./parallel_coloring --set-dir data/VAL --limit 20

For every instance and priority, the GPU colouring (--dump) must equal
gcpbench.parallel.color_parallel exactly, with the same number of rounds.
Also usable to check an LLM-generated kernel that accepts the same CLI.
"""
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gcpbench.datasets import load_set  # noqa: E402
from gcpbench.parallel import color_parallel, PRIORITIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--set-dir", default="data/VAL")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    bad = tot = 0
    dump = os.path.join(tempfile.mkdtemp(), "c.txt")
    for i, (row, g) in enumerate(load_set(a.set_dir)):
        if i >= a.limit:
            break
        path = os.path.join(a.set_dir, row["instance_id"])
        for p in PRIORITIES:
            for s in map(int, a.seeds.split(",")):
                out = subprocess.run([a.bin, path, "--priority", p, "--seed", str(s), "--reps", "1",
                                      "--warmup", "0", "--dump", dump], capture_output=True,
                                     text=True, errors="replace")
                gpu = [int(x) for x in open(dump).read().split()]
                f = out.stdout.strip().split(",")
                cpu, R = color_parallel(g, p, seed=s)
                tot += 1
                if gpu != cpu or int(f[6]) != R or f[7] != "1":
                    bad += 1
                    print(f"MISMATCH {row['instance_id']} {p} seed={s}: gpu colors={f[5]} rounds={f[6]} "
                          f"valid={f[7]}; emulator rounds={R}")
    print(f"{tot - bad}/{tot} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
