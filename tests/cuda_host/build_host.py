"""Build src/cuda/parallel_coloring.cu as a CPU program (kernels run as loops over
threads) so its logic can be checked against the emulator without a GPU.

    python tests/cuda_host/build_host.py   # -> tests/cuda_host/pc_host

Valid because each kernel only reads state written by earlier kernels, except
assign_kernel, whose writers (local maxima) are pairwise non-adjacent.
This does NOT replace running the real kernel: on a GPU use
scripts/verify_cuda.py, which compares GPU dumps with the emulator.
"""
import os
import re
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "..", "..", "src", "cuda", "parallel_coloring.cu")).read()
src = src.replace("#include <cuda_runtime.h>", '#include "cuda_runtime_func.h"')
src = re.sub(r"(\w+_kernel)<<<gs, bs>>>\((.*?)\);",
             r"for(blockIdx.x=0;blockIdx.x<n;++blockIdx.x){blockDim.x=1;threadIdx.x=0;\1(\2);}",
             src, flags=re.S)
cpp = os.path.join(here, "pc_host.cpp")
open(cpp, "w").write(src)
subprocess.check_call(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-w", cpp,
                       "-o", os.path.join(here, "pc_host")])
print("built", os.path.join(here, "pc_host"))
