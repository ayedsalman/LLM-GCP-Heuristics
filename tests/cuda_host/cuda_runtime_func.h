#pragma once
#include <cstddef>
#define __global__
#define __device__
#define __host__
#define __restrict__
typedef int cudaError_t; enum { cudaSuccess=0 }; typedef void* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct dim3d { int x; }; static dim3d blockIdx, blockDim, threadIdx;
struct cudaDeviceProp { char name[256]; };
inline const char* cudaGetErrorString(cudaError_t){return "";}
#include <cstdlib>
#include <cstring>
template<class T> cudaError_t cudaMalloc(T**p, size_t n){*p=(T*)malloc(n?n:1);return 0;}
inline cudaError_t cudaMemcpy(void*d,const void*s,size_t n,cudaMemcpyKind){memcpy(d,s,n);return 0;}
inline cudaError_t cudaMemset(void*d,int v,size_t n){memset(d,v,n);return 0;}
inline cudaError_t cudaFree(void*p){free(p);return 0;}
inline cudaError_t cudaEventCreate(cudaEvent_t*){return 0;}
inline cudaError_t cudaEventRecord(cudaEvent_t){return 0;}
inline cudaError_t cudaEventSynchronize(cudaEvent_t){return 0;}
inline cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t){return 0;}
inline cudaError_t cudaEventDestroy(cudaEvent_t){return 0;}
inline cudaError_t cudaGetLastError(){return 0;}
inline cudaError_t cudaGetDevice(int*){return 0;}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int){return 0;}
inline cudaError_t cudaDriverGetVersion(int*){return 0;}
inline cudaError_t cudaRuntimeGetVersion(int*){return 0;}
inline int __popcll(unsigned long long x){return __builtin_popcountll(x);}
inline int __ffsll(long long x){return __builtin_ffsll(x);}
inline int atomicAdd(int*a,int b){int o=*a;*a+=b;return o;}
inline int atomicMax(int*a,int b){int o=*a;if(b>o)*a=b;return o;}
