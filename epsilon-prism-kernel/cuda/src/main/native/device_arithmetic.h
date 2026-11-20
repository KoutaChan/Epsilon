#pragma once

#include <cuda_runtime.h>

namespace prism {

// a·b + c を 1 回だけ最近接偶数に丸める。
__device__ __forceinline__ float fusedMultiplyAdd(float a, float b, float c) { return __fmaf_rn(a, b, c); }

// 積と和をそれぞれ最近接偶数に丸める（ほかの積や和と 1 つの積和の命令にまとめない）。
__device__ __forceinline__ float roundedMultiply(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float roundedAdd(float a, float b) { return __fadd_rn(a, b); }

__device__ __forceinline__ float inverseSqrt(float x) { return rsqrtf(x); }

}  // namespace prism
