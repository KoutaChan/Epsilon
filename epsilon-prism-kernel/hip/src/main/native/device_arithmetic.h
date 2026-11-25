#pragma once

#include <hip/hip_runtime.h>

namespace prism {

// 単精度の近似命令を使う。正規化と指数の丸め位置は呼び出し側で決める。
__device__ __forceinline__ float fastExp2(float value) { return __builtin_amdgcn_exp2f(value); }
__device__ __forceinline__ float fastReciprocal(float value) { return __builtin_amdgcn_rcpf(value); }
__device__ __forceinline__ float inverseSqrt(float value) { return rsqrtf(value); }

// a·b + c を 1 回だけ最近接偶数に丸める（v_fma_f32）。
__device__ __forceinline__ float fusedMultiplyAdd(float a, float b, float c) { return fmaf(a, b, c); }

// 積と和をそれぞれ最近接偶数に丸める。コンパイラーがほかの積や和と 1 つの積和の命令にまとめることはない（ビルドの設定に依らない）。
__device__ __forceinline__ float roundedMultiply(float a, float b) {
#pragma clang fp contract(off)
  return a * b;
}
__device__ __forceinline__ float roundedAdd(float a, float b) {
#pragma clang fp contract(off)
  return a + b;
}

}  // namespace prism
