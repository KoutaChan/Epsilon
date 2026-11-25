#pragma once

#include "prism_hip_common.h"
#include "inference_runtime.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "Inference tiles require a 32-lane wave"
#endif

namespace prism::device {

using Buffer = BufferResource;
using U16x4 = uint16_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void scheduleLoads() { scheduleBoundary(); }
// 先読みしたワードを、射影が終わるまでレジスタに保持する。
__device__ __forceinline__ void keepPrefetched(uint32_t a, uint32_t b, uint32_t c) { retainPrefetch({a, b, c}); }

template <class T>
__device__ __forceinline__ void recomputeAddress(T& value) {
  opaque(value);
}
__device__ __forceinline__ int firstLane(int value) { return uniform(value); }
__device__ __forceinline__ uint32_t laneWord(uint32_t value, int lane) { return readLane(value, lane); }
__device__ __forceinline__ unsigned ballot(bool pred) { return vote(pred); }
__device__ __forceinline__ unsigned ballot(unsigned members, bool p) { return vote(p, members); }
__device__ __forceinline__ float exponent2(float value) { return fastExp2(value); }
__device__ __forceinline__ uint32_t upperPair(uint32_t a, uint32_t b) { return upperHalves(a, b); }
__device__ __forceinline__ uint32_t lowerHalfWord(uint32_t x) { return __builtin_amdgcn_ds_swizzle(x, 0x000F); }
__device__ __forceinline__ uint32_t upperHalfWord(uint32_t x) { return __builtin_amdgcn_ds_swizzle(x, 0x021F); }
__device__ __forceinline__ u32x4 loadWords(Buffer resource, uint32_t offset, int uniform) {
  return bufferLoad16(resource, offset, uniform);
}
__device__ __forceinline__ uint32_t loadWord(Buffer resource, uint32_t offset, int uniform) {
  return bufferLoad4(resource, offset, uniform);
}
__device__ __forceinline__ float dotPair(uint32_t a, uint32_t b, float acc) {
  return ::prism::dotPair(asPair(a), asPair(b), acc);
}
__device__ __forceinline__ bf16x16 joinWords(u32x4 lo, u32x4 hi) {
  return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}
__device__ __forceinline__ void sharedBarrier() { shareLocalWrites(); }
__device__ __forceinline__ unsigned long long loadTag(const unsigned long long* p) {
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}
template <class T, class U>
__device__ __forceinline__ T bits(U value) {
  return __builtin_bit_cast(T, value);
}

template <class T>
__device__ __forceinline__ T shuffleXor(T value, int offset) {
  return __shfl_xor(value, offset);
}
template <class T>
__device__ __forceinline__ T shuffle(T value, int lane) {
  return __shfl(value, lane);
}

// 読み元を含む参加レーンがこの位置にそろうと、wave32 は同じ命令で読み元の値を配る。
template <class T>
__device__ __forceinline__ T shuffle(unsigned /*members*/, T value, int source) {
  return shuffle(value, source);
}

}
