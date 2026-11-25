#pragma once
#include "device_arithmetic.h"
#include "inference_runtime.h"
#include "wave_tile.h"
#include <cuda_bf16.h>
#include <cuda/atomic>

namespace prism {
using cuda_tile::bf16x16;
using cuda_tile::bufferResource;
using cuda_tile::f32x4;
using cuda_tile::f32x8;
using cuda_tile::fragment;
using cuda_tile::halfSum;
using cuda_tile::kWave;
using cuda_tile::loadUnaligned16;
using cuda_tile::mma;
using cuda_tile::swapHalves;
using cuda_tile::u32x4;
using cuda_tile::waveSum;
using cuda_tile::waveSync;

__host__ __device__ inline uint16_t bf16RoundActivation(float x) {
  uint32_t bits = ::cuda::std::bit_cast<uint32_t>(x);
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
__host__ __device__ inline uint16_t bf16Round(float x) { return isnan(x) ? 0x7fc0u : bf16RoundActivation(x); }
__host__ __device__ inline float bf16Float(uint16_t x) { return ::cuda::std::bit_cast<float>(uint32_t(x) << 16); }
__device__ __forceinline__ uint32_t bf16RoundActivationPair(float low, float high) {
  return bf16RoundActivation(low) | (uint32_t(bf16RoundActivation(high)) << 16);
}
__device__ __forceinline__ void unpack8(u32x4 packed, float (&values)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[2 * i] = bf16Float(packed[i]);
    values[2 * i + 1] = bf16Float(packed[i] >> 16);
  }
}
__device__ __forceinline__ u32x4 pack8(const float (&values)[8]) {
  u32x4 words;
#pragma unroll
  for (int i = 0; i < 4; ++i) words[i] = bf16Round(values[2 * i]) | uint32_t(bf16Round(values[2 * i + 1])) << 16;
  return words;
}
__device__ __forceinline__ u32x4 pack8Activation(const float (&values)[8]) {
  u32x4 words;
#pragma unroll
  for (int i = 0; i < 4; ++i) words[i] = bf16RoundActivationPair(values[2 * i], values[2 * i + 1]);
  return words;
}
__device__ __forceinline__ float silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }
__device__ __forceinline__ float siluFast(float x) {
  return __fmul_rn(x,
                   cuda_tile::fastReciprocal(__fadd_rn(1.0f, cuda_tile::fastExp2(__fmul_rn(x, -1.4426950408889634f)))));
}
struct Bounds {
#if PRISM_CHECKED
  bool violated = false;
  __device__ size_t index(size_t at, size_t count, size_t limit) {
    bool valid = at <= limit && count <= limit - at;
    violated |= !valid;
    return valid ? at : 0;
  }
  __device__ void trap() const {
    if (violated) asm volatile("trap;");
  }
#else
  __device__ size_t index(size_t at, size_t, size_t) { return at; }
  __device__ void trap() const {}
#endif
};

namespace device {
__device__ __forceinline__ void keepPrefetched(uint32_t a, uint32_t b, uint32_t c) {
  asm volatile("" : : "r"(a), "r"(b), "r"(c));
}
template <class T>
__device__ __forceinline__ void recomputeAddress(T& value) {
  asm volatile("" : "+r"(value));
}
using Buffer = cuda_tile::BufferResource;
using U16x4 = cuda_tile::u16x4;
__device__ __forceinline__ void scheduleLoads() {}
__device__ __forceinline__ int firstLane(int x) { return cuda_tile::uniform(x); }
__device__ __forceinline__ uint32_t laneWord(uint32_t x, int lane) { return cuda_tile::readLane(x, lane); }
__device__ __forceinline__ unsigned ballot(bool p) { return cuda_tile::vote(p); }
__device__ __forceinline__ unsigned ballot(unsigned members, bool p) { return __ballot_sync(members, p); }
template <class T>
__device__ __forceinline__ T shuffle(unsigned members, T value, int sourceLane) {
  return __shfl_sync(members, value, sourceLane);
}
__device__ __forceinline__ float exponent2(float x) { return cuda_tile::fastExp2(x); }
__device__ __forceinline__ uint32_t upperPair(uint32_t x, uint32_t y) { return cuda_tile::upperHalves(x, y); }
__device__ __forceinline__ uint32_t lowerHalfWord(uint32_t x) { return laneWord(x, cuda_tile::tileLane() & 15); }
__device__ __forceinline__ uint32_t upperHalfWord(uint32_t x) { return laneWord(x, cuda_tile::tileLane() | 16); }
__device__ __forceinline__ u32x4 loadWords(Buffer base, uint32_t offset, int displacement) {
  return cuda_tile::bufferLoad16(base, offset, displacement);
}
__device__ __forceinline__ uint32_t loadWord(Buffer base, uint32_t offset, int displacement) {
  return cuda_tile::bufferLoad4(base, offset, displacement);
}
using cuda_tile::dotPair;
__device__ __forceinline__ bf16x16 joinWords(u32x4 lo, u32x4 hi) { return cuda_tile::combineHalves(lo, hi); }
__device__ __forceinline__ void sharedBarrier() { __syncthreads(); }
__device__ __forceinline__ unsigned long long loadTag(const unsigned long long* p) {
  return ::cuda::atomic_ref<const unsigned long long, ::cuda::thread_scope_device>(*p).load(
      ::cuda::memory_order_relaxed);
}
template <class T, class U>
__host__ __device__ inline T bits(U x) {
  return ::cuda::std::bit_cast<T>(x);
}
template <class T>
__device__ __forceinline__ T shuffleXor(T x, int i) {
  return __shfl_xor_sync(0xffffffffu, x, i);
}
template <class T>
__device__ __forceinline__ T shuffle(T x, int i) {
  return cuda_tile::readLane(x, i);
}
}
}
