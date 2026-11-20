#pragma once

#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <cstdint>
#include <cstring>

namespace prism::cuda_tile {

constexpr int kWave = 32;
constexpr int kMatrixTile = 16;
template <class T, int N, int Alignment>
struct alignas(Alignment) TileRegisters {
  T elements[N];
  __host__ __device__ T& operator[](int index) { return elements[index]; }
  __host__ __device__ const T& operator[](int index) const { return elements[index]; }
};
using u16x4 = TileRegisters<uint16_t, 4, 8>;
using u32x4 = TileRegisters<uint32_t, 4, 16>;
using f32x4 = TileRegisters<float, 4, 16>;
using f32x8 = TileRegisters<float, 8, 16>;
__device__ __forceinline__ f32x4& operator+=(f32x4& a, const f32x4& b) {
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = __fadd_rn(a[i], b[i]);
  return a;
}
__device__ __forceinline__ f32x4 operator+(f32x4 a, const f32x4& b) { return a += b; }
using bf16x2 = uint32_t;
using bf16x16 = TileRegisters<uint32_t, 8, 16>;

__device__ __forceinline__ int fragmentColumnOfLane(int lane) { return lane % kMatrixTile; }
__device__ __forceinline__ int fragmentParityOfLane(int lane) { return lane / kMatrixTile; }
__device__ __forceinline__ int accumulatorRow(int element, int parity) { return 2 * element + parity; }
__device__ __forceinline__ unsigned tileLane() {
  unsigned lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}
__device__ __forceinline__ int uniform(int value) { return __shfl_sync(0xffffffffu, value, 0); }
template <class T>
__device__ __forceinline__ T readLane(T value, int lane) {
  return __shfl_sync(0xffffffffu, value, lane);
}
template <class T>
__device__ __forceinline__ T swapHalves(T value) {
  return __shfl_xor_sync(0xffffffffu, value, 16);
}
__device__ __forceinline__ uint32_t vote(bool predicate) { return __ballot_sync(0xffffffffu, predicate); }
__device__ __forceinline__ uint32_t upperHalves(uint32_t lo, uint32_t hi) { return __byte_perm(lo, hi, 0x7632); }
__device__ __forceinline__ void waveSync() { __syncwarp(); }
template <int Width>
__device__ __forceinline__ float descendingSum(float value) {
#pragma unroll
  for (int stride = Width / 2; stride; stride /= 2)
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, stride, Width));
  return value;
}
__device__ __forceinline__ float waveSum(float value) { return descendingSum<32>(value); }
__device__ __forceinline__ float halfSum(float value) { return descendingSum<16>(value); }
__device__ __forceinline__ float fastExp2(float value) {
  float result;
  asm("ex2.approx.f32 %0,%1;" : "=f"(result) : "f"(value));
  return result;
}
__device__ __forceinline__ float fastReciprocal(float value) {
  float result;
  asm("rcp.approx.f32 %0,%1;" : "=f"(result) : "f"(value));
  return result;
}
__device__ __forceinline__ float dotPair(bf16x2 a, bf16x2 b, float sum) {
  sum = __fmaf_rn(__uint_as_float(a << 16), __uint_as_float(b << 16), sum);
  return __fmaf_rn(__uint_as_float(a & 0xffff0000u), __uint_as_float(b & 0xffff0000u), sum);
}
__device__ __forceinline__ u32x4 loadUnaligned16(const uint8_t* bytes) {
  u32x4 value;
  memcpy(&value, bytes, sizeof(value));
  return value;
}
struct BufferResource {
  const uint8_t* data;
#if PRISM_CHECKED
  size_t bytes;
#endif
};

template <class T>
__device__ __forceinline__ BufferResource bufferResource(const T* ptr, size_t elements) {
#if PRISM_CHECKED
  if (elements > SIZE_MAX / sizeof(T)) __trap();
  return {reinterpret_cast<const uint8_t*>(ptr), elements * sizeof(T)};
#else
  return {reinterpret_cast<const uint8_t*>(ptr)};
#endif
}

template <class T>
__device__ __forceinline__ T bufferLoad(BufferResource base, uint32_t lane, int displacement) {
  const int64_t offset = int64_t(lane) + displacement;
#if PRISM_CHECKED
  if (offset < 0 || size_t(offset) > base.bytes || sizeof(T) > base.bytes - size_t(offset)) __trap();
#endif
  return *reinterpret_cast<const T*>(base.data + offset);
}

__device__ __forceinline__ u32x4 bufferLoad16(BufferResource base, uint32_t lane, int displacement) {
  return bufferLoad<u32x4>(base, lane, displacement);
}
__device__ __forceinline__ uint32_t bufferLoad4(BufferResource base, uint32_t lane, int displacement) {
  return bufferLoad<uint32_t>(base, lane, displacement);
}
__device__ __forceinline__ bf16x16 combineHalves(u32x4 lo, u32x4 hi) {
  return {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}
__device__ __forceinline__ bf16x16 fragment(const uint16_t* row) {
  auto buffer = bufferResource(row, kMatrixTile);
  return combineHalves(bufferLoad16(buffer, 0, 0), bufferLoad16(buffer, 0, 16));
}
template <class T>
__device__ __forceinline__ T tileElement(const TileRegisters<T, 8, 16>& values, int source, int index) {
  T result{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    T value = __shfl_sync(0xffffffffu, values[i], source);
    if (index == i) result = value;
  }
  return result;
}

// NVIDIA のフラグメント配置で 16 行 8 列の BF16 積を FP32 の累積値に加える。
__device__ __forceinline__ void mma16x8(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                                        float (&accum)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 入力は列または行あたり BF16 が 16 個。出力は列 lane % 16、行 2i + lane / 16 の 8 個。
static __device__ __noinline__ f32x8 mma(const bf16x16& a, const bf16x16& b, const f32x8& c) {
  int lane = tileLane(), group = lane / 4, pair = lane % 4;
  uint32_t a0 = tileElement(a, group, pair), a1 = tileElement(a, group + 8, pair);
  uint32_t a2 = tileElement(a, group, pair + 4), a3 = tileElement(a, group + 8, pair + 4);
  f32x8 native;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint32_t b0 = tileElement(b, group + 8 * half, pair), b1 = tileElement(b, group + 8 * half, pair + 4);
    float accum[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = group + 8 * (i / 2), col = 8 * half + 2 * pair + i % 2;
      accum[i] = tileElement(c, col + 16 * (row % 2), row / 2);
    }
    mma16x8(a0, a1, a2, a3, b0, b1, accum);
#pragma unroll
    for (int i = 0; i < 4; ++i) native[4 * half + i] = accum[i];
  }
  f32x8 result;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int row = accumulatorRow(i, fragmentParityOfLane(lane)), col = fragmentColumnOfLane(lane);
    result[i] = tileElement(native, 4 * (row % 8) + (col % 8) / 2, 4 * (col / 8) + 2 * (row / 8) + col % 2);
  }
  return result;
}
}  // namespace prism::cuda_tile
