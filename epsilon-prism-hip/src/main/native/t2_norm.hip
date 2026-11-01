// 学習の RMSNorm の順伝播と逆伝播（BF16 の入出力、FP32 の統計・利得・勾配）。GPU の学習で、eager の EncoderLayer・RmsNorm の
// cast → rms_norm（FP32）→ cast の 3 段の代わりに使う。数式は同じ（⌊·⌉ は BF16 への最近接偶数の 1 回の丸め。c10::BFloat16 と同じ）:
//
//   r_i = rsqrt(Σ_j x_ij² / width + ε)、y_ij = ⌊(x_ij·r_i)·g_j⌉
//   x̂_ij = x_ij·r_i、m_i = Σ_j (dy_ij·g_j)·x̂_ij / width、dx_ij = ⌊r_i·(dy_ij·g_j − x̂_ij·m_i)⌉、dg_j = Σ_i dy_ij·x̂_ij
//
// 1 つの波が 1 行を受け持ち、lane l は 8 列の塊 l + 32k（k < kChunks、列 8(l + 32k) < width のものだけ）を 16 バイトで読み書きする。
// 行の和（Σx² と m）は lane の中を塊・列の順に足してから波の xor の木（waveSum）で足すので、行の中で順が決まり、行の値（y・r・dx）は
// 塊の組み方と行の数に依らない。
// 逆伝播の作業群は kNormGainRows 行の塊を受け持ち、波 w が塊の行 w, w + 8, … を順に回して dg の lane の部分和を持つ。波の部分和は LDS で
// 波の順に足し、塊ごとの部分和 [塊, width] に書く（呼び手が塊の順に足す。原子的な加算なしの決まった順の 2 段の和）。
// 検査つきの組み立てでは全ての全体メモリの読み書きを照らす。
#include <cstdint>
#include <stdexcept>
#include <string>

#include "prism_hip_common.h"
#include "prism_hip_training.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = 8;                                        // lane が一度に読み書きする列（BF16 の 16 バイト）
constexpr int kMaxChunks = kNormMaxWidth / (kWave * kChunk);     // lane あたりの塊の上限
static_assert(kMaxChunks * kWave * kChunk == kNormMaxWidth, "the widest row is a whole number of chunks per lane");
static_assert(kNormGainRows % kWaves == 0, "every wave walks the same number of rows of a full gain block");

// 検査つきの組み立ての上限つきの配列（NormForward・NormBackward と同じ並び）。
struct ForwardArgs {
  NormShape shape;
  Span<const uint16_t> input;
  Span<const float> gain;
  Span<uint16_t> out;
  Span<float> inverseRms;
};

struct BackwardArgs {
  NormShape shape;
  Span<const uint16_t> input;
  Span<const float> gain;
  Span<const float> inverseRms;
  Span<const uint16_t> gradOut;
  Span<uint16_t> gradInput;
  Span<float> gradGain;
};

// lane の塊 k の先頭の列。
__device__ __forceinline__ int column(int lane, int k) {
  return kChunk * (lane + kWave * k);
}

// lane の塊 k が行の中にあれば true（width は 8 の倍数なので、塊は丸ごと中か丸ごと外）。
__device__ __forceinline__ bool owns(int lane, int k, int width) {
  return column(lane, k) < width;
}

// 行の 8 列の BF16（16 バイト）を float にする。
__device__ __forceinline__ void load8(Span<const uint16_t> span, size_t at, float (&values)[8], Bounds& bounds) {
  unpack8(*reinterpret_cast<const u32x4*>(span.data + bounds.index(at, kChunk, span.limit)), values);
}

// 8 つの float を BF16 に丸めて行の 8 列に書く。
__device__ __forceinline__ void store8(Span<uint16_t> span, size_t at, const float (&values)[8], Bounds& bounds) {
  *reinterpret_cast<u32x4*>(span.data + bounds.index(at, kChunk, span.limit)) = pack8(values);
}

// 利得の 8 列 [col, col + 8)。
__device__ __forceinline__ void loadGain(Span<const float> gain, int col, float (&values)[8], Bounds& bounds) {
  const size_t at = bounds.index(col, kChunk, gain.limit);
  const f32x4 lo = *reinterpret_cast<const f32x4*>(gain.data + at);
  const f32x4 hi = *reinterpret_cast<const f32x4*>(gain.data + at + 4);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    values[e] = lo[e];
    values[e + 4] = hi[e];
  }
}

// 順伝播（作業群は 8 行、波が 1 行）。
template <int kChunks>
__global__ __launch_bounds__(kThreads) void normForwardKernel(ForwardArgs args) {
  Bounds bounds;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int row = static_cast<int>(blockIdx.x) * kWaves + static_cast<int>(threadIdx.x) / kWave;
  const int width = args.shape.width;
  if (row < args.shape.rows) {
    const size_t first = static_cast<size_t>(row) * width;
    float x[kChunks][kChunk];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        load8(args.input, first + column(lane, k), x[k], bounds);
      } else {
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          x[k][e] = 0.0f;
        }
      }
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        sum += x[k][e] * x[k][e];
      }
    }
    const float r = rsqrtf(waveSum(sum) / static_cast<float>(width) + args.shape.epsilon);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float g[kChunk];
        loadGain(args.gain, column(lane, k), g, bounds);
        float y[kChunk];
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          y[e] = (x[k][e] * r) * g[e];
        }
        store8(args.out, first + column(lane, k), y, bounds);
      }
    }
    if (lane == 0) {
      args.inverseRms.data[bounds.index(row, 1, args.inverseRms.limit)] = r;
    }
  }
  bounds.trap();
}

// 逆伝播（作業群は kNormGainRows 行の塊、波は塊の行を 8 行おきに）。dy·g は m の和と dx で同じ式を 2 回計算する（同じビット）。
template <int kChunks>
__global__ __launch_bounds__(kThreads) void normBackwardKernel(BackwardArgs args) {
  __shared__ float partial[kWaves][kChunks * kWave * kChunk];
  Bounds bounds;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int wave = static_cast<int>(threadIdx.x) / kWave;
  const int width = args.shape.width;
  const int begin = static_cast<int>(blockIdx.x) * kNormGainRows;
  const int end = min(begin + kNormGainRows, args.shape.rows);
  float g[kChunks][kChunk];
  float sums[kChunks][kChunk];
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    if (owns(lane, k, width)) {
      loadGain(args.gain, column(lane, k), g[k], bounds);
    } else {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        g[k][e] = 0.0f;
      }
    }
#pragma unroll
    for (int e = 0; e < kChunk; ++e) {
      sums[k][e] = 0.0f;
    }
  }
  for (int row = begin + wave; row < end; row += kWaves) {
    const size_t first = static_cast<size_t>(row) * width;
    const float r = args.inverseRms.data[bounds.index(row, 1, args.inverseRms.limit)];
    float normalized[kChunks][kChunk];
    float dy[kChunks][kChunk];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float x[kChunk];
        load8(args.input, first + column(lane, k), x, bounds);
        load8(args.gradOut, first + column(lane, k), dy[k], bounds);
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          normalized[k][e] = x[e] * r;
          dot += (dy[k][e] * g[k][e]) * normalized[k][e];
        }
      }
    }
    const float m = waveSum(dot) / static_cast<float>(width);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (owns(lane, k, width)) {
        float dx[kChunk];
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          dx[e] = r * (dy[k][e] * g[k][e] - normalized[k][e] * m);
          sums[k][e] += dy[k][e] * normalized[k][e];
        }
        store8(args.gradInput, first + column(lane, k), dx, bounds);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kChunks; ++k) {
    if (owns(lane, k, width)) {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        partial[wave][column(lane, k) + e] = sums[k][e];
      }
    }
  }
  __syncthreads();
  for (int col = static_cast<int>(threadIdx.x); col < width; col += kThreads) {
    float total = partial[0][col];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) {
      total += partial[w][col];
    }
    args.gradGain.data[bounds.index(static_cast<size_t>(blockIdx.x) * width + col, 1, args.gradGain.limit)] = total;
  }
  bounds.trap();
}

int chunksOf(int width) {
  return (width / kChunk + kWave - 1) / kWave;
}

template <class T>
void requireAligned(const T* data, const char* name) {
  if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
    throw std::invalid_argument(std::string("RMSNorm ") + name + " must be aligned to 16 bytes");
  }
}

}  // namespace

void checkNormShape(const NormShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(shape.rows > 0, "RMSNorm needs rows, got " + std::to_string(shape.rows));
  require(shape.width > 0 && shape.width % kChunk == 0 && shape.width <= kNormMaxWidth,
          "RMSNorm width must be a multiple of 8 in 8.." + std::to_string(kNormMaxWidth) + ", got " +
              std::to_string(shape.width));
  require(shape.epsilon > 0.0f, "RMSNorm epsilon must be positive");
}

void normForward(const NormShape& shape, const NormForward& arrays, hipStream_t stream) {
  checkNormShape(shape);
  requireAligned(arrays.input, "input");
  requireAligned(arrays.gain, "gain");
  requireAligned(arrays.out, "output");
  const size_t elements = static_cast<size_t>(shape.rows) * shape.width;
  ForwardArgs args{};
  args.shape = shape;
  args.input = {arrays.input, elements};
  args.gain = {arrays.gain, static_cast<size_t>(shape.width)};
  args.out = {arrays.out, elements};
  args.inverseRms = {arrays.inverseRms, static_cast<size_t>(shape.rows)};
  const dim3 grid((shape.rows + kWaves - 1) / kWaves);
  switch (chunksOf(shape.width)) {
    case 1:
      hipLaunchKernelGGL(normForwardKernel<1>, grid, dim3(kThreads), 0, stream, args);
      break;
    case 2:
      hipLaunchKernelGGL(normForwardKernel<2>, grid, dim3(kThreads), 0, stream, args);
      break;
    case 3:
      hipLaunchKernelGGL(normForwardKernel<3>, grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      hipLaunchKernelGGL(normForwardKernel<kMaxChunks>, grid, dim3(kThreads), 0, stream, args);
      break;
  }
  PRISM_HIP(hipGetLastError());
}

void normBackward(const NormShape& shape, const NormBackward& arrays, hipStream_t stream) {
  checkNormShape(shape);
  requireAligned(arrays.input, "input");
  requireAligned(arrays.gain, "gain");
  requireAligned(arrays.gradOut, "output gradient");
  requireAligned(arrays.gradInput, "input gradient");
  const size_t elements = static_cast<size_t>(shape.rows) * shape.width;
  const int blocks = (shape.rows + kNormGainRows - 1) / kNormGainRows;
  BackwardArgs args{};
  args.shape = shape;
  args.input = {arrays.input, elements};
  args.gain = {arrays.gain, static_cast<size_t>(shape.width)};
  args.inverseRms = {arrays.inverseRms, static_cast<size_t>(shape.rows)};
  args.gradOut = {arrays.gradOut, elements};
  args.gradInput = {arrays.gradInput, elements};
  args.gradGain = {arrays.gradGain, static_cast<size_t>(blocks) * shape.width};
  switch (chunksOf(shape.width)) {
    case 1:
      hipLaunchKernelGGL(normBackwardKernel<1>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
    case 2:
      hipLaunchKernelGGL(normBackwardKernel<2>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
    case 3:
      hipLaunchKernelGGL(normBackwardKernel<3>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
    default:
      hipLaunchKernelGGL(normBackwardKernel<kMaxChunks>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
