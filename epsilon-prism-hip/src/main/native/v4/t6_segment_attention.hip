#include "prism_hip_v4.h"

#include <stdexcept>

namespace prism::v4 {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kFeatures = 16;
constexpr int kShards = 32;

__device__ int codeRow(const SegmentAttention& d, const uint8_t* codes, int transition, int key, int segment) {
  const int tiles = d.input.tiles;
  const int keys = d.input.keys;
  const uint8_t* row = codes + static_cast<size_t>(transition) * d.input.layout.codeWidth;
  if (segment < 0) {
    const int id = row[key];
#if PRISM_HIP_CHECKED
    if (id >= d.input.baseCodes) __builtin_trap();
#endif
    return id;
  }
  const auto& span = d.input.layout.spans[segment];
  const int id = row[span.column + key - (keys - tiles)];
#if PRISM_HIP_CHECKED
  if (id >= span.rows) __builtin_trap();
#endif
  return span.firstRow + id;
}

__device__ float entry(const SegmentAttention& d, SegmentAttentionArrays a, int transition, int key, int channel) {
  const int width = d.input.width;
  float offset = a.base[codeRow(d, a.codes, transition, key, -1) * width + channel];
  if (key >= d.input.keys - d.input.tiles) {
#pragma unroll 1
    for (int s = 0; s < d.input.layout.count; ++s) {
      offset = __fadd_rn(offset, a.segments[codeRow(d, a.codes, transition, key, s) * width + channel]);
    }
  }
  const int group = a.groups[transition];
#if PRISM_HIP_CHECKED
  if (group < 0 || group >= d.rows) __builtin_trap();
#endif
  const size_t row = static_cast<size_t>(group) * d.input.keys + key;
  return a.memory[row * width + channel] + offset;
}

// Match the FP32 candidate attention's feature order, softmax tree and key order.
__global__ void segmentAttentionForward(SegmentAttention d, SegmentAttentionArrays a, float* output, float* probs) {
  __shared__ float weights[kWaves][kMaxCandidateKeys];
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int q = blockIdx.x * kWaves + wave;
  const int transition = q / d.heads;
  if (transition >= d.input.transitions) return;
  const int head = q % d.heads;
  const int channel = head * kFeatures;
  const int keys = d.input.keys;
  const float* query = a.query + static_cast<size_t>(q) * kFeatures;
  float maximum = -INFINITY;
  for (int key = lane; key < keys; key += kWave) {
    float score = 0.0f;
    for (int c = 0; c < kFeatures; ++c) {
      score = fmaf(query[c], entry(d, a, transition, key, channel + c), score);
    }
    weights[wave][key] = score * d.scale;
    maximum = fmaxf(maximum, weights[wave][key]);
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    maximum = fmaxf(maximum, __shfl_down(maximum, step));
  }
  maximum = __shfl(maximum, 0);
  float sum = 0.0f;
  for (int key = lane; key < keys; key += kWave) {
    const float weight = expf(weights[wave][key] - maximum);
    weights[wave][key] = weight;
    sum += weight;
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    sum += __shfl_down(sum, step);
  }
  const float inverse = 1.0f / __shfl(sum, 0);
  for (int key = lane; key < keys; key += kWave) {
    probs[static_cast<size_t>(q) * keys + key] = weights[wave][key] * inverse;
  }
  __builtin_amdgcn_wave_barrier();
  if (lane < kFeatures) {
    float result = 0.0f;
    for (int key = 0; key < keys; ++key) {
      const float value = entry(d, a, transition, key, d.input.width / 2 + channel + lane);
      result = fmaf(weights[wave][key], value, result);
    }
    output[static_cast<size_t>(q) * kFeatures + lane] = result * inverse;
  }
}

__global__ void segmentAttentionQueryGradient(SegmentAttention d, SegmentAttentionArrays a, const float* probs,
                                              const float* upstream, float* queryGradient, float* scoreGradient) {
  __shared__ float gradients[kWaves][kMaxCandidateKeys];
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int q = blockIdx.x * kWaves + wave;
  const int transition = q / d.heads;
  if (transition >= d.input.transitions) return;
  const int channel = q % d.heads * kFeatures;
  const int keys = d.input.keys;
  const size_t start = static_cast<size_t>(q) * keys;
  const float* g = upstream + static_cast<size_t>(q) * kFeatures;
  float weighted = 0.0f;
  for (int key = lane; key < keys; key += kWave) {
    float gradient = 0.0f;
    for (int c = 0; c < kFeatures; ++c) {
      gradient = fmaf(g[c], entry(d, a, transition, key, d.input.width / 2 + channel + c), gradient);
    }
    gradients[wave][key] = gradient;
    weighted = fmaf(probs[start + key], gradient, weighted);
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    weighted += __shfl_down(weighted, step);
  }
  weighted = __shfl(weighted, 0);
  for (int key = lane; key < keys; key += kWave) {
    const float gradient = probs[start + key] * (gradients[wave][key] - weighted);
    gradients[wave][key] = gradient;
    scoreGradient[start + key] = gradient;
  }
  __builtin_amdgcn_wave_barrier();
  if (lane < kFeatures) {
    float gradient = 0.0f;
    for (int key = 0; key < keys; ++key) {
      gradient = fmaf(gradients[wave][key], entry(d, a, transition, key, channel + lane), gradient);
    }
    queryGradient[static_cast<size_t>(q) * kFeatures + lane] = gradient * d.scale;
  }
}

__device__ float contribution(const SegmentAttention& d, SegmentAttentionArrays a, const float* probs,
                              const float* scoreGradient, const float* upstream, int t, int key, int c) {
  const int half = d.input.width / 2;
  const int feature = c % half;
  const size_t q = static_cast<size_t>(t) * d.heads + feature / kFeatures;
  const size_t at = q * d.input.keys + key;
  const size_t vector = static_cast<size_t>(t) * half + feature;
  return c < half ? (d.scale * scoreGradient[at]) * a.query[vector] : probs[at] * upstream[vector];
}

// Upstream transition groups are ordered by the batch row.
__device__ int transitionBound(const SegmentAttention& d, const int32_t* groups, int row) {
  int lo = 0;
  int hi = d.input.transitions;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (groups[mid] < row)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__global__ void segmentAttentionMemoryGradient(SegmentAttention d, SegmentAttentionArrays a, const float* probs,
                                               const float* score, const float* upstream, float* output) {
  __shared__ int first, last;
  const int row = blockIdx.x / d.input.keys;
  const int key = blockIdx.x % d.input.keys;
  if (threadIdx.x == 0) {
    first = transitionBound(d, a.groups, row);
    last = transitionBound(d, a.groups, row + 1);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < d.input.width; c += kThreads) {
    float sum = 0.0f;
    for (int t = first; t < last; ++t) {
      sum += contribution(d, a, probs, score, upstream, t, key, c);
    }
    output[static_cast<size_t>(blockIdx.x) * d.input.width + c] = sum;
  }
}

// CSR occurrence lists. Atomics reserve integer positions, never accumulate FP32 pair gradients.
template <bool Scatter>
__global__ void segmentAttentionOccurrences(SegmentAttention d, const uint8_t* codes, int32_t* cursor,
                                            const int32_t* offsets, int32_t* occurrences, int buckets) {
  extern __shared__ int32_t local[];
  int32_t* reservation = local + buckets;
  for (int row = threadIdx.x; row < buckets; row += kThreads) local[row] = 0;
  __syncthreads();
  const int pair = blockIdx.x * kThreads + threadIdx.x;
  const int t = pair / d.input.keys;
  const int key = pair % d.input.keys;
  const int end = key >= d.input.keys - d.input.tiles ? d.input.layout.count : 0;
  constexpr int kMaxCodesPerPair = kMaxSegmentSpans + 1;
  int rows[kMaxCodesPerPair];
  int slots[kMaxCodesPerPair];
  if (pair < d.input.transitions * d.input.keys) {
#pragma unroll
    for (int i = 0; i < kMaxCodesPerPair; ++i) {
      if (i <= end) {
        const int row = codeRow(d, codes, t, key, i - 1) + (i == 0 ? 0 : d.input.baseCodes);
        const int slot = atomicAdd(local + row, 1);
        if constexpr (Scatter) {
          rows[i] = row;
          slots[i] = slot;
        }
      }
    }
  }
  __syncthreads();
  for (int row = threadIdx.x; row < buckets; row += kThreads) {
    if (local[row] == 0) continue;
    const int start = atomicAdd(cursor + row, local[row]);
    if constexpr (Scatter) reservation[row] = offsets[row] + start;
  }
  if constexpr (Scatter) {
    __syncthreads();
    if (pair < d.input.transitions * d.input.keys) {
#pragma unroll
      for (int i = 0; i < kMaxCodesPerPair; ++i) {
        if (i <= end) occurrences[reservation[rows[i]] + slots[i]] = pair;
      }
    }
  }
}

__global__ void segmentAttentionOffsets(int rows, int32_t* cursor, int32_t* offsets) {
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int row = 0; row < rows; ++row) {
      offsets[row] = sum;
      sum += cursor[row];
      cursor[row] = 0;
    }
    offsets[rows] = sum;
  }
}

template <int Width, int Keys>
__global__ void segmentAttentionPartials(SegmentAttention d, SegmentAttentionArrays a, const float* probs,
                                         const float* score, const float* upstream, const int32_t* offsets,
                                         const int32_t* occurrences, float* partial) {
  constexpr int width = Width;
  const int keys = Keys == 0 ? d.input.keys : Keys;
  constexpr int half = width / 2;
  constexpr int heads = half / kFeatures;
  const int row = blockIdx.x;
  const int shard = blockIdx.y;
  const int begin = offsets[row];
  const int length = offsets[row + 1] - begin;
  // Keep neighbouring pairs in the same shard so they can reuse query and
  // upstream rows in cache. Divide the remainder among the first shards.
  const int small = length / kShards;
  const int extra = length % kShards;
  const int first = begin + shard * small + min(shard, extra);
  const int last = first + small + (shard < extra);
  for (int c = threadIdx.x; c < width; c += kThreads) {
    const bool keyGradient = c < half;
    const int feature = keyGradient ? c : c - half;
    const float* coefficients = (keyGradient ? score : probs) + feature / kFeatures * keys;
    const float* vectors = (keyGradient ? a.query : upstream) + feature;
    float sum = 0.0f;
    for (int i = first; i < last; ++i) {
      const int pair = occurrences[i];
      const int transition = pair / keys;
      const int key = pair % keys;
      float coefficient = coefficients[static_cast<size_t>(transition) * heads * keys + key];
      if (keyGradient) coefficient *= d.scale;
      sum += coefficient * vectors[static_cast<size_t>(transition) * half];
    }
    partial[(row * kShards + shard) * width + c] = sum;
  }
}

template <int Width>
void launchTablePartials(SegmentAttention d, SegmentAttentionArrays a, SegmentAttentionBackward b, int tableRows,
                         hipStream_t stream) {
  const dim3 grid(tableRows, kShards);
  const auto launch = [&](auto keys) {
    hipLaunchKernelGGL((segmentAttentionPartials<Width, decltype(keys)::value>), grid, dim3(kThreads), 0, stream, d, a,
                       b.probs, b.score, b.upstream, b.offsets, b.occurrences, b.partial);
  };
  if (!dispatchCandidateKeys(d.input.keys, launch)) launch(std::integral_constant<int, 0>{});
}

__global__ void segmentAttentionTables(SegmentAttention d, const float* partial, float* base, float* segments) {
  const int row = blockIdx.x;
  const int width = d.input.width;
  for (int c = threadIdx.x; c < width; c += kThreads) {
    float sum = 0.0f;
    for (int s = 0; s < kShards; ++s) sum += partial[(row * kShards + s) * width + c];
    if (row < d.input.baseCodes)
      base[row * width + c] = sum;
    else
      segments[(row - d.input.baseCodes) * width + c] = sum;
  }
}

void check(SegmentAttention d) {
  const auto s = d.input;
  const auto* capability = widthCapability(s.width);
  if (s.transitions <= 0 || s.keys <= 0 || s.keys > kMaxCandidateKeys || s.tiles < 1 || s.seats < 2 || s.seats > 4 ||
      s.keys != 1 + s.tiles + s.seats || s.baseCodes <= 0 || s.baseCodes > kU8Codes ||
      !s.layout.valid(s.keys, s.tiles) || (capability == nullptr || !capability->core) ||
      d.heads * 2 * kFeatures != s.width || d.rows <= 0 || !(d.scale > 0.0f) ||
      static_cast<int64_t>(s.transitions) * s.layout.codeWidth > INT32_MAX) {
    throw std::invalid_argument("invalid fused V4 candidate attention shape");
  }
}

}  // namespace

void launchSegmentAttention(SegmentAttention d, SegmentAttentionArrays a, float* result, float* probs,
                            hipStream_t stream) {
  check(d);
  const int blocks = (d.input.transitions * d.heads + kWaves - 1) / kWaves;
  hipLaunchKernelGGL(segmentAttentionForward, dim3(blocks), dim3(kThreads), 0, stream, d, a, result, probs);
  PRISM_HIP(hipGetLastError());
}

void launchSegmentAttentionBackward(SegmentAttention d, SegmentAttentionArrays a, SegmentAttentionBackward b,
                                    hipStream_t stream) {
  check(d);
  const int queries = (d.input.transitions * d.heads + kWaves - 1) / kWaves;
  const int tableRows = d.input.baseCodes + d.input.layout.tableRows;
  const int pairs = (d.input.transitions * d.input.keys + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(segmentAttentionQueryGradient, dim3(queries), dim3(kThreads), 0, stream, d, a, b.probs, b.upstream,
                     b.query, b.score);
  hipLaunchKernelGGL(segmentAttentionMemoryGradient, dim3(d.rows * d.input.keys), dim3(kThreads), 0, stream, d, a,
                     b.probs, b.score, b.upstream, b.memory);
  PRISM_HIP(hipMemsetAsync(b.cursor, 0, tableRows * sizeof(int32_t), stream));
  const size_t histogramBytes = tableRows * sizeof(int32_t);
  hipLaunchKernelGGL((segmentAttentionOccurrences<false>), dim3(pairs), dim3(kThreads), histogramBytes, stream, d,
                     a.codes, b.cursor, nullptr, nullptr, tableRows);
  hipLaunchKernelGGL(segmentAttentionOffsets, dim3(1), dim3(1), 0, stream, tableRows, b.cursor, b.offsets);
  hipLaunchKernelGGL((segmentAttentionOccurrences<true>), dim3(pairs), dim3(kThreads), 2 * histogramBytes, stream, d,
                     a.codes, b.cursor, b.offsets, b.occurrences, tableRows);
  forCoreWidth(d.input.width,
               [&](auto width) { launchTablePartials<decltype(width)::value>(d, a, b, tableRows, stream); });
  hipLaunchKernelGGL(segmentAttentionTables, dim3(tableRows), dim3(kThreads), 0, stream, d, b.partial, b.base,
                     b.segments);
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism::v4
