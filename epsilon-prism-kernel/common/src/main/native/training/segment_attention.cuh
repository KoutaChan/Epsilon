#pragma once

#include "gradients.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kFeatures = 16;
constexpr int kShards = kTableGradientShards;

__device__ int codeRow(const SegmentAttention& d, const uint8_t* codes, int transition, int key, int segment) {
  const int tiles = d.input.tiles;
  const int keys = d.input.keys;
  const uint8_t* row = codes + static_cast<size_t>(transition) * d.input.layout.codeWidth;
  if (segment < 0) {
    const int id = row[key];
#if PRISM_CHECKED
    if (id >= d.input.baseCodes) trainingTrap();
#endif
    return id;
  }
  const auto& span = d.input.layout.spans[segment];
  const int id = row[span.column + key - (keys - tiles)];
#if PRISM_CHECKED
  if (id >= span.rows) trainingTrap();
#endif
  return span.firstRow + id;
}

__device__ float entry(const SegmentAttention& d, SegmentAttentionArrays a, int transition, int key, int channel) {
  const int width = d.input.width;
  float offset = a.base[codeRow(d, a.codes, transition, key, -1) * width + channel];
  if (key >= d.input.keys - d.input.tiles) {
#pragma unroll 1
    for (int s = 0; s < d.input.layout.count; ++s) {
      offset = roundedAdd(offset, a.segments[codeRow(d, a.codes, transition, key, s) * width + channel]);
    }
  }
  const int group = a.groups[transition];
#if PRISM_CHECKED
  if (group < 0 || group >= d.rows) trainingTrap();
#endif
  const size_t row = static_cast<size_t>(group) * d.input.keys + key;
  return a.memory[row * width + channel] + offset;
}

// 各ヘッドを特徴の順に積和し、ソフトマックスの分母と値をキーの順で求める。
__global__ void segmentAttentionForward(SegmentAttention d, SegmentAttentionArrays a, float* output, float* probs) {
  __shared__ float weights[kWaves][kMaxCandidateKeys];
  const int wave = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int q = blockIdx.x * kWaves + wave;
  const int transition = q / d.heads;
  if (transition >= d.input.transitions) return;
#if PRISM_CHECKED
  // 逆伝播は行ごとの遷移の区間を二分探索で求めるので、行の番号は減らないこと。
  if (transition > 0 && a.groups[transition - 1] > a.groups[transition]) trainingTrap();
#endif
  const int head = q % d.heads;
  const int channel = head * kFeatures;
  const int keys = d.input.keys;
  const float* query = a.query + static_cast<size_t>(q) * kFeatures;
  float maximum = -INFINITY;
  for (int key = lane; key < keys; key += kWave) {
    float score = 0.0f;
    for (int c = 0; c < kFeatures; ++c) {
      score = fusedMultiplyAdd(query[c], entry(d, a, transition, key, channel + c), score);
    }
    weights[wave][key] = score * d.scale;
    maximum = fmaxf(maximum, weights[wave][key]);
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    maximum = fmaxf(maximum, shuffleDown(maximum, step));
  }
  maximum = broadcastFirst(maximum);
  float sum = 0.0f;
  for (int key = lane; key < keys; key += kWave) {
    const float weight = expf(weights[wave][key] - maximum);
    weights[wave][key] = weight;
    sum += weight;
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    sum += shuffleDown(sum, step);
  }
  const float inverse = 1.0f / broadcastFirst(sum);
  for (int key = lane; key < keys; key += kWave) {
    probs[static_cast<size_t>(q) * keys + key] = weights[wave][key] * inverse;
  }
  waveArrive();
  if (lane < kFeatures) {
    float result = 0.0f;
    for (int key = 0; key < keys; ++key) {
      const float value = entry(d, a, transition, key, d.input.width / 2 + channel + lane);
      result = fusedMultiplyAdd(weights[wave][key], value, result);
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
      gradient = fusedMultiplyAdd(g[c], entry(d, a, transition, key, d.input.width / 2 + channel + c), gradient);
    }
    gradients[wave][key] = gradient;
    weighted = fusedMultiplyAdd(probs[start + key], gradient, weighted);
  }
  for (int step = kWave / 2; step > 0; step /= 2) {
    weighted += shuffleDown(weighted, step);
  }
  weighted = broadcastFirst(weighted);
  for (int key = lane; key < keys; key += kWave) {
    const float gradient = training::scoreGradient(probs[start + key], gradients[wave][key], weighted);
    gradients[wave][key] = gradient;
    scoreGradient[start + key] = gradient;
  }
  waveArrive();
  if (lane < kFeatures) {
    float gradient = 0.0f;
    for (int key = 0; key < keys; ++key) {
      gradient = fusedMultiplyAdd(gradients[wave][key], entry(d, a, transition, key, channel + lane), gradient);
    }
    queryGradient[static_cast<size_t>(q) * kFeatures + lane] = gradient * d.scale;
  }
}

// 遷移 t・キー key の列 c の勾配の項を、積の 2 つの因子で返す。
__device__ float2 contributionFactors(const SegmentAttention& d, SegmentAttentionArrays a, const float* probs,
                                      const float* scoreGradient, const float* upstream, int t, int key, int c) {
  const int half = d.input.width / 2;
  const int feature = c % half;
  const size_t q = static_cast<size_t>(t) * d.heads + feature / kFeatures;
  const size_t at = q * d.input.keys + key;
  const size_t vector = static_cast<size_t>(t) * half + feature;
  return c < half ? make_float2(d.scale * scoreGradient[at], a.query[vector])
                  : make_float2(probs[at], upstream[vector]);
}

// 行番号順に並んだ遷移から、その行の先頭を求める。
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
  shareGroupWrites();
  for (int c = threadIdx.x; c < d.input.width; c += kThreads) {
    float sum = 0.0f;
    for (int t = first; t < last; ++t) {
      const float2 factors = contributionFactors(d, a, probs, score, upstream, t, key, c);
      sum = fusedMultiplyAdd(factors.x, factors.y, sum);
    }
    output[static_cast<size_t>(blockIdx.x) * d.input.width + c] = sum;
  }
}

// 表の各行に属する出現位置（遷移とキーの組）を、行ごとの区間に並べる。数える回では行ごとの個数を cursor に足し、並べる回では cursor から
// ブロックの区間を予約して、ブロックの中の予約の順に書く。区間の中の並びは原子的な予約の到着順で決まる。
template <bool Scatter>
__global__ void segmentAttentionOccurrences(SegmentAttention d, const uint8_t* codes, int32_t* cursor,
                                            const int32_t* offsets, int32_t* occurrences, int buckets) {
  extern __shared__ int32_t local[];
  int32_t* reservation = local + buckets;
  for (int row = threadIdx.x; row < buckets; row += kThreads) local[row] = 0;
  shareGroupWrites();
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
  shareGroupWrites();
  for (int row = threadIdx.x; row < buckets; row += kThreads) {
    if (local[row] == 0) continue;
    const int start = atomicAdd(cursor + row, local[row]);
    if constexpr (Scatter) reservation[row] = offsets[row] + start;
  }
  if constexpr (Scatter) {
    shareGroupWrites();
    if (pair < d.input.transitions * d.input.keys) {
#pragma unroll
      for (int i = 0; i < kMaxCodesPerPair; ++i) {
        if (i <= end) occurrences[reservation[rows[i]] + slots[i]] = pair;
      }
    }
  }
}

// 数えた個数から各行の区間の先頭を求め、並べる回のために cursor を 0 に戻す。
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
  const int row = blockIdx.x;
  const int shard = blockIdx.y;
  const int begin = offsets[row];
  const int length = offsets[row + 1] - begin;
  // 出現列を連続する区間に分け、余りを先頭の区間から配る。
  const int shardSize = length / kShards;
  const int extra = length % kShards;
  const int first = begin + shard * shardSize + min(shard, extra);
  const int last = first + shardSize + (shard < extra);
  for (int c = threadIdx.x; c < width; c += kThreads) {
    partial[(row * kShards + shard) * width + c] = training::tablePartial<width, kFeatures>(
        keys, c, d.scale, a.query, upstream, score, probs, occurrences, first, last);
  }
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
}  // namespace
}  // namespace prism
