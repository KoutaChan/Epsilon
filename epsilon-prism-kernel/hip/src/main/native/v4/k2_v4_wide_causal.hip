#define PRISM_V4_WIDE
#define PRISM_V4_CAUSAL
#include "k2_attention.hip"
#include "prism_hip_mlp.h"
#include "prism_hip_v4.h"

namespace prism {
namespace {

constexpr int kProjectionRows = 32;

// カテゴリ列の行幅は 4 の倍数とは限らない。整列していないワードも memcpy で読む。
__device__ __forceinline__ uint32_t readWord(const uint8_t* p) {
  uint32_t word;
  __builtin_memcpy(&word, p, sizeof(word));
  return word;
}

__device__ __forceinline__ uint32_t hashBytes(const uint8_t* p, int bytes, uint32_t hash) {
  int i = 0;
  for (; i + 4 <= bytes; i += 4) hash = (hash ^ readWord(p + i)) * 16777619u;
  for (; i < bytes; ++i) hash = (hash ^ p[i]) * 16777619u;
  return hash;
}

__device__ __forceinline__ bool sameBytes(const uint8_t* left, const uint8_t* right, int bytes) {
  int i = 0;
  for (; i + 4 <= bytes; i += 4) {
    if (readWord(left + i) != readWord(right + i)) return false;
  }
  for (; i < bytes; ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

struct RowBytes {
  const uint8_t* data;
  int width;
  __device__ const uint8_t* row(int i) const { return data + static_cast<size_t>(i) * width; }
};

struct InputKey {
  RowBytes columns[3];  // 埋め込みが参照するカテゴリ・スカラー・厳密値。省略された列の幅は 0。
};

// 空きの行を取った提出だけが値を書く。現在の提出で確保した行は元の入力を参照できるので、
// キーを書き終える前の別ウェーブとも照合できる。次の提出の読み出しはストリーム順で後になる。
__device__ int cachedInput(InputKey key, int kind, int row, uint32_t hash, v4::ProjectionCache cache, bool& compute) {
  int width = 0;
  for (const auto& col : key.columns) width += col.width;
  if (!cache.rows || width > cache.kKeyBytes) return -1;
  const int slot = hash & (cache.rows - 1);
  const unsigned stamp = cache.generation * kEmbeddingSlots + kind;
  const unsigned long long claim = (static_cast<unsigned long long>(stamp) << 32) | static_cast<unsigned>(row);
  auto* tag = cache.tags + slot;
  auto prior = __atomic_load_n(tag, __ATOMIC_RELAXED);
  if (!prior) prior = atomicCAS(tag, 0ull, claim);
  auto* saved = cache.keys + size_t(slot) * cache.kKeyBytes;
  if (!prior) {
    for (const auto& col : key.columns) {
      for (int i = 0; i < col.width; ++i) saved[i] = col.row(row)[i];
      saved += col.width;
    }
    compute = true;
    return slot;
  }
  const unsigned when = static_cast<unsigned>(prior >> 32);
  if (when % kEmbeddingSlots != kind) return -1;
  for (const auto& col : key.columns) {
    const auto* other = when == stamp ? col.row(static_cast<int>(static_cast<uint32_t>(prior))) : saved;
    if (!sameBytes(col.row(row), other, col.width)) return -1;
    saved += col.width;
  }
  compute = false;
  return slot;
}

// 種類ごとに表を区切り、列が全て一致したときだけ先に登録した代表を使う。
// 衝突が続く場合は自分を代表にするので、挿入順によらず同じ値になる。
__global__ void indexProjectionInputs(InputKey key, int kind, int first, int rows, int* owners, int* sources,
                                      int* table, int* count, v4::ProjectionCache cache) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int token = first + row;
  int owner = token;
  uint32_t hash = (2166136261u ^ kind) * 16777619u;
#pragma unroll
  for (const auto& column : key.columns) hash = hashBytes(column.row(row), column.width, hash);
  hash = (hash ^ (hash >> 16)) * 0x85ebca6bu;
  hash ^= hash >> 13;
  bool compute = true;
  const int page = cachedInput(key, kind, row, hash, cache, compute);
  if (page >= 0) owner = ~page;
  int slot = hash % rows;
  for (int probe = 0; page < 0 && probe < 4; ++probe) {
    const int previous = atomicCAS(table + slot, 0, row + 1) - 1;
    if (previous < 0) break;
    bool equal = true;
#pragma unroll
    for (const auto& column : key.columns) equal &= sameBytes(column.row(row), column.row(previous), column.width);
    if (equal) {
      owner = first + previous;
      compute = false;
      break;
    }
    slot = slot + 1 == rows ? 0 : slot + 1;
  }
  owners[token] = owner;
  const int lane = threadIdx.x % kWave;
  const unsigned mask = __ballot(compute);
  int begin = 0;
  if (lane == 0 && mask != 0) begin = atomicAdd(count, __popc(mask));
  begin = __shfl(begin, 0);
  if (compute) sources[begin + __popc(mask & ((1u << lane) - 1u))] = token;
}

// 代表の正規化済み入力をまとめて LDS に読み、全ヘッドの射影に使う。Q と K/V では元の
// projectHead と同じ左右のオペランドを使う。K の順、バイアスを足す位置、BF16 への丸めもそろえる。
template <int H>
__global__ __launch_bounds__(tile::kThreads) void projectDistinctInputs(AttentionArgs args, const int* sources,
                                                                        const int* count, Span<uint16_t> output,
                                                                        v4::ProjectionIndex index) {
  constexpr int A = H / 2;
  constexpr int parts = kProjectionRows / 16;
  const int first = blockIdx.x * kProjectionRows;
  const int rows = min(kProjectionRows, *count - first);
  if (rows <= 0) return;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
  const int column = tid % 16;
  const int half = (tid % kWave) >> 4;
  __shared__ int source[kProjectionRows], destination[kProjectionRows];
  __shared__ __align__(16) uint16_t input[kProjectionRows][H + tile::kPad];
  Bounds bounds;
  if (tid < kProjectionRows) {
    source[tid] = sources[first + min(tid, rows - 1)];
    destination[tid] = index.owners[source[tid]];
  }
  tile::barrier();
  for (int i = tid; i < kProjectionRows * (H / 8); i += tile::kThreads) {
    const int row = i / (H / 8);
    const int col = (i % (H / 8)) * 8;
    const size_t at = bounds.index(static_cast<size_t>(source[row]) * H + col, 8, args.normalized.limit);
    *reinterpret_cast<u32x4*>(&input[row][col]) = *reinterpret_cast<const u32x4*>(args.normalized.data + at);
  }
  tile::barrier();
  for (int tileId = wave; tileId < 3 * A / kHeadWidth; tileId += tile::kWaves) {
    const int pair = tileId / 6;
    const int head = tileId % 2;
    const int projection = (tileId % 6) / 2;
    const int base = (pair * (H / 16) * kStepBytes + (3 * head + projection) * kFragmentBytes) / sizeof(uint16_t);
    const int extent = ((H / 16 - 1) * kStepBytes + kFragmentBytes) / sizeof(uint16_t);
    const auto weights = bufferResource(args.qkv.data + bounds.index(base, extent, args.qkv.limit));
    f32x8 acc[parts]{};
    bf16x16 weight[3];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      weight[k] = bufferFragment(weights, column * 16, k * kStepBytes, k * kStepBytes + kFragmentBytes / 2);
    }
#pragma unroll
    for (int k = 0; k < H / 16; ++k) {
      if (k + 2 < H / 16) {
        const int at = (k + 2) * kStepBytes;
        weight[(k + 2) % 3] = bufferFragment(weights, column * 16, at, at + kFragmentBytes / 2);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int m = 0; m < parts; ++m) {
        const bf16x16 values = fragment(&input[16 * m + column][k * 16]);
        acc[m] = projection == 0 ? mma(weight[k % 3], values, acc[m]) : mma(values, weight[k % 3], acc[m]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    const int biasColumn = projection * A + pair * kPairWidth + head * kHeadWidth;
    const int outColumn = foldColumn(biasColumn, A);
    if (projection == 0) {
      const size_t at = bounds.index(biasColumn + 8 * half, 8, args.qkvBias.limit);
      const f32x8 bias = *reinterpret_cast<const f32x8*>(args.qkvBias.data + at);
#pragma unroll
      for (int m = 0; m < parts; ++m) {
        const int row = 16 * m + column;
        if (row < rows) {
          float values[8];
#pragma unroll
          for (int v = 0; v < 8; ++v) values[v] = acc[m][v] + bias[v];
          const int dst = destination[row];
          const auto target = dst < 0 ? index.cache.qkv : output;
          const size_t to =
              bounds.index(static_cast<size_t>(dst < 0 ? ~dst : dst) * (3 * A) + outColumn + 8 * half, 8, target.limit);
          *reinterpret_cast<u32x4*>(target.data + to) = pack8Activation(values);
        }
      }
    } else {
      const float bias = args.qkvBias.data[bounds.index(biasColumn + column, 1, args.qkvBias.limit)];
#pragma unroll
      for (int m = 0; m < parts; ++m) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int row = 16 * m + 2 * v + half;
          if (row < rows) {
            const int dst = destination[row];
            const auto target = dst < 0 ? index.cache.qkv : output;
            const size_t to =
                bounds.index(static_cast<size_t>(dst < 0 ? ~dst : dst) * (3 * A) + outColumn + column, 1, target.limit);
            target.data[to] = bf16RoundActivation(acc[m][v] + bias);
          }
        }
      }
    }
  }
  bounds.trap();
}

}  // namespace

void v4::indexLayerInputs(const EmbedArgs& args, const EmbeddingSpec (&specs)[kEmbeddingSlots], ProjectionIndex index,
                          hipStream_t stream) {
  if (index.tokens == 0) return;
  PRISM_HIP(hipMemsetAsync(index.table(), 0, (static_cast<size_t>(index.tokens) + 1) * sizeof(int), stream));
  auto column = [&](int section) -> RowBytes {
    if (section < 0) return {args.input.data, 0};
    return {args.input.data + args.section[section],
            args.sectionWidth[section] * sectionElementBytes(static_cast<Section>(section))};
  };
  int first = 0;
  for (int slot = 0; slot < static_cast<int>(EmbeddingSlot::kTransition); ++slot) {
    const int rows = args.count[slot];
    if (rows == 0) continue;
    const auto& spec = specs[slot];
    const InputKey key{{column(spec.category), column(spec.scalar), column(spec.exact)}};
    const int blocks = (rows + tile::kThreads - 1) / tile::kThreads;
    hipLaunchKernelGGL(indexProjectionInputs, dim3(blocks), dim3(tile::kThreads), 0, stream, key, slot, first, rows,
                       index.owners, index.sources(), index.table() + first, index.count(), index.cache);
    PRISM_HIP(hipGetLastError());
    first += rows;
  }
}

void v4::launchSharedProjection(AttentionArgs args, ProjectionIndex index, Span<uint16_t> projections, int headPairs,
                                const int (&groupRows)[kAttentionGroups], hipStream_t stream) {
  if (index.tokens == 0) return;
  const int blocks = (index.tokens + kProjectionRows - 1) / kProjectionRows;
  hipLaunchKernelGGL((projectDistinctInputs<kAttentionWidth.width>), dim3(blocks), dim3(tile::kThreads), 0, stream,
                     args, index.sources(), index.count(), projections, index);
  PRISM_HIP(hipGetLastError());
  args.qkv0 = {projections.data, projections.limit};
  const RowAttentionArgs mapped{
      args, {index.owners, static_cast<size_t>(index.tokens)}, {index.cache.qkv.data, index.cache.qkv.limit}};
  launchGroups<kAttentionWidth.width, false, true, EventKeys::kProjected>(
      mapped, groupRows, headPairs, stream, std::make_integer_sequence<int, kAttentionGroups>{});
}

}  // namespace prism
