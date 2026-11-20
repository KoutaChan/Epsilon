#include "prism_hip_mlp.h"
#include "prism_hip_capabilities.h"
#include "prism_hip_v4.h"

namespace prism::v4 {
namespace {

constexpr int kTokensPerBlock = 8;
constexpr int kEmbedThreads = kTokensPerBlock * kWave;
constexpr int kColumnsPerLane = 4;

using u16x4 = uint16_t __attribute__((ext_vector_type(kColumnsPerLane)));

struct EmbedRow {
  int slot, input, output;
};

// 代表の番号は局・席・牌・イベントを通した番号。入力の列は各種類の先頭から数える。
template <bool indexed>
__device__ __forceinline__ EmbedRow embedRow(const EmbedArgs& args, int slot, int token, const int* sources) {
  if constexpr (!indexed) return {slot, token, token};
  const int output = sources[token];
  int row = output;
  for (slot = 0; slot < static_cast<int>(EmbeddingSlot::kEvent); ++slot) {
    if (row < args.count[slot]) break;
    row -= args.count[slot];
  }
  return {slot, row, output};
}

template <int H, bool indexed>
__global__ void wideEmbedKernel(EmbedArgs args, int slot, Span<const float> gain, Span<uint16_t> normalized,
                                ProjectionIndex index) {
  const int* sources = indexed ? index.sources() : nullptr;
  __shared__ int tableRows[kTokensPerBlock][kMaxTables];
  __shared__ float features[kTokensPerBlock][kMaxDense];
  alignas(8) __shared__ uint16_t state[kTokensPerBlock][H + tile::kPad];
  const int first = blockIdx.x * kTokensPerBlock;
  const int count = min(kTokensPerBlock, (indexed ? *index.count() : args.count[slot]) - first);
  if (count <= 0) return;
  const int tid = threadIdx.x;
  const uint8_t* input = args.input.data;
  Bounds bounds;
  for (int i = tid; i < count * kMaxTables; i += kEmbedThreads) {
    const int row = i / kMaxTables;
    const int table = i % kMaxTables;
    const auto src = embedRow<indexed>(args, slot, first + row, sources);
    const auto& spec = args.spec[src.slot];
    if (table >= spec.tables) continue;
    const size_t base = args.section[spec.category] + static_cast<size_t>(src.input) * args.sectionWidth[spec.category];
    const auto& item = spec.table[table];
    int entry = 0;
    for (int m = 0; m < item.members; ++m) {
      entry += input[bounds.index(base + item.column[m], 1, args.input.limit)] * item.stride[m];
    }
    tableRows[row][table] = bounds.index(entry, 1, item.rows);
  }
  for (int i = tid; i < count * kMaxDense; i += kEmbedThreads) {
    const int row = i / kMaxDense;
    const int k = i % kMaxDense;
    const auto src = embedRow<indexed>(args, slot, first + row, sources);
    const auto& spec = args.spec[src.slot];
    if (k >= spec.dense) continue;
    const int kind = spec.source[k] >> 16;
    const int column = spec.source[k] & 0xffff;
    const size_t token = src.input;
    if (kind == static_cast<int>(DenseSource::kScalar)) {
      const size_t at = args.section[spec.scalar] + (token * args.sectionWidth[spec.scalar] + column) * 2;
      features[row][k] = bf16Float(*reinterpret_cast<const uint16_t*>(input + bounds.index(at, 2, args.input.limit)));
    } else if (kind == static_cast<int>(DenseSource::kExact)) {
      const size_t at = args.section[spec.exact] + (token * args.sectionWidth[spec.exact] + column) * 4;
      features[row][k] =
          bf16Float(bf16Round(*reinterpret_cast<const float*>(input + bounds.index(at, 4, args.input.limit))));
    } else {
      const size_t at = args.section[spec.category] + token * args.sectionWidth[spec.category] + column;
      features[row][k] = input[bounds.index(at, 1, args.input.limit)];
    }
  }
  tile::barrier();
  const uint8_t* pack = args.pack.data;
  // ウェーブごとにトークンを 1 個受け持つ。全列で共通の表番号と密な特徴を一度読み、列ごとの加算順を保つ。
  constexpr int kVectorsPerLane = H / (kColumnsPerLane * kWave);
  static_assert(H % (kColumnsPerLane * kWave) == 0);
  const int token = tid / kWave;
  const int lane = tid % kWave;
  if (token < count) {
    const auto src = embedRow<indexed>(args, slot, first + token, sources);
    const auto& spec = args.spec[src.slot];
    const int dstRow = indexed ? index.owners[src.output] : src.output;
    const auto out = dstRow < 0 ? index.cache.residual : args.out[src.slot];
    f32x4 sums[kVectorsPerLane]{};
    for (int t = 0; t < spec.tables; ++t) {
      const auto& table = spec.table[t];
      const size_t base = table.offset + static_cast<size_t>(tableRows[token][t]) * H * sizeof(float);
#pragma unroll
      for (int j = 0; j < kVectorsPerLane; ++j) {
        const size_t at = base + (j * kWave + lane) * sizeof(f32x4);
        sums[j] += *reinterpret_cast<const f32x4*>(pack + bounds.index(at, sizeof(f32x4), args.pack.limit));
      }
    }
    for (int k = 0; k < spec.dense; ++k) {
      const size_t base = spec.denseOffset + static_cast<size_t>(k) * H * sizeof(uint16_t);
      const float feature = features[token][k];
#pragma unroll
      for (int j = 0; j < kVectorsPerLane; ++j) {
        const size_t at = base + (j * kWave + lane) * sizeof(u16x4);
        const auto weight = *reinterpret_cast<const u16x4*>(pack + bounds.index(at, sizeof(u16x4), args.pack.limit));
#pragma unroll
        for (int col = 0; col < kColumnsPerLane; ++col) {
          sums[j][col] = fmaf(bf16Float(weight[col]), feature, sums[j][col]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kVectorsPerLane; ++j) {
      const int c = (j * kWave + lane) * kColumnsPerLane;
      const size_t at = spec.biasOffset + c * sizeof(float);
      sums[j] += *reinterpret_cast<const f32x4*>(pack + bounds.index(at, sizeof(f32x4), args.pack.limit));
      u16x4 rounded;
#pragma unroll
      for (int col = 0; col < kColumnsPerLane; ++col) rounded[col] = bf16RoundActivation(sums[j][col]);
      *reinterpret_cast<u16x4*>(&state[token][c]) = rounded;
      const size_t dst =
          bounds.index(static_cast<size_t>(dstRow < 0 ? ~dstRow : src.input) * H + c, kColumnsPerLane, out.limit);
      *reinterpret_cast<u16x4*>(out.data + dst) = rounded;
    }
  }
  if (normalized.data != nullptr) {
    tile::barrier();
    const int row = tid / 4;
    if (row < count) {
      const auto src = embedRow<indexed>(args, slot, first + row, sources);
      const int c = (tid % 4) * (H / 4);
      u32x4 values[H / 32];
#pragma unroll
      for (int j = 0; j < H / 32; ++j) {
        values[j] = *reinterpret_cast<const u32x4*>(&state[row][c + 8 * j]);
      }
      const float inverse = tile::inverseRms<H>(values, args.epsilon);
      const size_t at = static_cast<size_t>(src.output) * H + c;
#pragma unroll
      for (int j = 0; j < H / 32; ++j) {
        *reinterpret_cast<u32x4*>(normalized.data + bounds.index(at + 8 * j, 8, normalized.limit)) =
            tile::normalize8(values[j], gain.data, gain.limit, c + 8 * j, inverse, bounds);
      }
    }
  }
  bounds.trap();
}

}  // namespace

void launchWideEmbed(const EmbedArgs& args, Span<const float> gain, const Span<uint16_t> (&normalized)[kEmbeddingSlots],
                     ProjectionIndex index, hipStream_t stream) {
  int firstSlot = 0;
  if (index.owners) {
    const int blocks = (index.tokens + kTokensPerBlock - 1) / kTokensPerBlock;
    if (blocks > 0) {
      hipLaunchKernelGGL((wideEmbedKernel<kWidthM.width, true>), dim3(blocks), dim3(kEmbedThreads), 0, stream, args,
                         firstSlot, gain, normalized[firstSlot], index);
      PRISM_HIP(hipGetLastError());
    }
    firstSlot = static_cast<int>(EmbeddingSlot::kTransition);
  }
  for (int slot = firstSlot; slot < kEmbeddingSlots; ++slot) {
    const int blocks = (args.count[slot] + kTokensPerBlock - 1) / kTokensPerBlock;
    if (blocks > 0) {
      hipLaunchKernelGGL((wideEmbedKernel<kWidthM.width, false>), dim3(blocks), dim3(kEmbedThreads), 0, stream, args,
                         slot, gain, normalized[slot], index);
      PRISM_HIP(hipGetLastError());
    }
  }
}

}  // namespace prism::v4
