// K1（embed）: トークンの埋め込みと遷移の静的な部分（BF16 にした W と E の積を FP32 で畳んだ組の表の和）。
//
//   x = ⌊Σ_表 T[Σ_i 値(列_i)·刻み_i] + Σ_k W_d[k]·特徴_k + b⌉
//
// 表 T は FP32 の [行, H]（WeightPack の中で W·E を畳んだもの）、W_d は表に畳まない特徴（2 値の欄・数値・厳密な欄を BF16 にしたもの）の
// BF16 の [特徴, H]、b は BF16 に丸めた偏り。BF16 どうしの積は FP32 で厳密なので、eager の Linear との差は和の順だけ。和は 0 から
// 表の順、特徴の順、b の順に FP32 で足す（トークンの中で固定）。
//
// 作業群は 1 つの型の 64 トークン（波ごとに 8 トークン、lane は H/32 列）。型の作業群は EmbedArgs.blockStart の順に並ぶ。作業群の
// 始めに型の約束と W_d を LDS へ 1 回写す（全体メモリの約束は組み立て器から見ると出力と重なりうるので、そのまま読むとトークンごとに
// 読み直しになる）。
// 波は lane t に表 t の位置・欄の列と刻みを、lane k に特徴 k の元を持ち、トークンごとに
//   1. lane t は表 t の欄のバイトを全て、lane k は特徴 k を同時に読む（全体メモリの 1 往復）
//   2. lane t が表 t の行の先頭を作り、全ての表の行をまとめて読んで（1 往復）、表の順に足す
//   3. W_d（LDS から 8 特徴ずつまとめて読む）と特徴の積を k の順に足し、b を足す
// 全体メモリの位置は区画の中の 32 bit のバイトの位置で作る（重みの束と入力の区画は 2 GiB 未満と計画が確かめる）。行の先頭は SGPR、
// lane の列だけが VGPR なので、lane ごとの 64 bit の位置をトークンの輪の外に持たない。
// 局・席・牌・出来事のトークンは、層 0 の注意の入力 n = ⌊(x·r)·g⌉ も NORMALIZED へ書く（K2 は正規化をせずに読む）。
#include "prism_hip_common.h"
#include "prism_hip_kernels.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kTokens = 64;
constexpr int kTokensPerWave = kTokens / kWaves;
constexpr int kSpecInts = static_cast<int>(sizeof(EmbeddingSpec) / sizeof(int32_t));
constexpr int kDensePasses = kMaxDense / kWaves;  // W_d を写す回（波 w は特徴 w + 8·pass の行を受け持つ）
constexpr int kDenseGroup = 8;                    // W_d の LDS の読みをまとめる特徴の数
static_assert(kSpecInts <= kThreads, "one thread copies one int32 of the spec");
static_assert(kMaxDense % kWaves == 0 && kMaxDense % kDenseGroup == 0, "W_d is staged and read in whole groups");
static_assert(kMaxTables <= kWave && kMaxDense <= kWave, "lane t holds table t and lane k dense feature k");

// LDS。W_d は [特徴][lane] の 16 バイト（lane の 8 列の BF16）で、特徴の外の行は 0。波の b128 の読みは 512 バイトの連続になる。
struct alignas(16) EmbedShared {
  u32x4 dense[kMaxDense][kWave];
  EmbeddingSpec spec;
};

// 波で一様な値を SGPR に移す。約束の欄では表・特徴の輪の分岐がスカラーになり、トークンでは行の先頭がスカラーの命令で作られる。
__device__ __forceinline__ int uniform(int value) {
  return __builtin_amdgcn_readfirstlane(value);
}

// 波の lane 番目の lane が持つ value（波で一様）。
__device__ __forceinline__ uint32_t readLane(uint32_t value, int lane) {
  return __builtin_amdgcn_readlane(value, lane);
}

__device__ __forceinline__ float readLane(float value, int lane) {
  return __builtin_bit_cast(float, readLane(__builtin_bit_cast(uint32_t, value), lane));
}

// 特徴の値（bits は元の区画から元の幅のまま読んだもの）。2 値の欄はバイト、数値は BF16、厳密な欄は FP32 を BF16 に丸めて使う。
__device__ __forceinline__ float featureValue(int kind, uint32_t bits) {
  if (kind == static_cast<int>(DenseSource::kScalar)) {
    return bf16Float(static_cast<uint16_t>(bits));
  }
  if (kind == static_cast<int>(DenseSource::kExact)) {
    return bf16Float(bf16Round(__builtin_bit_cast(float, bits)));
  }
  return static_cast<float>(bits);
}

template <int H>
__global__ __launch_bounds__(kThreads) void embedKernel(EmbedArgs args) {
  constexpr int kPerLane = H / kWave;
  constexpr uint32_t kRowBytes = H * sizeof(float);
  static_assert(kPerLane == 8, "one lane holds 8 columns (16 bytes of BF16, 32 bytes of FP32)");
  __shared__ EmbedShared shared;

  int slot = 0;
  while (slot + 1 < kEmbeddingSlots && static_cast<int>(blockIdx.x) >= args.blockStart[slot + 1]) {
    ++slot;
  }
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int col0 = lane * kPerLane;
  const uint32_t laneOffset = col0 * sizeof(float);  // 表の行の中の lane の 8 列のバイトの位置
  const int count = args.count[slot];
  const uint8_t* input = args.input.data;
  const uint8_t* pack = args.pack.data;
  Bounds bounds;

  // 0. 約束を int32 ずつ、W_d を [特徴][lane] の 16 バイトずつ LDS へ写す。W_d の読みを全て出してから書くので全体メモリの 1 往復で済む
  //    （W_d の特徴の数と位置は全体メモリの約束から直に読む）。表・欄・特徴の数は計画が重みの公開のときに上限の内と確かめている。
  {
    const EmbeddingSpec& globalSpec = args.spec[slot];
    u32x4 chunk[kDensePasses];
#pragma unroll
    for (int pass = 0; pass < kDensePasses; ++pass) {
      const int k = pass * kWaves + wave;
      chunk[pass] = u32x4{0u, 0u, 0u, 0u};
      if (k < globalSpec.dense) {
        const size_t at = static_cast<size_t>(globalSpec.denseOffset) + (static_cast<size_t>(k) * H + col0) * 2;
        chunk[pass] = *reinterpret_cast<const u32x4*>(pack + bounds.index(at, 16, args.pack.limit));
      }
    }
    if (tid < kSpecInts) {
      reinterpret_cast<int32_t*>(&shared.spec)[tid] = reinterpret_cast<const int32_t*>(&globalSpec)[tid];
    }
#pragma unroll
    for (int pass = 0; pass < kDensePasses; ++pass) {
      shared.dense[pass * kWaves + wave][lane] = chunk[pass];
    }
  }
  __syncthreads();

  const EmbeddingSpec& spec = shared.spec;
  const int category = uniform(spec.category);
  const int scalar = uniform(spec.scalar);
  const int exact = uniform(spec.exact);
  const int tables = uniform(spec.tables);
  const int dense = uniform(spec.dense);
  const uint32_t categoryBase = static_cast<uint32_t>(args.section[category]);
  const uint32_t scalarBase = scalar >= 0 ? static_cast<uint32_t>(args.section[scalar]) : 0u;
  const uint32_t exactBase = exact >= 0 ? static_cast<uint32_t>(args.section[exact]) : 0u;
  const int categoryWidth = args.sectionWidth[category];
  const int scalarWidth = scalar >= 0 ? args.sectionWidth[scalar] : 0;
  const int exactWidth = exact >= 0 ? args.sectionWidth[exact] : 0;

  // lane t（< 表の数）は表 t の位置・行の数・欄の列と刻み（使わない欄は列 0・刻み 0 で行の番号に効かない。表を持たない lane は
  // 位置 0・行の番号 0）、lane k（< 特徴の数）は特徴 k の元と列（特徴を持たない lane は元 −1）。
  const EmbeddingTable& table = spec.table[min(lane, kMaxTables - 1)];
  const bool tableLane = lane < tables;
  const uint32_t tableOffset = tableLane ? static_cast<uint32_t>(table.offset) : 0u;
  const int tableRows = tableLane ? table.rows : 1;
  const int members = tableLane ? table.members : 0;
  uint32_t memberColumn[kMaxMembers];
  int memberStride[kMaxMembers];
#pragma unroll
  for (int m = 0; m < kMaxMembers; ++m) {
    memberColumn[m] = m < members ? static_cast<uint32_t>(table.column[m]) : 0u;
    memberStride[m] = m < members ? table.stride[m] : 0;
  }
  const int denseSource = spec.source[min(lane, kMaxDense - 1)];
  const int featureKind = lane < dense ? denseSource >> 16 : -1;
  const uint32_t featureColumn = static_cast<uint32_t>(denseSource & 0xffff);
  const size_t biasAt =
      bounds.index(static_cast<size_t>(uniform(spec.biasOffset)) + col0 * sizeof(float), 32, args.pack.limit);
  const f32x4 b0 = *reinterpret_cast<const f32x4*>(pack + biasAt);
  const f32x4 b1 = *reinterpret_cast<const f32x4*>(pack + biasAt + 16);

  for (int i = 0; i < kTokensPerWave; ++i) {
    // トークンは波で一様（SGPR。行の先頭はスカラーの命令で作る）。
    const int token =
        uniform((static_cast<int>(blockIdx.x) - args.blockStart[slot]) * kTokens + wave * kTokensPerWave + i);
    if (token >= count) {
      break;
    }
    // 1. lane t は表 t の行の番号、lane k は特徴 k の値。読みは全て同時に出す。
    const uint32_t categoryRow = categoryBase + static_cast<uint32_t>(token) * categoryWidth;
    const uint32_t scalarRow = scalarBase + static_cast<uint32_t>(token) * scalarWidth * 2;
    const uint32_t exactRow = exactBase + static_cast<uint32_t>(token) * exactWidth * 4;
    int entry = 0;
#pragma unroll
    for (int m = 0; m < kMaxMembers; ++m) {
      entry += input[bounds.index(categoryRow + memberColumn[m], 1, args.input.limit)] * memberStride[m];
    }
    // 特徴は元ごとに別のレジスタへ読む（1 つの変数に読むと 3 つの読みが同じレジスタにまとまり、読みごとに前の戻りを待つ）。lane の
    // 元は 1 つなので、読まなかった 2 つは 0 のままで、論理和が読んだ値になる。
    uint32_t flagBits = 0;
    uint32_t scalarBits = 0;
    uint32_t exactBits = 0;
    if (featureKind == static_cast<int>(DenseSource::kFlag)) {
      flagBits = input[bounds.index(categoryRow + featureColumn, 1, args.input.limit)];
    }
    if (featureKind == static_cast<int>(DenseSource::kScalar)) {
      const size_t at = bounds.index(scalarRow + featureColumn * 2, 2, args.input.limit);
      scalarBits = *reinterpret_cast<const uint16_t*>(input + at);
    }
    if (featureKind == static_cast<int>(DenseSource::kExact)) {
      const size_t at = bounds.index(exactRow + featureColumn * 4, 4, args.input.limit);
      exactBits = *reinterpret_cast<const uint32_t*>(input + at);
    }
    const float feature = featureValue(featureKind, flagBits | scalarBits | exactBits);

    // 2. lane t は表 t の行の先頭（バイトの位置）を作る。全ての表の行をまとめて読み、0 から表の順に足す。
    // 行の番号は語彙の中（書き手が確かめる）。検査つきの組み立ては表の行の数でも照らす。
    (void)bounds.index(static_cast<size_t>(entry), 1, static_cast<size_t>(tableRows));
    const uint32_t rowStart = tableOffset + static_cast<uint32_t>(entry) * kRowBytes;
    f32x4 lo[kMaxTables];
    f32x4 hi[kMaxTables];
#pragma unroll
    for (int t = 0; t < kMaxTables; ++t) {
      if (t < tables) {
        const size_t at = bounds.index(readLane(rowStart, t) + laneOffset, kPerLane * sizeof(float), args.pack.limit);
        lo[t] = *reinterpret_cast<const f32x4*>(pack + at);
        hi[t] = *reinterpret_cast<const f32x4*>(pack + at + 16);
      }
    }
    float acc[kPerLane];
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      acc[c] = 0.0f;
    }
#pragma unroll
    for (int t = 0; t < kMaxTables; ++t) {
      if (t < tables) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          acc[c] += lo[t][c];
          acc[4 + c] += hi[t][c];
        }
      }
    }

    // 3. W_d[k]·特徴_k を k の順に足す（W_d は 8 特徴の組ごとにまとめて LDS から読む。組の中の特徴の外の行は 0 で、掛けない）。
#pragma unroll
    for (int k0 = 0; k0 < kMaxDense; k0 += kDenseGroup) {
      if (k0 < dense) {
        u32x4 group[kDenseGroup];
#pragma unroll
        for (int j = 0; j < kDenseGroup; ++j) {
          group[j] = shared.dense[k0 + j][lane];
        }
#pragma unroll
        for (int j = 0; j < kDenseGroup; ++j) {
          if (k0 + j < dense) {
            const float f = readLane(feature, k0 + j);
            float weights[kPerLane];
            unpack8(group[j], weights);
#pragma unroll
            for (int c = 0; c < kPerLane; ++c) {
              acc[c] = fmaf(weights[c], f, acc[c]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      acc[c] += b0[c];
      acc[4 + c] += b1[c];
    }
    const Span<uint16_t> out = args.out[slot];
    const size_t at = bounds.index(static_cast<size_t>(token) * H + col0, kPerLane, out.limit);
    const u32x4 packed = pack8(acc);
    *reinterpret_cast<u32x4*>(out.data + at) = packed;
    if (slot != static_cast<int>(EmbeddingSlot::kTransition)) {
      // 層 0 の注意の入力 n = ⌊(x·r)·g⌉（x は丸めた値、r = rsqrt(mean(x²) + ε) は FP32 で波の 32 lane の和）。K2 が A の断片として直に読む。
      float x[kPerLane];
      unpack8(packed, x);
      float sum = 0.0f;
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) {
        sum += x[c] * x[c];
      }
      const float r = rsqrtf(waveSum(sum) / static_cast<float>(H) + args.epsilon);
      const size_t g = bounds.index(col0, kPerLane, args.attentionNorm.limit);
      float n[kPerLane];
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) {
        n[c] = (x[c] * r) * args.attentionNorm.data[g + c];
      }
      const Span<uint16_t> normalized = args.normalized[slot];
      const size_t nAt = bounds.index(static_cast<size_t>(token) * H + col0, kPerLane, normalized.limit);
      *reinterpret_cast<u32x4*>(normalized.data + nAt) = pack8(n);
    }
  }
  bounds.trap();
}

}  // namespace

void launchEmbed(int hidden, const EmbedArgs& args, int blocks, hipStream_t stream) {
  if (blocks <= 0) {
    return;
  }
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(embedKernel<256>, dim3(blocks), dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K1 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
