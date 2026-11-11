// K1（embed）: トークンの埋め込みと、遷移の静的な部分を作る。フィールドの値の組ごとに、BF16 にした W と E の積を FP32 で前もって
// 計算した表（組の表）を引いて足す。
//
//   x = ⌊acc⌉、acc = Σ_表 T[Σ_i 値(列_i)·ストライド_i] + Σ_k W_d[k]·特徴_k + b
//
// 表 T は FP32 の [行, H]（WeightPack の中で W·E を折り込んだもの）。W_d は表に折り込まない特徴（2 値のフィールド・数値・厳密な値を
// BF16 にしたもの）の重みで、BF16 の [特徴, H]。b は BF16 に丸めたバイアス。BF16 どうしの積は FP32 で厳密なので、eager の Linear との
// 差は足す順だけ。和は 0 から始めて、表の順、特徴の順、b の順に FP32 で足す（どのトークンでも同じ順）。
//
// 局・席・牌・イベントのトークンは、層 0 の Attention の q・k・v も折り込んだ表から作って QKV0 へ書く（K2 の層 0 は射影をせずにこれを読む）:
//
//   qkv0 = ⌊r·(Σ_表 T'[行] + Σ_k W'_d[k]·特徴_k + b') + b_qkv⌉、r = 1/√(mean(acc²) + ε)
//
// T' = T·diag(g0)·W_qkv0ᵀ、W'_d = W_d·diag(g0)·W_qkv0ᵀ、b' = b·diag(g0)·W_qkv0ᵀ は、重みを公開するときに FP64 で計算して FP32 に丸めた
// もの（WeightPack。列はヘッドの対ごとの並び foldColumn）。r は丸める前の acc から求めるので、x と n の BF16 の丸めは qkv0 の計算に
// 入らない（残差の x は丸めて書く）。和は 0 から始めて、表の順、特徴の順（fma）、b' の順に FP32 で足し、r との積と b_qkv との和は別々に
// 丸める（fma にしない）。r は、lane の 8 列の acc² を順に fma で足し、ウェーブの 32 lane を xor のツリー（waveSum）で足して、平均と ε の
// 和・平方根・割り算を IEEE の丸めで求める（rsqrt の近似は使わない）。CPU の参照実装（HipReference）は同じ順で同じ値を作る。
//
// ワークグループは 1 つの型の 64 トークンを受け持つ（ウェーブごとに 8 トークン。lane は x の H/32 列と qkv0 の 3A/32 列を持つ）。型ごとの
// ワークグループは EmbedArgs.blockStart の順に並ぶ。ワークグループの始めに、型の仕様と W_d・W'_d を LDS へ 1 回コピーする（グローバル
// メモリにある仕様は、コンパイラから見ると出力と重なりうるので、そのまま読むとトークンごとに読み直しになる）。W_d・W'_d の LDS は動的に確保し、
// 大きさは全ての型の最大（embedSharedBytes）。ウェーブでは lane t が表 t の位置・フィールドの列とストライドを、lane k が特徴 k の読み元を
// 持ち、トークンごとに次の順で計算する。
//   1. lane t は表 t のフィールドのバイトを全て、lane k は特徴 k を同時に読む（グローバルメモリの 1 往復）
//   2. lane t が表 t の行の先頭を求める。全ての表の行をまとめて読み（1 往復。折り込む型は 4 表ずつ）、表の順に足す
//   3. W_d（LDS から 8 特徴ずつ。折り込む型は 4 特徴ずつまとめて読む）と特徴の積を k の順に足し、b を足して x を書く
//   4. 折り込む型は、2 の後に読み始めた T' の行（4 表ずつ）を表の順に足す。W'_d（LDS）と特徴の積、b' を足して r を掛け、b_qkv を足して書く
// lane が持つ qkv0 の列は、4 列のチャンク c（< 3）ごとに 128c + 4·lane からの 4 列。T'・W'_d・b' の読み込みと qkv0 の書き込みは、
// ウェーブ全体でそれぞれ連続した 512 バイト・256 バイトになる。グローバルメモリの位置は、領域の中の 32 bit のバイト位置で計算する（重みの
// パックと入力の領域が 2 GiB 未満であることは計画が確かめる）。行の先頭は SGPR、lane の列だけが VGPR なので、lane ごとの 64 bit の
// 位置をトークンのループの外に持たない。
#include <algorithm>
#include <type_traits>

#include "prism_hip_common.h"
#include "prism_hip_kernels.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kTokens = 64;
constexpr int kTokensPerWave = kTokens / kWaves;
constexpr int kSpecInts = static_cast<int>(sizeof(EmbeddingSpec) / sizeof(int32_t));
constexpr int kDensePasses = kMaxDense / kWaves;  // W_d・W'_d を LDS へコピーする回数（ウェーブ w は特徴 w + 8·pass の行を受け持つ）
constexpr int kDenseGroup = 8;                    // LDS から W_d をまとめて読む特徴の数（遷移の型）
constexpr int kFoldColumns = 4;                   // lane が持つ qkv0 の 1 チャンクの列の数（FP32 で 16 バイト、BF16 で 8 バイト）
// 折り込む型の読み込みのまとめ方。T・T' の行は 4 表ずつ（局は 4 表、牌・イベントは 3 表で 1 往復、席だけは 6 表で 2 往復）、W_d は
// 4 特徴ずつ、W'_d は 2 特徴ずつ読み、b'・b_qkv はトークンごとに読む（グローバルメモリの同じ 3 KB）。どれも読み込みに使う VGPR を抑えて、
// 遷移の型と同じ 10 ウェーブ/SIMD に収めるため。
constexpr int kFoldTables = 4;
constexpr int kFoldWeightGroup = 4;
constexpr int kFoldDenseGroup = 2;
static_assert(kSpecInts <= kThreads, "one thread copies one int32 of the spec");
static_assert(kMaxDense % kWaves == 0 && kMaxDense % kDenseGroup == 0, "W_d is staged and read in whole groups");
static_assert(kDenseGroup % kFoldWeightGroup == 0 && kMaxDense % kFoldDenseGroup == 0, "W_d and W'_d are read in whole groups");
static_assert(kMaxTables <= kWave && kMaxDense <= kWave, "lane t holds table t and lane k dense feature k");
static_assert(kMaxTables % kFoldTables == 0, "T and T' rows are read in whole groups of tables");

// 動的な LDS（16 バイト単位）。型ごとに、まず W_d を [特徴][lane]（lane の 8 列の BF16）で 8 特徴の組の分の行だけ置き、折り込む型は
// 続けて W'_d を [特徴][チャンク][lane]（lane の 4 列の FP32）で置く。ウェーブの b128 の読み込みは連続した 512 バイトになる。組の中で
// 特徴の数を超える W_d の行はコピーしない（読むが掛けない）。W'_d は特徴の数の内の行だけを読む。
constexpr int denseRows(int dense) {
  return (dense + kDenseGroup - 1) / kDenseGroup * kDenseGroup;
}

// lane が持つ qkv0 の 4 列のチャンクの数。
template <int H>
constexpr int foldChunks() {
  return 3 * (H / 2) / (kFoldColumns * kWave);
}

// ウェーブで一様な値を SGPR に移す。仕様のフィールドに使うと表・特徴のループの分岐がスカラーになり、トークンに使うと行の先頭をスカラー
// 命令で計算できる。
__device__ __forceinline__ int uniform(int value) {
  return __builtin_amdgcn_readfirstlane(value);
}

// ウェーブの中で lane 番目の lane が持つ value を返す（結果はウェーブで一様）。
__device__ __forceinline__ uint32_t readLane(uint32_t value, int lane) {
  return __builtin_amdgcn_readlane(value, lane);
}

__device__ __forceinline__ float readLane(float value, int lane) {
  return __builtin_bit_cast(float, readLane(__builtin_bit_cast(uint32_t, value), lane));
}

// 特徴の値。bits は読み元の領域から元の幅のまま読んだもの。2 値のフィールドはバイト、数値は BF16 として使い、厳密な値は FP32 を
// BF16 に丸めて使う。
__device__ __forceinline__ float featureValue(int kind, uint32_t bits) {
  if (kind == static_cast<int>(DenseSource::kScalar)) {
    return bf16Float(static_cast<uint16_t>(bits));
  }
  if (kind == static_cast<int>(DenseSource::kExact)) {
    return bf16Float(bf16Round(__builtin_bit_cast(float, bits)));
  }
  return static_cast<float>(bits);
}

// r·s + b を、積と和で 2 回丸めて計算する（fma にしない。CPU の参照実装と同じ式）。
__device__ __forceinline__ float scaleAdd(float r, float s, float b) {
#pragma clang fp contract(off)
  return r * s + b;
}

// 1/√(total/H + ε) を、IEEE の丸めの割り算・和・平方根・割り算で計算する（rsqrt の近似は使わない。CPU の参照実装と同じ値）。
template <int H>
__device__ __forceinline__ float inverseRms(float total, float epsilon) {
#pragma clang fp contract(off)
  return 1.0f / sqrtf(total / static_cast<float>(H) + epsilon);
}

template <int H>
__global__ __launch_bounds__(kThreads) void embedKernel(EmbedArgs args) {
  constexpr int kPerLane = H / kWave;
  constexpr int kChunks = foldChunks<H>();
  constexpr int kFoldWidth = 3 * (H / 2);
  constexpr uint32_t kRowBytes = H * sizeof(float);
  constexpr uint32_t kFoldRowBytes = kFoldWidth * sizeof(float);
  constexpr uint32_t kChunkBytes = kFoldColumns * kWave * sizeof(float);  // ウェーブの 1 チャンク（T'・W'_d・b' の 512 バイト）
  static_assert(kPerLane == 8, "one lane holds 8 columns (16 bytes of BF16, 32 bytes of FP32)");
  static_assert(kChunks * kFoldColumns * kWave == kFoldWidth, "the lanes hold every qkv0 column in chunks of 4");
  __shared__ EmbeddingSpec spec;
  extern __shared__ u32x4 dense[];

  int slot = 0;
  while (slot + 1 < kEmbeddingSlots && static_cast<int>(blockIdx.x) >= args.blockStart[slot + 1]) {
    ++slot;
  }
  const bool folding = slot != static_cast<int>(EmbeddingSlot::kTransition);
  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int col0 = lane * kPerLane;
  const uint32_t laneOffset = col0 * sizeof(float);  // 表の行の中で lane の 8 列が始まるバイト位置
  const uint32_t foldLane = lane * kFoldColumns * static_cast<uint32_t>(sizeof(float));  // チャンクの中で lane の 4 列が始まるバイト位置
  const int count = args.count[slot];
  const uint8_t* input = args.input.data;
  const uint8_t* pack = args.pack.data;
  Bounds bounds;

  // 0. 仕様を int32 ずつ、W_d を [特徴][lane] の 16 バイトずつ、折り込む型は W'_d を [特徴][チャンク][lane] の 16 バイトずつ LDS へ
  //    コピーする。読み込みを全て発行してから書くので、グローバルメモリの 1 往復で済む（特徴の数と位置はグローバルメモリの仕様から直接
  //    読む）。表・フィールド・特徴の数が上限の内にあることは、重みを公開するときに計画が確かめている。
  {
    const EmbeddingSpec& globalSpec = args.spec[slot];
    const int features = globalSpec.dense;
    const int foldBase = denseRows(features) * kWave;
    u32x4 chunk[kDensePasses];
    u32x4 foldChunk[kDensePasses][kChunks];
#pragma unroll
    for (int pass = 0; pass < kDensePasses; ++pass) {
      const int k = pass * kWaves + wave;
      chunk[pass] = u32x4{0u, 0u, 0u, 0u};
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        foldChunk[pass][c] = u32x4{0u, 0u, 0u, 0u};
      }
      if (k < features) {
        const size_t at = static_cast<size_t>(globalSpec.denseOffset) + (static_cast<size_t>(k) * H + col0) * 2;
        chunk[pass] = *reinterpret_cast<const u32x4*>(pack + bounds.index(at, 16, args.pack.limit));
        if (folding) {
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            const size_t foldAt =
                static_cast<size_t>(globalSpec.foldOffset) + k * kFoldRowBytes + c * kChunkBytes + foldLane;
            foldChunk[pass][c] = *reinterpret_cast<const u32x4*>(pack + bounds.index(foldAt, 16, args.pack.limit));
          }
        }
      }
    }
    if (tid < kSpecInts) {
      reinterpret_cast<int32_t*>(&spec)[tid] = reinterpret_cast<const int32_t*>(&globalSpec)[tid];
    }
#pragma unroll
    for (int pass = 0; pass < kDensePasses; ++pass) {
      const int k = pass * kWaves + wave;
      if (k < features) {
        dense[k * kWave + lane] = chunk[pass];
        if (folding) {
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            dense[foldBase + (k * kChunks + c) * kWave + lane] = foldChunk[pass][c];
          }
        }
      }
    }
  }
  __syncthreads();

  const int category = uniform(spec.category);
  const int scalar = uniform(spec.scalar);
  const int exact = uniform(spec.exact);
  const int tables = uniform(spec.tables);
  const int features = uniform(spec.dense);
  const int foldBase = denseRows(features) * kWave;  // LDS の中の W'_d の先頭（16 バイト単位）
  const uint32_t categoryBase = static_cast<uint32_t>(args.section[category]);
  const uint32_t scalarBase = scalar >= 0 ? static_cast<uint32_t>(args.section[scalar]) : 0u;
  const uint32_t exactBase = exact >= 0 ? static_cast<uint32_t>(args.section[exact]) : 0u;
  const int categoryWidth = args.sectionWidth[category];
  const int scalarWidth = scalar >= 0 ? args.sectionWidth[scalar] : 0;
  const int exactWidth = exact >= 0 ? args.sectionWidth[exact] : 0;

  // lane t（< 表の数）は表 t の位置・T' の位置・行の数・フィールドの列とストライドを持つ（使わないフィールドは列 0・ストライド 0 なので
  // 行の番号に効かない。表を持たない lane は位置 0・行の番号 0）。lane k（< 特徴の数）は特徴 k の読み元と列を持つ（特徴を持たない lane
  // は読み元 −1）。
  const EmbeddingTable& table = spec.table[min(lane, kMaxTables - 1)];
  const bool tableLane = lane < tables;
  const uint32_t tableOffset = tableLane ? static_cast<uint32_t>(table.offset) : 0u;
  const uint32_t tableFoldOffset = tableLane && folding ? static_cast<uint32_t>(table.foldOffset) : 0u;
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
  const int featureKind = lane < features ? denseSource >> 16 : -1;
  const uint32_t featureColumn = static_cast<uint32_t>(denseSource & 0xffff);
  const size_t biasAt =
      bounds.index(static_cast<size_t>(uniform(spec.biasOffset)) + col0 * sizeof(float), 32, args.pack.limit);
  const f32x4 b0 = *reinterpret_cast<const f32x4*>(pack + biasAt);
  const f32x4 b1 = *reinterpret_cast<const f32x4*>(pack + biasAt + 16);

  // ウェーブが受け持つトークンを順に処理する。kFold は層 0 の q・k・v も作る型（局・席・牌・イベント）。
  const auto embedTokens = [&](auto fold) {
    constexpr bool kFold = decltype(fold)::value;
    constexpr int kTableGroup = kFold ? kFoldTables : kMaxTables;
    constexpr int kWeightGroup = kFold ? kFoldWeightGroup : kDenseGroup;
    // b' の行のバイト位置（後ろに b_qkv の行が続く。折り込む型だけ使う）。
    [[maybe_unused]] const uint32_t biasRow = static_cast<uint32_t>(uniform(spec.foldOffset)) + features * kFoldRowBytes;
    for (int i = 0; i < kTokensPerWave; ++i) {
      // トークンはウェーブで一様（SGPR に置き、行の先頭はスカラー命令で計算する）。
      const int token =
          uniform((static_cast<int>(blockIdx.x) - args.blockStart[slot]) * kTokens + wave * kTokensPerWave + i);
      if (token >= count) {
        break;
      }
      // 1. lane t は表 t の行の番号を、lane k は特徴 k の値を求める。読み込みは全て同時に発行する。
      const uint32_t categoryRow = categoryBase + static_cast<uint32_t>(token) * categoryWidth;
      const uint32_t scalarRow = scalarBase + static_cast<uint32_t>(token) * scalarWidth * 2;
      const uint32_t exactRow = exactBase + static_cast<uint32_t>(token) * exactWidth * 4;
      int entry = 0;
#pragma unroll
      for (int m = 0; m < kMaxMembers; ++m) {
        entry += input[bounds.index(categoryRow + memberColumn[m], 1, args.input.limit)] * memberStride[m];
      }
      // 特徴は読み元ごとに別のレジスタへ読む（1 つの変数に読むと 3 つの読み込みが同じレジスタにまとまり、読み込みのたびに前の結果を
      // 待つ）。lane の読み元は 1 つなので、読まなかった 2 つは 0 のままで、論理和が読んだ値になる。
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

      // 2. lane t は表 t の行の先頭（バイト位置）を求める。表の行をまとめて読み（折り込む型は 4 表ずつ）、0 から始めて表の順に足す。
      // 行の番号が語彙の中にあることは、入力を書く側が確かめる。範囲検査つきのビルドでは、表の行の数とも照合する。
      (void)bounds.index(static_cast<size_t>(entry), 1, static_cast<size_t>(tableRows));
      const uint32_t rowStart = tableOffset + static_cast<uint32_t>(entry) * kRowBytes;
      float acc[kPerLane];
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) {
        acc[c] = 0.0f;
      }
#pragma unroll
      for (int g = 0; g < kMaxTables; g += kTableGroup) {
        if (g < tables) {
          f32x4 lo[kTableGroup];
          f32x4 hi[kTableGroup];
#pragma unroll
          for (int t = 0; t < kTableGroup; ++t) {
            if (g + t < tables) {
              const size_t at =
                  bounds.index(readLane(rowStart, g + t) + laneOffset, kPerLane * sizeof(float), args.pack.limit);
              lo[t] = *reinterpret_cast<const f32x4*>(pack + at);
              hi[t] = *reinterpret_cast<const f32x4*>(pack + at + 16);
            }
          }
#pragma unroll
          for (int t = 0; t < kTableGroup; ++t) {
            if (g + t < tables) {
#pragma unroll
              for (int c = 0; c < 4; ++c) {
                acc[c] += lo[t][c];
                acc[4 + c] += hi[t][c];
              }
            }
          }
        }
      }

      // 4 の前半: 最初の 4 表の T' の行を読み始める（3 で W_d を掛けて x を書く間に、読み込みの結果を待つ）。
      [[maybe_unused]] const uint32_t foldStart = tableFoldOffset + static_cast<uint32_t>(entry) * kFoldRowBytes;
      [[maybe_unused]] f32x4 part[kFoldTables][kChunks];
      if constexpr (kFold) {
#pragma unroll
        for (int t = 0; t < kFoldTables; ++t) {
          if (t < tables) {
            const uint32_t row = readLane(foldStart, t) + foldLane;
#pragma unroll
            for (int c = 0; c < kChunks; ++c) {
              part[t][c] = *reinterpret_cast<const f32x4*>(pack + bounds.index(row + c * kChunkBytes, 16, args.pack.limit));
            }
          }
        }
      }

      // 3. W_d[k]·特徴_k を k の順に足す（W_d は 8 特徴（折り込む型は 4 特徴）の組ごとに LDS からまとめて読む。組の中で特徴の数を
      //    超える行は掛けない）。
#pragma unroll
      for (int k0 = 0; k0 < kMaxDense; k0 += kWeightGroup) {
        if (k0 < features) {
          u32x4 group[kWeightGroup];
#pragma unroll
          for (int j = 0; j < kWeightGroup; ++j) {
            group[j] = dense[(k0 + j) * kWave + lane];
          }
#pragma unroll
          for (int j = 0; j < kWeightGroup; ++j) {
            if (k0 + j < features) {
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
      *reinterpret_cast<u32x4*>(out.data + at) = pack8(acc);

      if constexpr (kFold) {
        // 4. r = 1/√(mean(acc²) + ε)（acc は丸める前の x）。
        float square = 0.0f;
#pragma unroll
        for (int c = 0; c < kPerLane; ++c) {
          square = fmaf(acc[c], acc[c], square);
        }
        const float r = inverseRms<H>(waveSum(square), args.epsilon);
        // T' の行を 0 から始めて表の順に足す（5 表目からは、前の 4 表を足した後に読む）。
        float sum[kChunks][kFoldColumns];
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
#pragma unroll
          for (int v = 0; v < kFoldColumns; ++v) {
            sum[c][v] = 0.0f;
          }
        }
#pragma unroll
        for (int g = 0; g < kMaxTables; g += kFoldTables) {
          if (g < tables) {
            if (g > 0) {
#pragma unroll
              for (int t = 0; t < kFoldTables; ++t) {
                if (g + t < tables) {
                  const uint32_t row = readLane(foldStart, g + t) + foldLane;
#pragma unroll
                  for (int c = 0; c < kChunks; ++c) {
                    part[t][c] =
                        *reinterpret_cast<const f32x4*>(pack + bounds.index(row + c * kChunkBytes, 16, args.pack.limit));
                  }
                }
              }
            }
#pragma unroll
            for (int t = 0; t < kFoldTables; ++t) {
              if (g + t < tables) {
#pragma unroll
                for (int c = 0; c < kChunks; ++c) {
#pragma unroll
                  for (int v = 0; v < kFoldColumns; ++v) {
                    sum[c][v] += part[t][c][v];
                  }
                }
              }
            }
          }
        }
        // W'_d[k]·特徴_k を k の順に足す（LDS から 2 特徴ずつ読む。組の中で特徴の数を超える分は、最後の特徴の行を読み直すが掛けない）。
#pragma unroll
        for (int k0 = 0; k0 < kMaxDense; k0 += kFoldDenseGroup) {
          if (k0 < features) {
            f32x4 weights[kFoldDenseGroup][kChunks];
#pragma unroll
            for (int j = 0; j < kFoldDenseGroup; ++j) {
              const int k = min(k0 + j, features - 1);
#pragma unroll
              for (int c = 0; c < kChunks; ++c) {
                weights[j][c] = __builtin_bit_cast(f32x4, dense[foldBase + (k * kChunks + c) * kWave + lane]);
              }
            }
#pragma unroll
            for (int j = 0; j < kFoldDenseGroup; ++j) {
              if (k0 + j < features) {
                const float f = readLane(feature, k0 + j);
#pragma unroll
                for (int c = 0; c < kChunks; ++c) {
#pragma unroll
                  for (int v = 0; v < kFoldColumns; ++v) {
                    sum[c][v] = fmaf(weights[j][c][v], f, sum[c][v]);
                  }
                }
              }
            }
          }
        }
        // b' を足し、r を掛けて b_qkv を足し、BF16 に丸めてチャンクごとに 8 バイトを書く。
        const Span<uint16_t> qkv0 = args.qkv0[slot];
        const size_t row = bounds.index(static_cast<size_t>(token) * kFoldWidth, kFoldWidth, qkv0.limit);
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          const size_t at = bounds.index(biasRow + c * kChunkBytes + foldLane, kFoldRowBytes + 16, args.pack.limit);
          const f32x4 foldBias = *reinterpret_cast<const f32x4*>(pack + at);
          const f32x4 qkvBias = *reinterpret_cast<const f32x4*>(pack + at + kFoldRowBytes);
          uint32_t words[kFoldColumns / 2];
#pragma unroll
          for (int w = 0; w < kFoldColumns / 2; ++w) {
            const float low = scaleAdd(r, sum[c][2 * w] + foldBias[2 * w], qkvBias[2 * w]);
            const float high = scaleAdd(r, sum[c][2 * w + 1] + foldBias[2 * w + 1], qkvBias[2 * w + 1]);
            words[w] = static_cast<uint32_t>(bf16Round(low)) | (static_cast<uint32_t>(bf16Round(high)) << 16);
          }
          *reinterpret_cast<uint2*>(qkv0.data + row + c * kFoldColumns * kWave + lane * kFoldColumns) =
              make_uint2(words[0], words[1]);
        }
      }
    }
  };
  if (folding) {
    embedTokens(std::true_type{});
  } else {
    embedTokens(std::false_type{});
  }
  bounds.trap();
}

}  // namespace

size_t embedSharedBytes(int hidden, const EmbeddingSpec (&spec)[kEmbeddingSlots]) {
  if (hidden != 256) {
    throw std::invalid_argument("K1 has no specialisation for hidden " + std::to_string(hidden));
  }
  const size_t foldRow = static_cast<size_t>(foldChunks<256>()) * kWave * sizeof(u32x4);
  size_t bytes = 0;
  for (int e = 0; e < kEmbeddingSlots; ++e) {
    const bool folding = static_cast<EmbeddingSlot>(e) != EmbeddingSlot::kTransition;
    const size_t slot = static_cast<size_t>(denseRows(spec[e].dense)) * kWave * sizeof(u32x4) +
                        (folding ? static_cast<size_t>(spec[e].dense) * foldRow : 0);
    bytes = std::max(bytes, slot);
  }
  return bytes;
}

void launchEmbed(int hidden, const EmbedArgs& args, int blocks, size_t sharedBytes, hipStream_t stream) {
  if (blocks <= 0) {
    return;
  }
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(embedKernel<256>, dim3(blocks), dim3(kThreads), sharedBytes, stream, args);
      break;
    default:
      throw std::invalid_argument("K1 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
