#pragma once

#include "gradients.h"
#include "static_shared_memory.h"

namespace prism {
namespace {

// 学習用の Attention（関係のバイアスつき、バリューには相対の項を足さない）の順伝播と逆伝播。タイルの積和の命令を使い、FlashAttention-2 と同じ形で
// 計算する。GPU で学習するとき、eager の EncoderLayer が呼ぶ packedRelationScaledDotProductAttention（djl-rocm）の代わりに使う。数式は同じ:
//
//   ℓ_ij = (q_i·k_j)·scale + Σ_s ρ[entries·s + c_s(i, j), h]（キー j が有効なときだけ）、p_i = softmax_j(ℓ_i)、o_i = Σ_j p_ij v_j
//
// 内側の計算は ℓ₂ = ℓ·log₂e の単位で行う（exp2 を使う）。バイアスは、セグメント 2 つずつを先に足した組の表
// τ_t[a + 16b] = (ρ(2t, a) + ρ(2t + 1, b))·log₂e をコードのバイト t で 4 回引き、((τ₀ + τ₁) + τ₂) + τ₃ の順に足す。ワークグループは、
// ヘッドの ρ の列（セグメント × 値の 128 個、範囲外は 0）をスレッドで分担して 1 回で共有メモリに置き、表はその共有メモリから作る（表の 1 項ごとに
// グローバルメモリを読まない）。順伝播と逆伝播は同じ式なので、ℓ₂ はビットまで同じになる。
//
// タイルの積和に入れる P と dS は、FP32 を BF16 の 3 項 hi + mid + lo（どれも上位 16 bit を切り捨てた値）に分けて 3 回掛ける。3 項の和は元の
// FP32 と等しく、BF16 の q・k・v・do との積も FP32 で正確なので、変わるのは FP32 で足す順だけ。q·k と do·v は BF16 どうしの積なので、
// そのまま掛ける。
//
// 順伝播（ワークグループは 1 行のヘッド 2 つを受け持ち、ウェーブはクエリのタイル（16 個）を 1 つ受け持つ）:
//   S^T = K·Qᵀ、ℓ₂ = S^T·scale·log₂e + τ、オンラインの softmax、O^T += V^T·P^T。o は BF16 に 1 回丸める。逆伝播のために、FP32 の o と
//   log Σ exp = (m₂ + log₂ Σ)·ln 2 も書く。
// 逆伝播（ワークグループは 1 行のヘッド 1 つを受け持つ。クエリのタイルごとに 2 段で計算し、段の間にワークグループのバリアを置く）:
//   段 A（ウェーブ kt がキーのタイル kt を受け持ち、lane はキー 1 つとクエリ 8 つを持つ）: S = Q·Kᵀ、dP = dO·Vᵀ、P = 2^(ℓ₂ − lse₂)、
//        dS = P·(dP − δ)、dV^T += dO^T·P、dK^T += Q^T·dS。dS とコードは、共有メモリにクエリのタイルの行として置く。
//   段 B: ウェーブ 0〜3 は、セグメント 2 つずつ ρ の勾配を lane ごとの部分和に足す（アトミックな加算は使わない）。ウェーブ 4 は
//        dQ^T = K^T·dS^T（キーのタイルの順）を書き、次のクエリのタイルの Q^T・dO^T を共有メモリに置く。グローバルメモリの読み込みは dQ の前に
//        出し、待ち時間を dQ の計算で隠す。
//   dS とコードの共有メモリの行のストライドは、64 を法として 36 ワードにする。こうすると、段 B で 16 行 × 2 つの半分を 16 バイトずつ読むときの
//   バンク競合は 2-way（最小）になり、段 A で 2 つの半分を書くときはバンクが重ならない。
//   δ_i = Σ_d do_id·o_id（o は FP32）。dq・dk・dv は FP32 で作り、ちょうど 0 になった和を +0 にそろえてから（positiveZero）、BF16 に
//   最近接偶数で 1 回丸める（c10 の型変換と同じビット）。ρ の勾配はワークグループ（行 × ヘッド）ごとの部分和を FP32 で書き、呼び出し元が
//   行について足し合わせる（順の決まったリダクション）。
//   計算するのは生きたクエリ（i < queries で、do_i の 16 値に 0 でない値があるもの）だけ。do_i = 0 なら dP_i = δ_i = 0、dS_i = 0 なので、
//   dk・dv・dq・ρ の勾配への寄与はちょうど 0 になる（近似ではない）。死んだクエリは範囲外のクエリと同じに扱い（Q^T・dO^T は 0、P = +0、
//   dS = +0、ρ の勾配に足さない、dq は +0）、生きたクエリを含む最後のタイルより後ろのタイルは計算しない。
// リダクションの順は、全てワークグループの中で決まっている（タイルの積和の累積、キーのタイルの順、lane の部分和を足す順）。有効なキーの dk・dv に
// 入るタイルの積和は、パディングのクエリ（出力の勾配がちょうど 0）を足しても同じ順で、A・B・C もビットで同じ（パディングの q はタイルの積和に入れない。
// タイルの積和は、積が 0 でも A の値で積の丸めが変わりうるため）。よって結果は、行ごとに、バッチの組み方とパディングの幅に
// 依らずビットで同じになる。前提は、パディングの出力の勾配がちょうど 0 であること。
// グローバルメモリのアドレスは、一様な行の先頭と lane ごとの 32 bit のオフセットに分ける（読み書きがウェーブで一様なレジスタの先頭と lane ごとのレジスタの
// オフセットになる）。
// 検査つきのビルドでは、全てのグローバルメモリの読み書きについて、行の範囲が配列の中にあることと、lane のオフセットが行の範囲の中にある
// ことの両方を確かめる。
// 範囲外のコード（セグメント s < segments の値が entries 以上）が来たら止める。検査するのは、有効なキーを 1 つでも含むキーのタイルの全ての
// コード（タイルの中の無効なキーと、範囲外のクエリの行も含む。範囲外のクエリは最後の行を読む。有効なキーが無いタイルは読まない）。学習用の関係コードでは
// パディングの組にも範囲内のコードを書く。entries = 16 なら、この検査は一様な分岐で飛ばす。
constexpr int kF = kAttentionFeatures;
constexpr int kTables = 4;          // 組の表（コードのバイト t ごとに 1 つ。セグメント 2t と 2t + 1）
constexpr int kTableEntries = 256;  // 組の表の項の数（バイトの値の数）
constexpr int kRhoEntries =
    kAttentionMaxSegments * kAttentionMaxEntries;  // 共有メモリに置くヘッドの ρ の列（セグメント × 値）
constexpr int kScatterWaves = 4;                   // 段 B で ρ の勾配を足すウェーブ（セグメント 2 つずつ）
constexpr int kServiceWave = kScatterWaves;        // 段 B で dQ と次の Q^T・dO^T を受け持つウェーブ
constexpr int kMinBackwardWaves = kServiceWave + 1;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
static_assert(2 * kScatterWaves == kAttentionMaxSegments, "the scatter waves cover every segment of a code word");
static_assert(2 * kTables == kAttentionMaxSegments, "the pair tables cover every segment of a code word");
static_assert(kAttentionMaxEntries == 16, "a segment code is one nibble");

// キーのタイルの数の上限。カーネルはこの 3 段のどれかで作り、選んだ段で共有メモリの大きさが決まる（S ≤ 16·kBlocks）。
constexpr int kSmallBlocks = 5;
constexpr int kMediumBlocks = 8;
constexpr int kLargeBlocks = kAttentionMaxTokens / 16;

// 次元 d を A のフラグメントの行 r(d) = 2(d & 7) + (d >> 3) に置くと、C のフラグメントの行 2v + half が次元 v + 8·half になる（lane が
// 連続する 8 次元を持ち、16 バイトで書ける）。
__device__ __forceinline__ int featureRow(int d) { return 2 * (d & 7) + (d >> 3); }

// C のフラグメントの行 m = 2v + half の値を、B のフラグメントの K の位置 κ(m) = v + 8·half に置く（lane の 8 値と、対になる lane の 8 値を
// 並べるだけで B のフラグメントになる）。
__device__ __forceinline__ int fragmentColumn(int m) { return (m >> 1) + 8 * (m & 1); }

// lane の 8 つの FP32 を BF16 の 3 項 hi + mid + lo（どれも上位 16 bit を切り捨てた値）に分け、対になる lane の 8 値と合わせて B の
// フラグメント 3 つにする。hi が仮数の上位 8 bit、mid が残りの上位 8 bit、lo が残り（8 bit 以下）を持つので、3 項の和は元の値と
// 等しい（BF16 の正規数の範囲で）。
__device__ __forceinline__ void splitFragments(const float (&x)[8], bf16x16 (&terms)[3]) {
  u32x4 own[3];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float a = x[2 * w];
    float b = x[2 * w + 1];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      const uint32_t ua = __float_as_uint(a);
      const uint32_t ub = __float_as_uint(b);
      own[t][w] = upperHalves(ua, ub);
      a -= __uint_as_float(ua & 0xffff0000u);
      b -= __uint_as_float(ub & 0xffff0000u);
    }
  }
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    terms[t] = joinHalves(own[t]);
  }
}

// 積 p·d を splitFragments と同じ 3 項に分ける。hi は丸めた積から取り、mid と lo を取り出す残りは、丸める前の積から hi を引いて 1 回で
// 丸めた値にする。
__device__ __forceinline__ void splitProductFragments(const float (&p)[8], const float (&d)[8], bf16x16 (&terms)[3]) {
  u32x4 own[3];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float a = roundedMultiply(p[2 * w], d[2 * w]);
    float b = roundedMultiply(p[2 * w + 1], d[2 * w + 1]);
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      const uint32_t ua = __float_as_uint(a);
      const uint32_t ub = __float_as_uint(b);
      own[t][w] = upperHalves(ua, ub);
      if (t == 0) {
        a = fusedMultiplyAdd(p[2 * w], d[2 * w], -__uint_as_float(ua & 0xffff0000u));
        b = fusedMultiplyAdd(p[2 * w + 1], d[2 * w + 1], -__uint_as_float(ub & 0xffff0000u));
      } else {
        a -= __uint_as_float(ua & 0xffff0000u);
        b -= __uint_as_float(ub & 0xffff0000u);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    terms[t] = joinHalves(own[t]);
  }
}

// acc += A·(lo + mid + hi)（小さい項から足す）。
__device__ __forceinline__ f32x8 mma3(bf16x16 a, const bf16x16 (&terms)[3], f32x8 acc) {
  acc = mma(a, terms[2], acc);
  acc = mma(a, terms[1], acc);
  return mma(a, terms[0], acc);
}

// 16 バイトの値 2 つを 8 値の配列に並べる。
template <class T, class V>
__device__ __forceinline__ void unpackEight(V first, V second, T (&values)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = first[i];
    values[4 + i] = second[i];
  }
}

// 指定したヘッドの ρ の列を共有メモリの rho[16s + c] に置く（s < segments かつ c < entries のときだけ表の値、ほかは 0）。ワークグループの
// スレッドで添字を分担し、1 回で読む。
__device__ __forceinline__ void stageRho(float* rho, const AttentionShape& shape, Span<const float> table, int head,
                                         int tid, int threads, Bounds& bounds) {
  for (int e = tid; e < kRhoEntries; e += threads) {
    const int s = e / kAttentionMaxEntries;
    const int c = e % kAttentionMaxEntries;
    float value = 0.0f;
    if (s < shape.segments && c < shape.entries) {
      const size_t at = (static_cast<size_t>(s) * shape.entries + c) * shape.heads + head;
      value = table.data[bounds.index(at, 1, table.limit)];
    }
    rho[e] = value;
  }
}

// 組の表の項 τ_t[a + 16b] = (ρ(2t, a) + ρ(2t + 1, b))·log₂e（e = 256t + a + 16b。rho は stageRho が置いた列）。
__device__ __forceinline__ float pairEntry(const float* rho, int e) {
  const int t = e / kTableEntries;
  const float first = rho[kAttentionMaxEntries * (2 * t) + (e & 15)];
  const float second = rho[kAttentionMaxEntries * (2 * t + 1) + ((e >> 4) & 15)];
  return (first + second) * kLog2e;
}

// コードのバイアス ((τ₀ + τ₁) + τ₂) + τ₃（log₂e を掛けた値）。
__device__ __forceinline__ float pairBias(uint32_t code, const float* table) {
  return ((table[code & 0xffu] + table[kTableEntries + ((code >> 8) & 0xffu)]) +
          table[2 * kTableEntries + ((code >> 16) & 0xffu)]) +
         table[3 * kTableEntries + (code >> 24)];
}

// ヘッド 2 つ分の表から引く（ヘッドごとに、上の pairBias と同じ順で足す）。
__device__ __forceinline__ float2 pairBias(uint32_t code, const float2* table) {
  const float2 t0 = table[code & 0xffu];
  const float2 t1 = table[kTableEntries + ((code >> 8) & 0xffu)];
  const float2 t2 = table[2 * kTableEntries + ((code >> 16) & 0xffu)];
  const float2 t3 = table[3 * kTableEntries + (code >> 24)];
  return make_float2(((t0.x + t1.x) + t2.x) + t3.x, ((t0.y + t1.y) + t2.y) + t3.y);
}

// 範囲外のコード（セグメント s < segments の値が entries 以上）は、djl-rocm の演算と同じく止める。4 bit の値を 8 bit のフィールド
// 2 つ（偶数と奇数のセグメント）に分けて 128 − entries を足すと、フィールドの最上位 bit が「値 ≥ entries」を表す。entries が
// 16 なら全て範囲内なので、呼び出し元はタイルごとに 1 回の一様な分岐でこの検査を飛ばす（内側のループを 1 つの基本ブロックのまま保つ）。
__device__ __forceinline__ void checkCodes(const uint32_t (&code)[8], int segments, int entries) {
  const uint32_t bias = static_cast<uint32_t>(128 - entries) * 0x01010101u;
  uint32_t even = 0;
  uint32_t odd = 0;
#pragma unroll
  for (int t = 0; t < kTables; ++t) {
    even |= 2 * t < segments ? 0x80u << (8 * t) : 0u;
    odd |= 2 * t + 1 < segments ? 0x80u << (8 * t) : 0u;
  }
  uint32_t outside = 0;
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    outside |= (((code[v] & 0x0f0f0f0fu) + bias) & even) | ((((code[v] >> 4) & 0x0f0f0f0fu) + bias) & odd);
  }
  if (outside != 0) {
    trainingTrap();
  }
}

// 検査つきのビルドで使う、上限つきの配列（AttentionForward・AttentionBackward と同じ並び。codes は 1 組 1 ワード）。
struct ForwardArgs {
  AttentionShape shape;
  Span<const uint16_t> query;
  Span<const uint16_t> keyValue;
  Span<const float> mask;
  Span<const uint32_t> codes;
  Span<const float> table;
  Span<uint16_t> out;
  Span<float> outFloat;
  Span<float> logSumExp;
  int fixed;
};

struct BackwardArgs {
  AttentionShape shape;
  Span<const uint16_t> query;
  Span<const uint16_t> keyValue;
  Span<const float> mask;
  Span<const uint32_t> codes;
  Span<const float> table;
  Span<const float> outFloat;
  Span<const float> logSumExp;
  Span<const uint16_t> gradOut;
  Span<uint16_t> gradQuery;
  Span<uint16_t> gradKeyValue;
  Span<float> gradTable;
  int fixed;
};

// 行の範囲。一様な先頭のアドレスと、lane ごとの 32 bit のオフセットで読み書きする（グローバルメモリの読み書きが、ウェーブで一様なレジスタの先頭と lane ごとのレジスタの
// 32 bit のバイトオフセットになる）。検査つきのビルドでは、範囲 [first, first + size) が span の中にあるかを rowOf が確かめ、
// 読み書きのたびに lane の [offset, offset + count) が範囲の中にあるかを at が確かめる（lane のオフセットは範囲に収めた添字から作るので、
// ふつうは収まる）。通常のビルドでは size は最適化で消える。
template <class T>
struct Row {
  T* data;
  size_t size;

  __device__ __forceinline__ T* at(uint32_t offset, uint32_t count, Bounds& bounds) const {
    using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
    const uint32_t checked = static_cast<uint32_t>(bounds.index(offset, count, size));
    return reinterpret_cast<T*>(reinterpret_cast<Byte*>(data) + checked * static_cast<uint32_t>(sizeof(T)));
  }
};

// span の [first, first + size) の範囲。
template <class T>
__device__ __forceinline__ Row<T> rowOf(Span<T> span, size_t first, size_t size, Bounds& bounds) {
  return Row<T>{span.data + bounds.index(first, size, span.limit), size};
}

// キーのタイル kt の有効・無効を表す 16 bit（キー 16kt + m が bit m。マスクが 0 でないキーが有効で、keys より後ろは無効）を共有メモリの
// keyBits[kt] に置く（タイルをウェーブで分け、ballot の下位 16 bit を使う）。
__device__ __forceinline__ void stageKeyBits(uint32_t* keyBits, Row<const float> maskRow, int keys, int wave, int waves,
                                             int lane, Bounds& bounds) {
  const int blocks = (keys + 15) / 16;
  for (int kt = wave; kt < blocks; kt += waves) {
    const int key = 16 * kt + (lane & 15);
    const bool present = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    const uint32_t bits = vote(present) & 0xffffu;
    if (lane == 0) {
      keyBits[kt] = bits;
    }
  }
}

// 共有メモリ。V^T は推論の Attention と同じ並び: ヘッド h の行 16h + r(d)、キー 16kt + m は列 16kt + κ(m)（無効なキーは 0）。コードの写しはウェーブごとにキーの
// タイル 1 つ分で、クエリ n の行の列 κ(m) にキー m のコードを置く（lane が持つ 8 キー 2v + half が 8·half から続く）。
template <int kBlocks>
struct alignas(16) ForwardShared {
  uint16_t valueT[2 * kF][16 * kBlocks + 8];
  float rho[2][kRhoEntries];             // 2 ヘッドの ρ の列（stageRho）
  float2 bias[kTables * kTableEntries];  // 2 ヘッドの組の表
  uint32_t codes[kBlocks][16][16];
  uint32_t keyBits[kBlocks];
};

template <int kBlocks, bool Causal>
__global__ __launch_bounds__(kWave* kBlocks) void forwardKernel(ForwardArgs args) {
  static_assert(sizeof(ForwardShared<kBlocks>) <= device::kStaticSharedBytes,
                "the block exceeds the static shared memory of the device");
  __shared__ ForwardShared<kBlocks> shared;

  const AttentionShape shape = args.shape;
  const int pair = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int waves = threads / kWave;
  const int lane = tid % kWave;
  const int wave = uniform(tid / kWave);
  const int column = fragmentColumnOfLane(lane);  // A・B のフラグメントの行、C のフラグメントの列
  const int half = fragmentParityOfLane(lane);    // C のフラグメントで奇数の行を持つ lane
  const int width = shape.heads * kF;
  const int keys = shape.keys;
  const int queries = shape.queries;
  const int blocks = (keys + 15) / 16;
  const float scale = shape.scale * kLog2e;
  Bounds bounds;
  // 行の範囲（クエリとキー・バリューは、ヘッドの対の列から行の終わりまで）。
  const int columns = 2 * pair * kF;
  const Row<const uint16_t> queryRow =
      rowOf(args.query, static_cast<size_t>(row) * queries * width + columns, queries * width - columns, bounds);
  const Row<const uint16_t> keyRow =
      rowOf(args.keyValue, static_cast<size_t>(row) * keys * 2 * width + columns, keys * 2 * width - columns, bounds);
  const Row<const uint32_t> codeRow = rowOf(args.codes, static_cast<size_t>(row) * shape.codeQueries * keys,
                                            static_cast<size_t>(shape.codeQueries) * keys, bounds);
  const Row<const float> maskRow = rowOf(args.mask, static_cast<size_t>(row) * keys, keys, bounds);

  // 0. 2 ヘッドの ρ の列、キーのタイルの有効・無効、V^T を置く（キーのタイルとヘッドの組をウェーブで分ける）。バリアの後に、ρ の列から
  //    2 ヘッドの組の表を作る。
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    stageRho(shared.rho[h], shape, args.table, 2 * pair + h, tid, threads, bounds);
  }
  stageKeyBits(shared.keyBits, maskRow, keys, wave, waves, lane, bounds);
  for (int unit = wave; unit < 2 * blocks; unit += waves) {
    const int kt = unit >> 1;
    const int h = unit & 1;
    const int key = 16 * kt + column;
    const bool present = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    u32x4 values = *reinterpret_cast<const u32x4*>(
        keyRow.at(static_cast<uint32_t>(min(key, keys - 1) * 2 * width + width + h * kF + 8 * half), 8, bounds));
    if (!present) {
      values = u32x4{0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      shared.valueT[kF * h + featureRow(8 * half + i)][16 * kt + fragmentColumn(column)] =
          static_cast<uint16_t>(values[i >> 1] >> (16 * (i & 1)));
    }
  }
  shareLocalWrites();
  for (int e = tid; e < kTables * kTableEntries; e += threads) {
    shared.bias[e] = make_float2(pairEntry(shared.rho[0], e), pairEntry(shared.rho[1], e));
  }
  shareLocalWrites();

  // 1. ウェーブ qt がクエリのタイル qt を受け持つ。lane はクエリ 16qt + column のキー 16kt + 2v + half（v < 8）を持つ。コード（行
  //    2t + half。キー column の 16 lane で 64 バイト続く）と K の行は、次のキーのタイルの分を先読みする。
  const int qt = wave;
  const int query = 16 * qt + column;
  bf16x16 queryFragment[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t at = static_cast<uint32_t>(min(query, queries - 1) * width + h * kF);
    queryFragment[h] = fragment(queryRow.at(at, kF, bounds));
  }
  auto loadBlock = [&](int kt, uint32_t (&code)[8], bf16x16(&key)[2]) {
    const int j = min(16 * kt + column, keys - 1);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      code[t] = *codeRow.at(static_cast<uint32_t>(min(16 * qt + 2 * t + half, queries - 1) * keys + j), 1, bounds);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      key[h] = fragment(keyRow.at(static_cast<uint32_t>(j * 2 * width + h * kF), kF, bounds));
    }
  };
  uint32_t nextCode[8];
  bf16x16 nextKey[2];
  loadBlock(0, nextCode, nextKey);
  float running[2] = {-INFINITY, -INFINITY};
  float total[2] = {0.0f, 0.0f};
  f32x8 out[2] = {f32x8{}, f32x8{}};
  for (int kt = 0; kt < blocks; ++kt) {
    uint32_t staged[8];
    bf16x16 keyFragment[2];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      staged[t] = nextCode[t];
    }
    keyFragment[0] = nextKey[0];
    keyFragment[1] = nextKey[1];
    if (kt + 1 < blocks) {
      loadBlock(kt + 1, nextCode, nextKey);
    }
    const uint32_t keyBits = shared.keyBits[kt];
    if (!Causal && keyBits == 0) continue;
    // コードをクエリの行に並べ替える（ウェーブの中で共有メモリを通す）。
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      shared.codes[wave][2 * t + half][fragmentColumn(column)] = staged[t];
    }
    waveSync();
    uint32_t code[8];
    unpackEight(*reinterpret_cast<const u32x4*>(&shared.codes[wave][column][8 * half]),
                *reinterpret_cast<const u32x4*>(&shared.codes[wave][column][8 * half + 4]), code);
    if (shape.entries < kAttentionMaxEntries) {
      checkCodes(code, shape.segments, shape.entries);
    }

    f32x8 score[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      score[h] = mma(keyFragment[h], queryFragment[h], f32x8{});
    }
    float logit[2][8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      // 無効なキーは −∞。表を引く読み込みを分岐の中に入れないよう、条件で選ぶのはタイルの積和の値と定数だけにして、−∞ は足して付ける。
      bool present = ((keyBits >> (accumulatorRow(v, half))) & 1u) != 0;
      if constexpr (Causal) {
        const int key = 16 * kt + accumulatorRow(v, half);
        present = key < keys && causalKey(present, args.fixed, query, key);
      }
      const float floor = present ? 0.0f : -INFINITY;
      const float2 bias = pairBias(code[v], shared.bias);
      logit[0][v] = fusedMultiplyAdd(present ? score[0][v] : 0.0f, scale, bias.x) + floor;
      logit[1][v] = fusedMultiplyAdd(present ? score[1][v] : 0.0f, scale, bias.y) + floor;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float block = logit[h][0];
#pragma unroll
      for (int v = 1; v < 8; ++v) {
        block = fmaxf(block, logit[h][v]);
      }
      block = fmaxf(block, swapHalves(block));
      const float next = fmaxf(running[h], block);
      const float origin = Causal && next == -INFINITY ? 0.0f : next;
      const float alpha = fastExp2(running[h] - origin);
      running[h] = next;
      float e[8];
      float sum = 0.0f;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        e[v] = fastExp2(logit[h][v] - origin);
        sum += e[v];
      }
      total[h] = fusedMultiplyAdd(total[h], alpha, sum);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        out[h][v] *= alpha;
      }
      bf16x16 terms[3];
      splitFragments(e, terms);
      out[h] = mma3(fragment(&shared.valueT[kF * h + column][16 * kt]), terms, out[h]);
    }
  }

  // 2. 出力。lane はクエリ query の次元 8·half〜8·half + 7 を書く（BF16 は 16 バイト、FP32 は 32 バイト）。半分どうしで Σ を
  //    足す処理（16 lane 離れた相方との交換）は、ウェーブの全 lane がそろっている分岐の外で行う。
  float sums[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sums[h] = total[h] + swapHalves(total[h]);
  }
  if (query < queries) {
    const size_t first = static_cast<size_t>(row) * queries * width + columns;
    const Row<uint16_t> outRow = rowOf(args.out, first, queries * width - columns, bounds);
    const Row<float> outFloatRow = rowOf(args.outFloat, first, queries * width - columns, bounds);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sum = sums[h];
      const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
      float o[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        o[v] = out[h][v] * inverse;
      }
      const uint32_t at = static_cast<uint32_t>(query * width + h * kF + 8 * half);
      *reinterpret_cast<u32x4*>(outRow.at(at, 8, bounds)) = pack8(o);
      float* outFloat = outFloatRow.at(at, 8, bounds);
      *reinterpret_cast<f32x4*>(outFloat) = f32x4{o[0], o[1], o[2], o[3]};
      *reinterpret_cast<f32x4*>(outFloat + 4) = f32x4{o[4], o[5], o[6], o[7]};
      if (half == 0) {
        const size_t lse = (static_cast<size_t>(row) * shape.heads + 2 * pair + h) * queries + query;
        args.logSumExp.data[bounds.index(lse, 1, args.logSumExp.limit)] =
            sum > 0.0f ? (running[h] + log2f(sum)) * kLn2 : 0.0f;
      }
    }
  }
  bounds.trap();
}

// dS とコードの共有メモリの行のストライド（ワード）: 16·blocks 以上で、64 を法として 36（100 / 164 / 164）。バンクが 64 のとき、段 B の
// 16 バイトの読み込み（lane が行 column の 8·half から読む）は 16 行 × 2 つの半分で 2-way（最小）になり、段 A の書き込み（行 2v + half の
// 16 列）は 2 つの半分のバンクが重ならない。行の先頭は 16 バイトにそろう。
constexpr int scoreStride(int blocks) { return 16 * blocks + ((36 - 16 * blocks) % 64 + 64) % 64; }
static_assert(scoreStride(kSmallBlocks) == 100 && scoreStride(kMediumBlocks) == 164 && scoreStride(kLargeBlocks) == 164,
              "the score rows are 36 words apart modulo 64 banks");

// 共有メモリ。dS とコードは、クエリのタイルの行 [クエリ][キー] に置く（行のストライドは scoreStride）。ρ の列は stageRho が置く。K^T はキーの
// タイルごとの A のフラグメント（行 r(d)、キーは自然な順）、Q^T・dO^T はクエリのタイルの A のフラグメント（行 r(d)、クエリ m は列
// κ(m)）で、無効なキーと生きていないクエリは 0。lse₂・δ はクエリのタイルごとに列 κ(m) に置く（生きていないクエリは 0）。queryBits は
// タイルごとの生きたクエリを表す 16 bit（列 κ(m) の bit がクエリ m）。ρ の勾配の部分和は、ウェーブ w が、セグメント 2w + p の
// コード c ごとに lane ごとに持つ（セグメントが segments 以上のフィールドは書き出さない）。
template <int kBlocks>
struct alignas(16) BackwardShared {
  // ρ からバイアス表を作り終えた後、同じ領域にスコアの勾配を置く。
  union {
    float rho[kRhoEntries];
    float gradScore[16][scoreStride(kBlocks)];
  };
  uint32_t codes[16][scoreStride(kBlocks)];
  uint16_t keyT[kBlocks][kF][16];
  uint16_t queryT[kF][16];
  uint16_t gradOutT[kF][16];
  float logSumExp[kBlocks][16];
  float delta[kBlocks][16];
  float bias[kTables * kTableEntries];
  float bins[kScatterWaves][2][kAttentionMaxEntries][kWave];
  uint32_t keyBits[kBlocks];
  uint32_t queryBits[kBlocks];
};

// 列 κ のクエリが生きていれば true。bits はクエリのタイルごとの生きたクエリの bit（queryBits）。
__device__ __forceinline__ bool liveQuery(uint32_t bits, int k) { return ((bits >> k) & 1u) != 0; }

// ウェーブの 2 セグメントの部分和 bins[p][c][lane] に、lane が持つ連続する 2 キーの dS を足す。セグメントごとに順に足し（コードが同じなら
// 2 つ目は 1 つ目の結果に足す。1 つずつ読み書きするのとビットで同じ）、2 セグメント分の 4 つの読み込みはまとめて出して待ち時間を重ねる。
// shift はセグメント 2w のコードが始まる bit の位置。
__device__ __forceinline__ void accumulatePair(float (*bins)[kAttentionMaxEntries][kWave], int lane, int shift,
                                               uint32_t first, uint32_t second, float x0, float x1) {
  uint32_t c0[2];
  uint32_t c1[2];
  float v0[2];
  float v1[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    c0[p] = (first >> (shift + 4 * p)) & 15u;
    c1[p] = (second >> (shift + 4 * p)) & 15u;
    v0[p] = bins[p][c0[p]][lane];
    v1[p] = bins[p][c1[p]][lane];
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    v0[p] += x0;
    v1[p] = (c1[p] == c0[p] ? v0[p] : v1[p]) + x1;
    bins[p][c0[p]][lane] = v0[p];
    bins[p][c1[p]][lane] = v1[p];
  }
}

// 勾配の 0 の符号を +0 にそろえる。x + (+0) は −0 だけを +0 にし、ほかの値（非正規数も）は変えない（IEEE の加算。nsz の最適化は
// 使っていないので消されない）。無効なキーの dk・dv は積が全て 0 の和で、その 0 の符号は和に入る 0 の符号（タイルの積和の結果）で決まる。
// そろえると、ちょうど 0 の和は常に +0 で書かれる。ただし −0 が全く出ないわけではない。0 でない和の積 x·scale がアンダーフローすると
// scaledPositiveZero は符号つきの 0 を返し、pack8 は絶対値の小さい負の値を BF16 の −0 に丸める。この和はバッチの組み方に依らず同じ値
// なので、行どうしの独立は保たれる。
__device__ __forceinline__ float positiveZero(float x) { return x + 0.0f; }

// x·scale の 0 の符号を +0 にそろえる。積と +0 の和を 1 回で丸める（fma(x, scale, +0)）。
__device__ __forceinline__ float scaledPositiveZero(float x, float scale) { return fusedMultiplyAdd(scale, x, 0.0f); }

// 段 A でクエリのタイル 1 つに使う、グローバルメモリからの入力（lane のキーに対する 8 クエリのコードと、Q・dO の行のフラグメント）。次の
// タイルの分を先読みする。
struct QueryBlockInputs {
  uint32_t code[8];
  bf16x16 query;
  bf16x16 gradOut;
};

// クエリのタイルの Q^T・dO^T の元（ウェーブ 1 つ分。lane はクエリ 16qt + (lane & 15) の次元 8·half〜+7 を、q と do で 1 つずつ持つ）。
struct QueryRows {
  u32x4 query;
  u32x4 gradOut;
};

// クエリのタイル qt の q・do の行をグローバルメモリから読む（どちらも一様な行の先頭と lane の 32 bit のオフセットで読む）。範囲外のクエリは
// 最後の行を読み、生きていないクエリを 0 にするのは storeQueries が行う。こうすると、読み込みと共有メモリに置く処理の間に挟んだ計算で、
// グローバルメモリの待ち時間を隠せる。
__device__ __forceinline__ QueryRows loadQueries(Row<const uint16_t> queryRow, Row<const uint16_t> gradOutRow,
                                                 int queries, int width, int qt, int lane, Bounds& bounds) {
  const uint32_t at = static_cast<uint32_t>(min(16 * qt + (lane & 15), queries - 1) * width + 8 * (lane >> 4));
  return QueryRows{*reinterpret_cast<const u32x4*>(queryRow.at(at, 8, bounds)),
                   *reinterpret_cast<const u32x4*>(gradOutRow.at(at, 8, bounds))};
}

// loadQueries で読んだ行を共有メモリの Q^T・dO^T に置く（行 r(d)、クエリ m は列 κ(m)）。生きていないクエリ（範囲外を含む。queryBits の bit が
// 0）は q も do も 0 にする。dk^T += Q^T·dS の K の位置で、dS = +0 の相手の q が、トークンの軸の長さ S に依らず同じ
// +0 になるようにするため。
template <int kBlocks>
__device__ __forceinline__ void storeQueries(BackwardShared<kBlocks>& shared, QueryRows rows, int qt, int lane) {
  const int m = fragmentColumnOfLane(lane);
  const int half = fragmentParityOfLane(lane);
  if (!liveQuery(shared.queryBits[qt], fragmentColumn(m))) {
    rows.query = rows.gradOut = u32x4{0u, 0u, 0u, 0u};
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int shift = 16 * (i & 1);
    shared.queryT[featureRow(8 * half + i)][fragmentColumn(m)] = static_cast<uint16_t>(rows.query[i >> 1] >> shift);
    shared.gradOutT[featureRow(8 * half + i)][fragmentColumn(m)] = static_cast<uint16_t>(rows.gradOut[i >> 1] >> shift);
  }
}

template <int kBlocks, bool Causal>
__global__ __launch_bounds__(kWave* kBlocks) void backwardKernel(BackwardArgs args) {
  static_assert(kBlocks >= kMinBackwardWaves, "stage B needs the scatter waves and the service wave");
  static_assert(sizeof(BackwardShared<kBlocks>) <= device::kStaticSharedBytes,
                "the block exceeds the static shared memory of the device");
  __shared__ BackwardShared<kBlocks> shared;

  const AttentionShape shape = args.shape;
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int lane = tid % kWave;
  const int wave = uniform(tid / kWave);
  const int column = fragmentColumnOfLane(lane);
  const int half = fragmentParityOfLane(lane);
  const int width = shape.heads * kF;
  const int keys = shape.keys;
  const int queries = shape.queries;
  const int keyBlocks = (keys + 15) / 16;
  const int queryBlocks = (queries + 15) / 16;
  const float scale = shape.scale * kLog2e;
  Bounds bounds;
  // 行の範囲（クエリとキー・バリューは、ヘッドの列から行の終わりまで）。
  const int columns = head * kF;
  const size_t queryFirst = static_cast<size_t>(row) * queries * width + columns;
  const size_t querySize = static_cast<size_t>(queries) * width - columns;
  const size_t keyFirst = static_cast<size_t>(row) * keys * 2 * width + columns;
  const size_t keySize = static_cast<size_t>(keys) * 2 * width - columns;
  const Row<const uint16_t> queryRow = rowOf(args.query, queryFirst, querySize, bounds);
  const Row<const uint16_t> gradOutRow = rowOf(args.gradOut, queryFirst, querySize, bounds);
  const Row<const float> outFloatRow = rowOf(args.outFloat, queryFirst, querySize, bounds);
  const Row<const float> logSumExpRow =
      rowOf(args.logSumExp, (static_cast<size_t>(row) * shape.heads + head) * queries, queries, bounds);
  const Row<const uint16_t> keyRow = rowOf(args.keyValue, keyFirst, keySize, bounds);
  const Row<const uint32_t> codeRow = rowOf(args.codes, static_cast<size_t>(row) * shape.codeQueries * keys,
                                            static_cast<size_t>(shape.codeQueries) * keys, bounds);
  const Row<const float> maskRow = rowOf(args.mask, static_cast<size_t>(row) * keys, keys, bounds);
  const Row<uint16_t> gradQueryRow = rowOf(args.gradQuery, queryFirst, querySize, bounds);

  // 0. ρ の列、部分和の 0 クリア、生きたクエリの bit と lse₂・δ（生きていないクエリは 0）、キーのタイルの有効・無効、K^T（ウェーブごとに
  //    キーのタイル 1 つ）を置き、最初の q・do の行を読む。バリアの後に、最初の Q^T・dO^T（生きたクエリの bit を使う）を置き、ρ の列から組の表を作る。
  stageRho(shared.rho, shape, args.table, head, tid, threads, bounds);
  for (int e = tid; e < kScatterWaves * 2 * kAttentionMaxEntries * kWave; e += threads) {
    (&shared.bins[0][0][0][0])[e] = 0.0f;
  }
  // e の lane は e mod 32（threads は 32 の倍数）なので、ウェーブの下の 16 lane がタイル e >> 4 の列 κ = 0〜15 を、上の 16 lane が次の
  // タイルを持つ。
  for (int e = tid; e < 16 * queryBlocks; e += threads) {
    const int qt = e >> 4;
    const int k = e & 15;
    const int query = 16 * qt + 2 * (k & 7) + (k >> 3);  // κ(m) = k になるクエリ m
    float lse = 0.0f;
    float delta = 0.0f;
    uint32_t bits = 0;  // do の BF16 のビットの OR（符号を除いて 0 でなければ生きている。NaN・Inf も生きている）
    if (query < queries) {
      const uint16_t* gradOut = gradOutRow.at(static_cast<uint32_t>(query * width), kF, bounds);
      const float* outFloat = outFloatRow.at(static_cast<uint32_t>(query * width), kF, bounds);
#pragma unroll
      for (int d = 0; d < kF; ++d) {
        bits |= gradOut[d];
        delta = fusedMultiplyAdd(bf16Float(gradOut[d]), outFloat[d], delta);
      }
      lse = *logSumExpRow.at(static_cast<uint32_t>(query), 1, bounds) * kLog2e;
    }
    const bool live = (bits & 0x7fffu) != 0;
    shared.logSumExp[qt][k] = live ? lse : 0.0f;
    shared.delta[qt][k] = live ? delta : 0.0f;
    const uint32_t ballot = vote(live);
    if (k == 0) {
      shared.queryBits[qt] = (ballot >> (lane & 16)) & 0xffffu;
    }
  }
  stageKeyBits(shared.keyBits, maskRow, keys, wave, threads / kWave, lane, bounds);
  // 段 A のウェーブ（kt = ウェーブの番号）のキーの行: S・dP の B のフラグメント（レジスタに残す）と K^T。
  const int kt = wave;
  const int key = 16 * kt + column;
  const uint32_t keyAt = static_cast<uint32_t>(min(key, keys - 1) * 2 * width);
  bool keyPresent = false;
  bf16x16 keyFragment = {};
  bf16x16 valueFragment = {};
  if (kt < keyBlocks) {
    keyPresent = key < keys && *maskRow.at(static_cast<uint32_t>(min(key, keys - 1)), 1, bounds) != 0.0f;
    keyFragment = fragment(keyRow.at(keyAt, kF, bounds));
    valueFragment = fragment(keyRow.at(keyAt + width, kF, bounds));
    u32x4 values = *reinterpret_cast<const u32x4*>(keyRow.at(keyAt + 8 * half, 8, bounds));
    if (!keyPresent) {
      values = u32x4{0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      shared.keyT[kt][featureRow(8 * half + i)][column] = static_cast<uint16_t>(values[i >> 1] >> (16 * (i & 1)));
    }
  }
  QueryRows firstRows = {};
  if (wave == kServiceWave) {
    firstRows = loadQueries(queryRow, gradOutRow, queries, width, 0, lane, bounds);
  }
  auto loadInputs = [&](int qt, QueryBlockInputs& in) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      in.code[v] = *codeRow.at(
          static_cast<uint32_t>(min(16 * qt + accumulatorRow(v, half), queries - 1) * keys + min(key, keys - 1)), 1,
          bounds);
    }
    const uint32_t at = static_cast<uint32_t>(min(16 * qt + column, queries - 1) * width);
    in.query = fragment(queryRow.at(at, kF, bounds));
    in.gradOut = fragment(gradOutRow.at(at, kF, bounds));
  };
  QueryBlockInputs next;
  if (kt < keyBlocks) {
    loadInputs(0, next);
  }
  shareLocalWrites();
  if (wave == kServiceWave) {
    storeQueries(shared, firstRows, 0, lane);
  }
  for (int e = tid; e < kTables * kTableEntries; e += threads) {
    shared.bias[e] = pairEntry(shared.rho, e);
  }
  // 計算するクエリのタイルは、生きたクエリを含む最後のタイルまで（ワークグループで一様な値。後ろのタイルを落とすだけなので、段 A の次の
  // タイルの先読みと、段 B の次の Q^T・dO^T の流れは変わらない）。
  int liveBlocks = 0;
  for (int qt = 0; qt < queryBlocks; ++qt) {
    liveBlocks = shared.queryBits[qt] != 0 ? qt + 1 : liveBlocks;
  }
  liveBlocks = uniform(liveBlocks);
  shareLocalWrites();
  const bool ownsKeys = kt < keyBlocks && (shared.keyBits[kt] != 0 || (Causal && 16 * (kt + 1) > args.fixed));

  f32x8 gradKeyT = {};
  f32x8 gradValueT = {};
  for (int qt = 0; qt < liveBlocks; ++qt) {
    const uint32_t liveBits = shared.queryBits[qt];
    // 段 A: lane はキー key のクエリ 16qt + 2v + half（v < 8。列 κ = v + 8·half）を持つ。
    if (ownsKeys) {
      const QueryBlockInputs in = next;
      if (qt + 1 < liveBlocks) {
        loadInputs(qt + 1, next);
      }
      if (shape.entries < kAttentionMaxEntries) {
        checkCodes(in.code, shape.segments, shape.entries);
      }
      const bf16x16 queryT = fragment(&shared.queryT[column][0]);
      const bf16x16 gradOutT = fragment(&shared.gradOutT[column][0]);
      float lse[8];
      float delta[8];
      unpackEight(*reinterpret_cast<const f32x4*>(&shared.logSumExp[qt][8 * half]),
                  *reinterpret_cast<const f32x4*>(&shared.logSumExp[qt][8 * half + 4]), lse);
      unpackEight(*reinterpret_cast<const f32x4*>(&shared.delta[qt][8 * half]),
                  *reinterpret_cast<const f32x4*>(&shared.delta[qt][8 * half + 4]), delta);

      const f32x8 score = mma(in.query, keyFragment, f32x8{});
      const f32x8 gradProbability = mma(in.gradOut, valueFragment, f32x8{});
      float probability[8];
      float difference[8];
      float gradScore[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        // 無効なキーと生きていないクエリは P = +0、dS = +0·(0 − 0) = +0（順伝播と同じく −∞ を足して付け、表を引く読み込みを分岐の中に
        // 入れない）。
        bool present = keyPresent;
        if constexpr (Causal)
          present = key < keys && causalKey(keyPresent, args.fixed, 16 * qt + accumulatorRow(v, half), key);
        const bool valid = present && liveQuery(liveBits, v + 8 * half);
        const float floor = valid ? 0.0f : -INFINITY;
        const float logit = fusedMultiplyAdd(valid ? score[v] : 0.0f, scale, pairBias(in.code[v], shared.bias));
        probability[v] = fastExp2((logit - lse[v]) + floor);
        difference[v] = (valid ? gradProbability[v] : 0.0f) - delta[v];
        gradScore[v] = training::scoreGradient(probability[v], valid ? gradProbability[v] : 0.0f, delta[v]);
      }
      bf16x16 terms[3];
      splitFragments(probability, terms);
      gradValueT = mma3(gradOutT, terms, gradValueT);
      splitProductFragments(probability, difference, terms);
      gradKeyT = mma3(queryT, terms, gradKeyT);
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        shared.gradScore[accumulatorRow(v, half)][key] = gradScore[v];
        shared.codes[accumulatorRow(v, half)][key] = in.code[v];
      }
    }
    shareLocalWrites();

    // 段 B。
    if (wave < kScatterWaves) {
      // lane はクエリ 16qt + column のキー 16k + 8·half〜+7 の dS を、セグメント 2w と 2w + 1（w はウェーブの番号）の部分和（lane ごと）に
      // 足す。足すのは生きたクエリだけ。セグメントが segments 以上の部分和にも足すが、書き出さない（本体を分岐なしに保つため）。
      if (liveQuery(liveBits, fragmentColumn(column))) {
        for (int k = 0; k < keyBlocks; ++k) {
          if (shared.keyBits[k] == 0) {
            continue;
          }
          const int first = 16 * k + 8 * half;
          float x[8];
          uint32_t c[8];
          unpackEight(*reinterpret_cast<const f32x4*>(&shared.gradScore[column][first]),
                      *reinterpret_cast<const f32x4*>(&shared.gradScore[column][first + 4]), x);
          unpackEight(*reinterpret_cast<const u32x4*>(&shared.codes[column][first]),
                      *reinterpret_cast<const u32x4*>(&shared.codes[column][first + 4]), c);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            accumulatePair(shared.bins[wave], lane, 8 * wave, c[e], c[e + 1], x[e], x[e + 1]);
          }
        }
      }
    } else if (wave == kServiceWave) {
      // 次のクエリのタイルの q・do の行は dQ の前に読み（グローバルメモリの待ち時間を dQ の計算で隠す）、dQ の後で共有メモリに置く。
      const bool stagesNext = qt + 1 < liveBlocks;
      QueryRows nextRows = {};
      if (stagesNext) {
        nextRows = loadQueries(queryRow, gradOutRow, queries, width, qt + 1, lane, bounds);
      }
      // dQ^T = K^T·dS^T（キーのタイルの順）: lane はクエリ 16qt + column の次元 8·half〜+7 を持つ。
      f32x8 gradQueryT = {};
      for (int k = 0; k < keyBlocks; ++k) {
        if (shared.keyBits[k] == 0) {
          continue;
        }
        float own[8];
        unpackEight(*reinterpret_cast<const f32x4*>(&shared.gradScore[column][16 * k + 8 * half]),
                    *reinterpret_cast<const f32x4*>(&shared.gradScore[column][16 * k + 8 * half + 4]), own);
        bf16x16 terms[3];
        splitFragments(own, terms);
        gradQueryT = mma3(fragment(&shared.keyT[k][column][0]), terms, gradQueryT);
      }
      // 死んだクエリの dq はちょうど 0（dS = 0）なので +0 を書く。
      const int query = 16 * qt + column;
      if (query < queries) {
        const bool live = liveQuery(liveBits, fragmentColumn(column));
        float values[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          values[v] = live ? scaledPositiveZero(gradQueryT[v], shape.scale) : 0.0f;
        }
        *reinterpret_cast<u32x4*>(gradQueryRow.at(static_cast<uint32_t>(query * width + 8 * half), 8, bounds)) =
            pack8(values);
      }
      if (stagesNext) {
        storeQueries(shared, nextRows, qt + 1, lane);
      }
    }
    shareLocalWrites();
  }
  // 計算しなかった後ろのタイルのクエリ（全て死んでいる）の dq は +0（クエリ 1 つは 16 バイトの書き込み 2 つ）。
  for (int e = tid; e < 2 * (queries - 16 * liveBlocks); e += threads) {
    const int query = 16 * liveBlocks + (e >> 1);
    *reinterpret_cast<u32x4*>(gradQueryRow.at(static_cast<uint32_t>(query * width + 8 * (e & 1)), 8, bounds)) =
        u32x4{0u, 0u, 0u, 0u};
  }

  // 3. dk・dv と ρ の勾配の部分和を書く。dk・dv はウェーブ kt がキーのタイル kt を受け持ち、lane はキー key の次元 8·half〜+7 を書く
  //    （無効なキーのタイルは 0 を書く）。ρ の勾配は、セグメント 2w + half（w はウェーブの番号）のコード column の部分和を、32 lane の順に足す。
  if (kt < keyBlocks && key < keys) {
    float gradKey[8];
    float gradValue[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      gradKey[v] = scaledPositiveZero(gradKeyT[v], shape.scale);
      gradValue[v] = positiveZero(gradValueT[v]);
    }
    const Row<uint16_t> gradKeyRow = rowOf(args.gradKeyValue, keyFirst, keySize, bounds);
    const uint32_t at = static_cast<uint32_t>(key * 2 * width + 8 * half);
    *reinterpret_cast<u32x4*>(gradKeyRow.at(at, 8, bounds)) = pack8(gradKey);
    *reinterpret_cast<u32x4*>(gradKeyRow.at(at + width, 8, bounds)) = pack8(gradValue);
  }
  if (wave < kScatterWaves) {
    const int s = 2 * wave + half;
    if (s < shape.segments && column < shape.entries) {
      float sum = 0.0f;
#pragma unroll
      for (int l = 0; l < kWave; l += 4) {
        const f32x4 part = *reinterpret_cast<const f32x4*>(&shared.bins[wave][half][column][l]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sum += part[i];
        }
      }
      const int tableRows = shape.segments * shape.entries;
      const size_t at = (static_cast<size_t>(row) * shape.heads + head) * tableRows + s * shape.entries + column;
      args.gradTable.data[bounds.index(at, 1, args.gradTable.limit)] = sum;
    }
  }
  bounds.trap();
}
}  // namespace
}  // namespace prism
