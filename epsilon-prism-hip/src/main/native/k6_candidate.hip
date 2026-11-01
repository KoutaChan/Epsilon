// K6（candidate）: 48 遷移の作業群（256 スレッド = 8 波）。eager の CandidateLayer（静的な部分 t_s は K1）と DiscardValue。
//
//   t0 = ⌊t_s + ⌊y_T[τ]·W_Eᵀ⌉⌉                  τ = 鳴いた後に捨てる牌、なければ主牌、なければ項なし
//   q  = ⌊⌊N_q(t0)⌉·W_qᵀ + b_q⌉
//   o  = ⌊Σ_j softmax_j((q·K_j + q·E_k[c_j])·¼)·(V_j + E_v[c_j])⌉   j は行の局・席・牌、FP32
//   z  = mlpStage(t0, o)                           （K3 と同じ式、候補の重み）
//   B  = Σ_g silu(⌊⌊N_B(z)⌉·W_Bᵀ + c_B⌉)·wB_g + Σ_k x^B_k·uBx_k   （FP32）
//
// 縮約は全て遷移の中（交差注意は鍵の順、B の和は列の決まった順）で、同じ遷移の z と B は塊の組み方に依らずビットで同じ。
// 塊は K3 と同じ 48 行（64 行より LDS が小さく、1 WGP に載る作業群が多い）。
// 遷移ごとの縮約の順（行列積の K の歩、RMSNorm の 4 スレッドの木、B の halfSum と波の順の和）は塊の行の数に依らない。交差注意は
// (頭, 遷移) の項目をスレッドに配る（attendPipelined）。RMSNorm の受け持ち（4 スレッドで 1 行）は 64 行ぶんあるので、段 3 は行 48
// 以上（波 6・7）を飛ばす（normalizeTile と同じ）。
// 行列積は全て段ずらし（tileEntry・query・discardHidden は gemmTile、mlpStage は計画の断片の順の写しを読む gemmFragmentsFrom。
// prism_hip_mlp.h）。t_s を最初に読み、t0 の epilogue を 1/rms と同じ受け持ちの 1 回に畳むので、tileEntry の段ずらしも
// spill しない。交差注意は鍵を kAhead 個先まで生の buffer で読む段ずらしの輪（attendPipelined）。FFN の h と B の silu は siluFast、
// 活性の丸めは bf16RoundActivation（NaN の選択なし）、待ち合わせは全て tile::barrier（LDS だけ。読む区画と書く区画が別という約束）。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

using tile::kChunk;
using tile::kThreads;
using tile::kWaves;

constexpr int kTile = 48;            // 作業群の遷移（MLP の段の行）
constexpr int kRowTiles = kTile / 16;
constexpr int kKeys = 39;            // 行の局・席・牌（入力の約束 1 + 4 + 34。起動のときに照らす）
constexpr int kAhead = 3;            // 交差注意で読み込みを先に出す鍵の数（5 にすると spill する）
constexpr int kMaxDiscardExact = 8;  // B の道の幅の上限（DiscardExact の 8 欄。起動のときに照らす）
static_assert(kKeys > 32 && kKeys <= 48, "the key codes are read as three 16-byte pieces");

template <int H>
struct alignas(16) CandidateShared {
  tile::MlpShared<H, kTile> mlp;
  float inverse[kTile];
  float partial[kWaves][kTile];
  int row[kTile];
  int tile[kTile];
};

// 交差注意の鍵 1 つの読み込み: 行の鍵の頭の 16 次元（K か V、BF16、b128 × 2）と、その符号の E_k か E_v の頭の 16 次元（FP32、
// b128 × 4）。
struct KeySlot {
  u32x4 key[2];
  f32x4 code[4];

  // 読み込みを出す。lane ごとに違う位置は行の鍵の先頭 rowByte と符号の E の行 codeRow（どちらも項目の頭の位置を含む。VGPR）、波で
  // 一様な位置は鍵の中の memoryByte（soffset）。
  __device__ __forceinline__ void issue(__amdgpu_buffer_rsrc_t memory, uint32_t rowByte, int memoryByte,
                                        __amdgpu_buffer_rsrc_t table, uint32_t codeRow) {
    key[0] = __builtin_amdgcn_raw_buffer_load_b128(memory, rowByte, memoryByte, 0);
    key[1] = __builtin_amdgcn_raw_buffer_load_b128(memory, rowByte, memoryByte + 16, 0);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      code[w] = __builtin_bit_cast(f32x4, __builtin_amdgcn_raw_buffer_load_b128(table, codeRow, 16 * w, 0));
    }
  }

  // q·K（v_dot2_f32_bf16: BF16 の対の積を FP32 で 1 本の鎖に足す）。
  __device__ __forceinline__ float keyDot(const u32x4 (&query)[2]) const {
    float dot = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        dot = __builtin_amdgcn_fdot2_f32_bf16(asPair(query[half][w]), asPair(key[half][w]), dot, false);
      }
    }
    return dot;
  }

  // q·E_k（FMA の 1 本の鎖）。
  __device__ __forceinline__ float codeDot(const float (&query)[2][8]) const {
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      dot = fmaf(query[d >> 3][d & 7], code[d >> 2][d & 3], dot);
    }
    return dot;
  }

  // value += p·(V + E_v)（16 次元、次元ごとに鍵の順の FMA の鎖）。
  __device__ __forceinline__ void accumulate(float p, float (&value)[2][8]) const {
    float v[2][8];
    unpack8(key[0], v[0]);
    unpack8(key[1], v[1]);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        value[half][d] = fmaf(p, v[half][d] + code[2 * half + (d >> 2)][d & 3], value[half][d]);
      }
    }
  }
};

// 鍵 j の符号の E の行（A 個の FP32）のバイトの位置。符号の 39 バイトは [0, 16)・[16, 32)・[39 − 16, 39) の 3 つの 16 バイトに
// 読んである。E の行ごとに照らす。
template <int A>
__device__ __forceinline__ uint32_t codeRow(const u32x4 (&codes)[3], int j, size_t limit, Bounds& bounds) {
  const int part = j < 16 ? 0 : j < 32 ? 1 : 2;
  const int byte = j - (part == 0 ? 0 : part == 1 ? 16 : kKeys - 16);
  const uint32_t code = (codes[part][byte >> 2] >> (8 * (byte & 3))) & 0xffu;
  return static_cast<uint32_t>(bounds.index(static_cast<size_t>(code) * A, A, limit) * sizeof(float));
}

// 段 5: 交差注意。(頭, 遷移) の kHeads·kTile 個の項目を item = tid + 256u（u = 0, 1）で配り、項目は頭 h = item / kTile、遷移
// i = item % kTile（波 0〜3 が 2 項目、波 4〜7 が 1 項目）。
//   ℓ_j = (q·K_j + q·E_k[c_j])·scale·log2e（v_dot2 の鎖と FMA の鎖）、p_j = 2^(ℓ_j − max)·(1 / Σ_j 2^(ℓ_j − max))、
//   o = ⌊Σ_j p_j·(V_j + E_v[c_j])⌉
// 項目の式と縮約の順は受け持ちに依らない。h は lane ごとに違うので、鍵の中の頭は rowByte に、E の行の中の頭は codeRow に足す（VGPR）。
// 鍵 j + kAhead の読み込みを出してから鍵 j を使う段ずらしの輪（kAhead + 1 の場所を回す。sched_barrier で区切らないと組み立て器が
// 読み込みを使う直前へ寄せる）。読み込みは生の buffer で、鍵ごとの 64 bit の番地を作らない（作ると LLVM が 39 組を項目の輪の外へ
// 持ち出して spill する）。項目の本体は別の関数に分けない: 分けると先に展開された鍵の符号の行の位置を LLVM が項目の輪の外へ持ち出し、
// ℓ の輪と o の輪で共有して spill する。
template <int H>
__device__ __forceinline__ void attendPipelined(const CandidateArgs& args, CandidateShared<H>& shared, int t0, int rows,
                                                int tid, Bounds& bounds) {
  constexpr int A = H / 2;
  constexpr int kHeads = A / 16;
  constexpr int kItems = kHeads * kTile;
  constexpr int kRounds = (kItems + kThreads - 1) / kThreads;
  constexpr int kKeyBytes = 2 * A * sizeof(uint16_t);   // MEMORY の鍵 1 つ（K と V）
  constexpr int kValueBytes = A * sizeof(uint16_t);     // 鍵の中の V
  constexpr int kHeadKeyBytes = 16 * sizeof(uint16_t);  // 鍵の K・V の中の頭 1 つ
  constexpr int kHeadCodeBytes = 16 * sizeof(float);    // E の行の中の頭 1 つ
  constexpr int kSlots = kAhead + 1;
  static_assert(kItems % kWave == 0, "a wave either has an item in a round or not");
  auto& inner = shared.mlp.inner;
  const float logitScale = args.scale * 1.4426950408889634f;  // softmax は exp2 で取る
  const __amdgpu_buffer_rsrc_t memory = bufferResource(args.memory.data);
  const __amdgpu_buffer_rsrc_t codeKey = bufferResource(args.codeKey.data);
  const __amdgpu_buffer_rsrc_t codeValue = bufferResource(args.codeValue.data);
#pragma unroll 1
  for (int u = 0; u < kRounds; ++u) {
    const int item = tid + kThreads * u;
    const int i = item % kTile;
    if (item >= kItems || i >= rows) {
      continue;
    }
    const int h = item / kTile;
    // 行の kKeys 個の鍵（K と V）は 1 つの区画として照らし、鍵の中の頭の位置を足す。
    const size_t keyAt =
        bounds.index(static_cast<size_t>(shared.row[i]) * kKeys * (2 * A), kKeys * (2 * A), args.memory.limit);
    const uint32_t rowByte = static_cast<uint32_t>(keyAt * sizeof(uint16_t)) + h * kHeadKeyBytes;
    // 鍵の符号（揃わない 16 バイトの読みを 3 つ。どれも遷移の欄の中）。
    const size_t codeAt = static_cast<size_t>(t0 + i) * kKeys;
    u32x4 codes[3];
    codes[0] = loadUnaligned16(args.code.data + bounds.index(codeAt, 16, args.code.limit));
    codes[1] = loadUnaligned16(args.code.data + bounds.index(codeAt + 16, 16, args.code.limit));
    codes[2] = loadUnaligned16(args.code.data + bounds.index(codeAt + kKeys - 16, 16, args.code.limit));
    const uint32_t tableLane = h * kHeadCodeBytes;  // E の行の中の頭
    u32x4 query[2];
    query[0] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16]);
    query[1] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16 + 8]);
    float q[2][8];
    unpack8(query[0], q[0]);
    unpack8(query[1], q[1]);

    // ℓ と最大。
    KeySlot ring[kSlots];
#pragma unroll
    for (int j = 0; j < kAhead; ++j) {
      ring[j].issue(memory, rowByte, j * kKeyBytes, codeKey, codeRow<A>(codes, j, args.codeKey.limit, bounds) + tableLane);
    }
    float logits[kKeys];
    float maximum = -INFINITY;
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      const int ahead = j + kAhead;
      if (ahead < kKeys) {
        ring[ahead % kSlots].issue(memory, rowByte, ahead * kKeyBytes, codeKey,
                                   codeRow<A>(codes, ahead, args.codeKey.limit, bounds) + tableLane);
      }
      __builtin_amdgcn_sched_barrier(0);
      const KeySlot& slot = ring[j % kSlots];
      logits[j] = (slot.keyDot(query) + slot.codeDot(q)) * logitScale;
      maximum = fmaxf(maximum, logits[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    float total = 0.0f;
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      logits[j] = __builtin_amdgcn_exp2f(logits[j] - maximum);
      total += logits[j];
    }
    const float inverse = 1.0f / total;

    // o = Σ_j p_j·(V_j + E_v[c_j])。
#pragma unroll
    for (int j = 0; j < kAhead; ++j) {
      ring[j].issue(memory, rowByte, j * kKeyBytes + kValueBytes, codeValue,
                    codeRow<A>(codes, j, args.codeValue.limit, bounds) + tableLane);
    }
    float value[2][8];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        value[half][d] = 0.0f;
      }
    }
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      const int ahead = j + kAhead;
      if (ahead < kKeys) {
        ring[ahead % kSlots].issue(memory, rowByte, ahead * kKeyBytes + kValueBytes, codeValue,
                                   codeRow<A>(codes, ahead, args.codeValue.limit, bounds) + tableLane);
      }
      __builtin_amdgcn_sched_barrier(0);
      ring[j % kSlots].accumulate(logits[j] * inverse, value);
      __builtin_amdgcn_sched_barrier(0);
    }
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}

template <int H>
__global__ __launch_bounds__(kThreads) void candidateKernel(CandidateArgs args) {
  constexpr int A = H / 2;
  constexpr int kColumns = H / 16 / kWaves;
  constexpr int kParts = H / 32;  // RMSNorm の受け持ち（4 スレッドで 1 行、H/4 列）の 16 バイトの数
  static_assert(A == kChunk, "the query and the attention output are staged in the hidden buffer");
  static_assert(H % kChunk == 0, "the query projection reads the stream in chunks");
  static_assert(sizeof(CandidateShared<H>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ CandidateShared<H> shared;
  auto& stream = shared.mlp.stream;
  auto& inner = shared.mlp.inner;

  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int half = lane >> 4;
  const int column = lane & 15;
  const int t0 = blockIdx.x * kTile;
  const int rows = min(kTile, args.transitions - t0);
  const TokenLayout layout = args.layout;
  Bounds bounds;

  // 段 3 の t_s は最初に読んでおく（RMSNorm の受け持ち。塊の外の行は最後の行に寄せて読み、使わない）。受け持ちは 64 行ぶんあり、
  // 行 kTile 以上（波 6・7）に仕事はない。
  const int normRow = tid >> 2;
  const int normCol = (tid & 3) * (H / 4);
  u32x4 staticPart[kParts];
  {
    const int r = min(normRow, rows - 1);
#pragma unroll
    for (int c = 0; c < kParts; ++c) {
      staticPart[c] = *reinterpret_cast<const u32x4*>(
          args.statics.data + bounds.index(static_cast<size_t>(t0 + r) * H + normCol + 8 * c, 8, args.statics.limit));
    }
  }

  // 0. 遷移の行（行の最初の遷移の 2 分探索）と牌 τ（ID、0 はなし）。
  if (tid < kTile) {
    int row = 0;
    int tileId = 0;
    if (tid < rows) {
      const int t = t0 + tid;
      int lo = 0;
      int hi = layout.rows - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        const int first = args.rowColumn.data[bounds.index(
            static_cast<size_t>(mid) * args.rowWidth + args.rowFirstTransition, 1, args.rowColumn.limit)];
        if (first <= t) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      row = lo;
      const size_t at = static_cast<size_t>(t) * args.categoryWidth;
      const int callDiscard = args.category.data[bounds.index(at + args.callDiscard, 1, args.category.limit)];
      const int mainTile = args.category.data[bounds.index(at + args.mainTile, 1, args.category.limit)];
      tileId = callDiscard > 0 ? callDiscard : mainTile;
    }
    shared.row[tid] = row;
    shared.tile[tid] = tileId;
  }
  tile::barrier();

  // 1. y_T[τ] を stream へ（τ がない遷移と塊の外は 0）。読み込みは τ がなければ y の先頭の行に寄せ、分岐なしで一度に飛ばす。
  {
    constexpr int kVectors = H / 8;
    constexpr int kPasses = kTile * kVectors / kThreads;
    const int col = (tid % kVectors) * 8;
    u32x4 values[kPasses];
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int r = tile::tileRow<H, kTile>(pass, tid);
      const int tileId = shared.tile[r];
      const size_t token = tileId > 0 ? static_cast<size_t>(1 + layout.seats) * layout.rows +
                                            static_cast<size_t>(layout.tiles) * shared.row[r] + tileId - 1
                                      : 0;
      values[pass] = *reinterpret_cast<const u32x4*>(args.state.data + bounds.index(token * H + col, 8, args.state.limit));
    }
    const u32x4 zero = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int r = tile::tileRow<H, kTile>(pass, tid);
      *reinterpret_cast<u32x4*>(&stream[r][col]) = shared.tile[r] > 0 ? values[pass] : zero;
    }
  }
  tile::barrier();

  // 2. ⌊y_T·W_Eᵀ⌉（偏りなし）を C の断片の並びのまま stream へ。
  int output[kColumns];
#pragma unroll
  for (int j = 0; j < kColumns; ++j) {
    output[j] = (wave * kColumns + j) * 16 + column;
  }
  {
    f32x8 acc[kRowTiles][kColumns];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = f32x8{};
      }
    }
    tile::gemmTile<kColumns, H>(acc, stream, args.tileEntry.data, args.tileEntry.limit, H, 0, output, column, bounds);
    tile::barrier();
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
#pragma unroll
      for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          stream[m * 16 + 2 * v + half][output[j]] = bf16RoundActivation(acc[m][j][v]);
        }
      }
    }
  }
  tile::barrier();

  // 3. t0 = ⌊t_s + ⌊E⌉⌉ を RMSNorm の受け持ちでその場で作り（塊の外の行は t_s を 0 とする）、そのまま 1/rms(t0) を inverse へ。
  //    行 kTile 以上の受け持ちは飛ばす（4 スレッドの組ごとに飛ぶので inverseRms の木は欠けない）。
  if (normRow < kTile) {
    const bool real = normRow < rows;
    u32x4 part[kParts];
#pragma unroll
    for (int c = 0; c < kParts; ++c) {
      const u32x4 entry = *reinterpret_cast<const u32x4*>(&stream[normRow][normCol + 8 * c]);
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float baseLo = real ? bf16Float(static_cast<uint16_t>(staticPart[c][w] & 0xffffu)) : 0.0f;
        const float baseHi = real ? bf16Float(static_cast<uint16_t>(staticPart[c][w] >> 16)) : 0.0f;
        const uint16_t lo = bf16RoundActivation(baseLo + bf16Float(static_cast<uint16_t>(entry[w] & 0xffffu)));
        const uint16_t hi = bf16RoundActivation(baseHi + bf16Float(static_cast<uint16_t>(entry[w] >> 16)));
        part[c][w] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
      *reinterpret_cast<u32x4*>(&stream[normRow][normCol + 8 * c]) = part[c];
    }
    const float inverse = tile::inverseRms<H>(part, args.epsilon);
    if ((tid & 3) == 0) {
      shared.inverse[normRow] = inverse;
    }
  }
  tile::barrier();

  // 4. q = ⌊⌊N_q(t0)⌉·W_qᵀ + b_q⌉。正規化した t0 を 128 列ずつ inner に置き、K の順に累積する。
  {
    f32x8 q[kRowTiles][1];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
      q[m][0] = f32x8{};
    }
    const int units[1] = {wave * 16 + column};
    for (int k0 = 0; k0 < H; k0 += kChunk) {
      constexpr int kVectors = kChunk / 8;
#pragma unroll
      for (int pass = 0; pass < kTile * kVectors / kThreads; ++pass) {
        const int index = pass * kThreads + tid;
        const int r = index / kVectors;
        const int col = (index % kVectors) * 8;
        float values[8];
        unpack8(*reinterpret_cast<const u32x4*>(&stream[r][k0 + col]), values);
        const size_t g = bounds.index(k0 + col, 8, args.queryNorm.limit);
        const float inverse = shared.inverse[r];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          values[c] = (values[c] * inverse) * args.queryNorm.data[g + c];
        }
        *reinterpret_cast<u32x4*>(&inner[r][col]) = pack8Activation(values);
      }
      tile::barrier();
      tile::gemmTile<1, kChunk>(q, inner, args.query.data, args.query.limit, H, k0, units, column, bounds);
      tile::barrier();
    }
    const float bias = tile::load(args.queryBias.data, args.queryBias.limit, units[0], bounds);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        inner[m * 16 + 2 * v + half][units[0]] = bf16RoundActivation(q[m][0][v] + bias);
      }
    }
  }
  tile::barrier();

  // 5. 交差注意。⌊o⌉ を q の上に書く。
  attendPipelined<H>(args, shared, t0, rows, tid, bounds);
  tile::barrier();

  // 6. z = mlpStage(t0, o)（候補の O 射影と FFN）を書く。
  tile::mlpStage<H>(shared.mlp, args.weights, args.ffn, args.epsilon, bounds);
  tile::storeTile<H>(args.transition.data, args.transition.limit, stream, t0, rows, tid, bounds);
  tile::barrier();

  // 7. B。z を ⌊N_B(z)⌉ にその場で替え、G を 16 列ずつ波に配る。和は lane の列 → 16 lane の木 → 波の順。
  tile::normalizeTile<H>(stream, args.discardNorm.data, args.discardNorm.limit, args.epsilon, tid, bounds);
  tile::barrier();
  float rowSum[kRowTiles][8];
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      rowSum[m][v] = 0.0f;
    }
  }
  for (int n = wave; n < args.gate / 16; n += kWaves) {
    f32x8 u[kRowTiles][1];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
      u[m][0] = f32x8{};
    }
    const int units[1] = {n * 16 + column};
    tile::gemmTile<1, H>(u, stream, args.discardHidden.data, args.discardHidden.limit, H, 0, units, column, bounds);
    const float bias = tile::load(args.discardHiddenBias.data, args.discardHiddenBias.limit, units[0], bounds);
    const float weight = tile::load(args.discardOutput.data, args.discardOutput.limit, units[0], bounds);
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        rowSum[m][v] += siluFast(bf16Float(bf16RoundActivation(u[m][0][v] + bias))) * weight;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float sum = halfSum(rowSum[m][v]);
      if (column == 0) {
        shared.partial[wave][m * 16 + 2 * v + half] = sum;
      }
    }
  }
  tile::barrier();
  if (tid < rows) {
    float value = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
      value += shared.partial[w][tid];
    }
    // Σ_k x^B_k·uBx_k（k の順）。読み込みは kMaxDiscardExact 個を一度に飛ばす（幅の外は列 0 に寄せて読み、足さない）。
    const size_t exactAt = static_cast<size_t>(t0 + tid) * args.discardExactWidth;
    float features[kMaxDiscardExact];
    float weights[kMaxDiscardExact];
#pragma unroll
    for (int k = 0; k < kMaxDiscardExact; ++k) {
      const int at = k < args.discardExactWidth ? k : 0;
      features[k] = args.discardExact.data[bounds.index(exactAt + at, 1, args.discardExact.limit)];
      weights[k] = tile::load(args.discardWeights.data, args.discardWeights.limit, at, bounds);
    }
    float exact = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxDiscardExact; ++k) {
      if (k < args.discardExactWidth) {
        exact += features[k] * weights[k];
      }
    }
    args.value.data[bounds.index(t0 + tid, 1, args.value.limit)] = value + exact;
  }
  bounds.trap();
}

}  // namespace

void launchCandidate(int hidden, const CandidateArgs& args, hipStream_t stream) {
  if (args.transitions <= 0) {
    return;
  }
  if (args.layout.fixed() != kKeys) {
    throw std::invalid_argument("K6 is built for " + std::to_string(kKeys) + " keys per row, the schema has " +
                                std::to_string(args.layout.fixed()));
  }
  if (args.discardExactWidth > kMaxDiscardExact) {
    throw std::invalid_argument("K6 reads at most " + std::to_string(kMaxDiscardExact) +
                                " B path features per transition, the schema has " +
                                std::to_string(args.discardExactWidth));
  }
  const dim3 grid((args.transitions + kTile - 1) / kTile);
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(candidateKernel<256>, grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K6 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
