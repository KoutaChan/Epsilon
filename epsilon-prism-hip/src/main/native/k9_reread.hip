// K9（REREAD）: 候補層の 2 段目。ワークグループは遷移を 48 個ずつ受け持つ（256 スレッド = 8 ウェーブ）。eager の CandidateReread と
// DiscardValue と同じ値を計算する。
//
//   q2 = ⌊⌊N_q(z1)⌉·W_qᵀ + b_q⌉                          z1 は K6 が書いた 1 段目の出力
//   o2 = ⌊Σ_j softmax_j((q2·K_j + q2·E_k[c_j])·¼)·(V_j + E_v[c_j])⌉   j は行の局・席・牌の F 個とイベントの E 個、FP32
//   z  = ⌊z1 + ⌊o2·W_Oᵀ + b_O⌉⌉
//   B  = 打牌の値（prism_hip_value.h）
//
// キーとバリューは REREAD_MEMORY にある。局・席・牌の分は [R, F, 2A]（K8 が書く）、イベントの分は [N_E, 2A]（最後の全トークンの層の
// K3 が書く）。キーのコード c_j は次のとおり。
// - 局・席・牌のキー: 遷移のコード。
// - イベントのキー: τ（鳴いた後に捨てる牌、なければ主牌）とイベントの牌の両方があれば keyCodes + 1 + 牌の関係の種類、
//   なければ keyCodes。
//
// Cross-Attention は (ヘッド, 遷移) の項目ごとに、キーを番号 j の順に 16 個ずつのブロックに分けてオンラインの softmax をかける。
// ブロックごとに、それまでの和とバリューの累積を 2^(前の最大 − 新しい最大) 倍する。ブロックの切れ目は j の 16 の倍数で決まり、
// リダクションは全て遷移の中で閉じるので、同じ遷移の z と B はバッチの組み方に依らずビットまで同じ。W_O と b_O が 0 なら、z は z1 と
// ビットまで同じ。
//
// 待ち時間を減らす工夫:
// - キーの読み込みは、kAhead 個先まで発行するソフトウェアパイプラインのリングで行う（K6 と同じ形。3 個先にすると VGPR が 224 本を
//   越える）。ブロックごとにキーの段（K と E_k）、バリューの段（V と E_v）の順に途切れず回し、バリューの段の終わりに次のブロックの
//   キーの読み込みを発行する。
// - 次のブロックのコードの材料（遷移のコードとイベントの牌）は、ブロックの初めに読み込みを発行し、キーの段の後に 16 バイトに詰める。
// q2 の射影は K6 の段 4、出力の射影は mlpStage の段 1・2 と同じ形と丸め。バリアは全て tile::barrier（LDS だけ）。
#include "prism_hip_candidate.h"
#include "prism_hip_relation.h"
#include "prism_hip_value.h"

namespace prism {
namespace {

using candidate::KeySlot;
using tile::kChunk;
using tile::kThreads;
using tile::kWaves;

constexpr int kTile = 48;   // ワークグループが受け持つ遷移の数
constexpr int kRowTiles = kTile / 16;
constexpr int kBlock = 16;  // オンラインの softmax で 1 ブロックにまとめるキーの数
constexpr int kAhead = 2;   // 読み込みを何段先まで発行するか

template <int H>
struct alignas(16) RereadShared {
  tile::MlpShared<H, kTile> mlp;  // stream: z1 → z → ⌊N_B(z)⌉、inner: ⌊N_q(z1)⌉ の 128 列 → q2 → o2
  float inverse[kTile];           // 1/rms(z1)
  float partial[kWaves][kTile];   // B のウェーブごとの部分和
  int row[kTile];
  int tile[kTile];        // τ（牌の ID、0 はなし）
  int firstEvent[kTile];  // 行の最初のイベント
  int events[kTile];      // 行のイベントの数
  uint8_t tileRelation[kMaxTiles * kMaxTiles];
};

// ブロックのコードの材料（読み込みを発行したもの）。局・席・牌のキーのコードの 16 バイトと、イベントのキーの牌（ID、0 はなし）。
struct BlockSource {
  u32x4 fixed;
  int tiles[kBlock];
};

// 16 バイトを S バイト（定数）だけ下へずらしたときの w 番目のワードを返す。16 バイトの外から入る上のバイトは 0。
template <int S>
__device__ __forceinline__ uint32_t shiftedWord(const u32x4& bytes, int w) {
  const int q = w + S / 4;
  constexpr int r = S % 4;
  const uint32_t lo = q < 4 ? bytes[q & 3] : 0u;
  const uint32_t hi = q + 1 < 4 ? bytes[(q + 1) & 3] : 0u;
  return r == 0 ? lo : (lo >> (8 * r)) | (hi << (32 - 8 * r));
}

// ワード w のうち、16 バイトの先頭 n バイトに入るバイトのマスク。ワークグループで一様な値から作るので SGPR に載る。
__device__ __forceinline__ uint32_t leadingBytes(int n, int w) {
  const int count = n - 4 * w;
  return count >= 4 ? 0xffffffffu : count <= 0 ? 0u : (1u << (8 * count)) - 1u;
}

// 値をコンパイラから隠す（命令は出ない）。キーの段とバリューの段は、同じキーのアドレスを同じ式で作る。隠さないと、コンパイラがキーの段で
// 作ったアドレスをバリューの段まで持ち続ける（キーごとに VGPR 2 本）。
template <class T>
__device__ __forceinline__ void opaque(T& value) {
  asm volatile("" : "+v"(value));
}

// (ヘッド, 遷移) の項目 1 つについて、キーの引き方を持つ。キー j の MEMORY の行は、局・席・牌のキー（j < F）なら row·F + j、
// イベントのキーなら F·R + 行の最初のイベント + (j − F)。ブロックの終わりで行のキーの数を超える j は最後のキーを読み、ロジットを −∞ に
// して使わない。
template <int A, int F>
struct ItemKeys {
  int count;          // キーの数（F + イベントの数）
  int events;         // 行のイベントの数
  int fixedRow;       // 行の局・席・牌の最初のキーがある MEMORY の行
  int eventToken;     // 行の最初のイベントのトークン（MEMORY と関係の属性の行）
  int relationRow;    // 牌の関係の表で τ の行が始まる位置
  bool tau;           // τ があるか
  int codeAt;         // 遷移のコードの先頭
  uint32_t headKey;   // キーの中のヘッドの位置（バイト）
  uint32_t headCode;  // E の行の中のヘッドの位置（バイト）

  // キーのアドレスの材料をコンパイラから隠す。バリューの段の前に呼ぶ。
  __device__ __forceinline__ void conceal() {
    opaque(count);
    opaque(fixedRow);
    opaque(eventToken);
    opaque(headKey);
    opaque(headCode);
  }

  // キー j の MEMORY でのバイト位置（ヘッドの位置を含む）。キーの K と V を合わせた 1 行の範囲を確かめる。
  __device__ __forceinline__ uint32_t keyByte(int j, size_t limit, Bounds& bounds) const {
    const int key = min(j, count - 1);
    const int memoryRow = key < F ? fixedRow + key : eventToken + (key - F);
    return static_cast<uint32_t>(bounds.index(static_cast<size_t>(memoryRow) * (2 * A), 2 * A, limit) *
                                 sizeof(uint16_t)) +
           headKey;
  }

  // ブロック b のコードの材料の読み込みを発行する。読むのは遷移のコードの 16 バイト（遷移の末尾を越えないよう [F − 16, F) に寄せる）と、
  // イベントの牌 16 個（番号は行のイベントの範囲に寄せ、イベントのない行は前のトークンを読む）。どのブロックでも同じ読み込みを発行し、
  // 使わない値は codes が捨てる。ブロックの種類で分岐すると、コンパイラがキーの段の読み込みをブロックの初めに集めて spill する。
  __device__ __forceinline__ BlockSource source(int b, const RereadArgs& args, Bounds& bounds) const {
    BlockSource s;
    const int first = kBlock * b;
    s.fixed = loadUnaligned16(args.code.data +
                              bounds.index(static_cast<size_t>(codeAt) + min(first, F - kBlock), kBlock, args.code.limit));
#pragma unroll
    for (int k = 0; k < kBlock; ++k) {
      const int event = min(max(first + k - F, 0), events - 1);
      const size_t at = static_cast<size_t>(eventToken + event) * args.relationWidth + args.relationTile;
      s.tiles[k] = args.relation.data[bounds.index(at, 1, args.relation.limit)];
    }
    return s;
  }

  // ブロック b のキーのコードを 16 バイトに詰める（キー 16b + k のコードがバイト k）。ブロックの先頭 n = clamp(F − 16b, 0, 16) 個は
  // 局・席・牌のキーで遷移のコードを、残りはイベントのキーのコードを入れる。局・席・牌とイベントが混ざるブロックでは、コードの 16 バイトを
  // 遷移の末尾に寄せて読んでいるので、F mod 16 個を上のバイトから取る。分岐も、読み込みをオペランドに持つ select も作らない（作ると
  // コンパイラが LDS の読み込みを分岐の中へ沈める）。
  __device__ __forceinline__ u32x4 codes(int b, const BlockSource& s, const uint8_t* table, int keyCodes, int tableSize,
                                         Bounds& bounds) const {
    constexpr int kShift = (kBlock - F % kBlock) % kBlock;  // 混ざるブロックの最初のキーが、寄せて読んだ 16 バイトの何バイト目か
    const int first = kBlock * b;
    const int fixedKeys = min(max(F - first, 0), kBlock);
    const uint32_t mixed = first + kBlock > F ? 0xffffffffu : 0u;
    u32x4 events = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int k = 0; k < kBlock; ++k) {
      const int tile = s.tiles[k];
      const int relation = table[bounds.index(static_cast<size_t>(relationRow + max(tile - 1, 0)), 1, tableSize)];
      const int related = tau && tile > 0 ? 1 : 0;
      events[k >> 2] |= static_cast<uint32_t>(keyCodes + (1 + relation) * related) << (8 * (k & 3));
    }
    u32x4 packed;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t fixed = (s.fixed[w] & ~mixed) | (shiftedWord<kShift>(s.fixed, w) & mixed);
      const uint32_t keep = leadingBytes(fixedKeys, w);
      packed[w] = (fixed & keep) | (events[w] & ~keep);
    }
    return packed;
  }
};

// 段 3: Cross-Attention。(ヘッド, 遷移) の kHeads·kTile 個の項目を item = tid + 256u でスレッドに配る。項目 item は遷移
// i = item / kHeads とヘッド h = item % kHeads の組。1 つの遷移のヘッドは隣り合う 8 lane が受け持ち、キーの 1 行のヘッドを並んで
// 読む。o2 は q2 の上に書く。項目ごとの計算:
//   ℓ_j = (q2·K_j + q2·E_k[c_j])·scale·log2e
//   ブロックごとに M' = max(M, max_j ℓ_j) とし、和 S とバリューの累積 o を 2^(M − M') 倍してから
//   S += Σ_j 2^(ℓ_j − M')、o += Σ_j 2^(ℓ_j − M')·(V_j + E_v[c_j])（キーの順）
//   最後に o2 = ⌊o·(1/S)⌉
// リングのスロットは、読み込みの段の番号 mod kSlots で決まる。段はブロックごとにキー 16 個の K、続けて V の順に並ぶ。ブロックの段の数は
// kSlots の倍数なので、スロットの割り当てはブロックに依らない。
template <int H, int F>
__device__ __forceinline__ void attend(const RereadArgs& args, RereadShared<H>& shared, int t0, int rows, int tid,
                                       Bounds& bounds) {
  constexpr int A = H / 2;
  constexpr int kHeads = A / 16;
  constexpr int kItems = kHeads * kTile;
  constexpr int kRounds = (kItems + kThreads - 1) / kThreads;
  constexpr int kValueBytes = A * sizeof(uint16_t);     // キーの中の V の位置
  constexpr int kHeadKeyBytes = 16 * sizeof(uint16_t);  // キーの K・V の中のヘッド 1 つ
  constexpr int kHeadCodeBytes = 16 * sizeof(float);    // E の行の中のヘッド 1 つ
  constexpr int kSlots = 4;  // リングのスロットの数（同時に使うのは kAhead + 1 個）
  constexpr int kSteps = 2 * kBlock;
  static_assert(kItems % kWave == 0, "a wave either has an item in a round or not");
  static_assert(kAhead < kSlots && kBlock % kSlots == 0, "every block starts at ring slot 0");
  auto& inner = shared.mlp.inner;
  const float logitScale = args.scale * 1.4426950408889634f;  // softmax は exp2 で計算する
  const __amdgpu_buffer_rsrc_t memory = bufferResource(args.memory.data);
  const __amdgpu_buffer_rsrc_t codeKey = bufferResource(args.codeKey.data);
  const __amdgpu_buffer_rsrc_t codeValue = bufferResource(args.codeValue.data);
  const int fixedTokens = F * args.layout.rows;
  const int tableSize = args.layout.tiles * args.layout.tiles;
#pragma unroll 1
  for (int u = 0; u < kRounds; ++u) {
    const int item = tid + kThreads * u;
    const int i = item / kHeads;
    if (item >= kItems || i >= rows) {
      continue;
    }
    const int h = item % kHeads;
    const int tau = shared.tile[i];
    ItemKeys<A, F> keys;
    keys.events = shared.events[i];
    keys.count = F + keys.events;
    keys.fixedRow = shared.row[i] * F;
    keys.eventToken = fixedTokens + shared.firstEvent[i];
    keys.relationRow = max(tau - 1, 0) * args.layout.tiles;
    keys.tau = tau > 0;
    keys.codeAt = (t0 + i) * F;
    keys.headKey = h * kHeadKeyBytes;
    keys.headCode = h * kHeadCodeBytes;

    u32x4 query[2];
    query[0] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16]);
    query[1] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16 + 8]);
    float q[2][8];
    unpack8(query[0], q[0]);
    unpack8(query[1], q[1]);

    // ブロック b のキー k について、K と E_k（issueKey）か V と E_v（issueValue）の読み込みをスロットに発行する。
    const auto issueKey = [&](KeySlot& slot, int b, int k, const u32x4& codes) {
      slot.issue(memory, keys.keyByte(kBlock * b + k, args.memory.limit, bounds), 0, codeKey,
                 candidate::codeTableRow<A>(codes, k, args.codeKey.limit, bounds) + keys.headCode);
    };
    const auto issueValue = [&](KeySlot& slot, int b, int k, const u32x4& codes) {
      slot.issue(memory, keys.keyByte(kBlock * b + k, args.memory.limit, bounds), kValueBytes, codeValue,
                 candidate::codeTableRow<A>(codes, k, args.codeValue.limit, bounds) + keys.headCode);
    };

    KeySlot ring[kSlots];
    u32x4 codes = keys.codes(0, keys.source(0, args, bounds), shared.tileRelation, args.keyCodes, tableSize, bounds);
#pragma unroll
    for (int k = 0; k < kAhead; ++k) {
      issueKey(ring[k], 0, k, codes);
    }
    float maximum = -INFINITY;
    float total = 0.0f;
    float value[2][8];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        value[half][d] = 0.0f;
      }
    }
    const int blocks = (keys.count + kBlock - 1) / kBlock;
#pragma unroll 1
    for (int b = 0; b < blocks; ++b) {
      const BlockSource next = keys.source(b + 1, args, bounds);

      // キーの段: ℓ を求める（行のキーの外は −∞）。
      float logits[kBlock];
#pragma unroll
      for (int k = 0; k < kBlock; ++k) {
        const int ahead = k + kAhead;
        if (ahead < kBlock) {
          issueKey(ring[ahead % kSlots], b, ahead, codes);
        } else {
          issueValue(ring[ahead % kSlots], b, ahead - kBlock, codes);
        }
        __builtin_amdgcn_sched_barrier(0);
        // 行のキーの外は −∞ を足す（ロジットを select のオペランドにすると、コンパイラが積を分岐の中へ沈めてパイプラインが崩れる）。
        const KeySlot& slot = ring[k % kSlots];
        const float outside = kBlock * b + k < keys.count ? 0.0f : -INFINITY;
        logits[k] = (slot.keyDot(query) + slot.codeDot(q)) * logitScale + outside;
        __builtin_amdgcn_sched_barrier(0);
      }
      float blockMaximum = logits[0];
#pragma unroll
      for (int k = 1; k < kBlock; ++k) {
        blockMaximum = fmaxf(blockMaximum, logits[k]);
      }
      const float updated = fmaxf(maximum, blockMaximum);
      const float shrink = __builtin_amdgcn_exp2f(maximum - updated);  // 最初のブロックでは 2^−∞ = 0
      total *= shrink;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          value[half][d] *= shrink;
        }
      }
#pragma unroll
      for (int k = 0; k < kBlock; ++k) {
        logits[k] = __builtin_amdgcn_exp2f(logits[k] - updated);
        total += logits[k];
      }
      maximum = updated;
      const u32x4 nextCodes = keys.codes(b + 1, next, shared.tileRelation, args.keyCodes, tableSize, bounds);
      keys.conceal();
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        uint32_t word = codes[w];
        opaque(word);
        codes[w] = word;
      }

      // バリューの段: o += 2^(ℓ − M')·(V + E_v)。終わりに次のブロックのキーの読み込みを発行する。
#pragma unroll
      for (int k = 0; k < kBlock; ++k) {
        const int ahead = kBlock + k + kAhead;
        if (ahead < kSteps) {
          issueValue(ring[ahead % kSlots], b, ahead - kBlock, codes);
        } else {
          issueKey(ring[ahead % kSlots], b + 1, ahead - kSteps, nextCodes);
        }
        __builtin_amdgcn_sched_barrier(0);
        ring[k % kSlots].accumulate(logits[k], value);
        __builtin_amdgcn_sched_barrier(0);
      }
      codes = nextCodes;
    }
    const float inverse = 1.0f / total;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        value[half][d] *= inverse;
      }
    }
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}

template <int H, int F>
__global__ __launch_bounds__(kThreads) void rereadKernel(RereadArgs args) {
  constexpr int A = H / 2;
  constexpr int kColumns = H / 16 / kWaves;
  static_assert(A == kChunk, "the query and the attention output are staged in the hidden buffer");
  static_assert(H % kChunk == 0, "the query projection reads the stream in chunks");
  static_assert(sizeof(RereadShared<H>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ RereadShared<H> shared;
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

  // 0. z1 を stream に読み、遷移の行・τ・行のイベントと、牌の関係の表を LDS に置く。
  tile::stageTile<H>(stream, args.transition.data, args.transition.limit, t0, rows, tid, bounds);
  if (tid < kTile) {
    int row = 0;
    int tileId = 0;
    int firstEvent = 0;
    int events = 0;
    if (tid < rows) {
      const int t = t0 + tid;
      row = candidate::transitionRow(args.rowColumn, args.rowWidth, args.rowFirstTransition, layout.rows, t, bounds);
      tileId = candidate::discardTile(args.category, args.categoryWidth, args.callDiscard, args.mainTile, t, bounds);
      const size_t at = static_cast<size_t>(row) * args.rowWidth;
      firstEvent = args.rowColumn.data[bounds.index(at + args.rowFirstEvent, 1, args.rowColumn.limit)];
      events = args.rowColumn.data[bounds.index(at + args.rowEvents, 1, args.rowColumn.limit)];
    }
    shared.row[tid] = row;
    shared.tile[tid] = tileId;
    shared.firstEvent[tid] = firstEvent;
    shared.events[tid] = events;
  }
  for (int e = tid; e < layout.tiles * layout.tiles; e += kThreads) {
    shared.tileRelation[e] = args.tileRelation.data[bounds.index(e, 1, args.tileRelation.limit)];
  }
  tile::barrier();

  // 1. 1/rms(z1) を inverse に置く。4 スレッドで 1 行を受け持ち、kTile 行より後ろの受け持ちは飛ばす。
  {
    const int normRow = tid >> 2;
    const int normCol = (tid & 3) * (H / 4);
    if (normRow < kTile) {
      u32x4 part[H / 32];
#pragma unroll
      for (int c = 0; c < H / 32; ++c) {
        part[c] = *reinterpret_cast<const u32x4*>(&stream[normRow][normCol + 8 * c]);
      }
      const float inverse = tile::inverseRms<H>(part, args.epsilon);
      if ((tid & 3) == 0) {
        shared.inverse[normRow] = inverse;
      }
    }
  }
  tile::barrier();

  // 2. q2 = ⌊⌊N_q(z1)⌉·W_qᵀ + b_q⌉。正規化した z1 を 128 列ずつ inner に置き、K の順に累積する（z1 は stream に残す）。
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

  // 3. Cross-Attention。⌊o2⌉ を q2 の上に書く。
  attend<H, F>(args, shared, t0, rows, tid, bounds);
  tile::barrier();

  // 4. z = ⌊z1 + ⌊o2·W_Oᵀ + b_O⌉⌉ を stream の z1 の上に作り、TRANSITION へ書く。受け持ちと丸めは mlpStage の段 1・2 と同じ。
  {
    const int fragmentWave = __builtin_amdgcn_readfirstlane(wave);  // 重みの写しを読む soffset を SGPR に置く
    const tile::FragmentWeights output = tile::fragmentWeights(args.output, A, column);
    f32x8 acc[kRowTiles][kColumns];
#pragma unroll
    for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kColumns; ++j) {
        acc[m][j] = f32x8{};
      }
    }
    int columns[kColumns];
    float bias[kColumns];
    int base[kColumns];
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
      columns[j] = (fragmentWave * kColumns + j) * 16 + column;
      bias[j] = tile::load(args.outputBias.data, args.outputBias.limit, columns[j], bounds);
      base[j] = tile::fragmentBase<A / 16>(output, fragmentWave * kColumns + j, 0, bounds);
    }
    tile::gemmFragments<kColumns, A>(acc, inner, output, base, column);
#pragma unroll
    for (int j = 0; j < kColumns; ++j) {
#pragma unroll
      for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int row = m * 16 + 2 * v + half;
          const float y = bf16Float(bf16RoundActivation(acc[m][j][v] + bias[j]));
          stream[row][columns[j]] = bf16RoundActivation(bf16Float(stream[row][columns[j]]) + y);
        }
      }
    }
  }
  tile::barrier();
  tile::storeTile<H>(args.transition.data, args.transition.limit, stream, t0, rows, tid, bounds);
  tile::barrier();

  // 5. 打牌の値 B。
  discard::value<H>(stream, shared.partial, args.discard, args.discardExact, args.discardExactWidth, args.gate,
                    args.epsilon, args.value, t0, rows, bounds);
  bounds.trap();
}

}  // namespace

void launchReread(int hidden, const RereadArgs& args, hipStream_t stream) {
  if (args.transitions <= 0) {
    return;
  }
  if (args.discardExactWidth > discard::kMaxExact) {
    throw std::invalid_argument("K9 reads at most " + std::to_string(discard::kMaxExact) +
                                " B path features per transition, the schema has " +
                                std::to_string(args.discardExactWidth));
  }
  const int tiles = args.layout.tiles;
  if (tiles <= 0 || tiles > kMaxTiles || args.tileRelation.limit < static_cast<size_t>(tiles) * tiles) {
    throw std::invalid_argument("K9 keeps the tile relation table of " + std::to_string(tiles) +
                                " tiles in LDS, which holds at most " + std::to_string(kMaxTiles));
  }
  if (args.keyCodes <= 0 || args.keyCodes > 255) {
    throw std::invalid_argument("K9 needs 1..255 transition key codes, got " + std::to_string(args.keyCodes));
  }
  if (hidden != 256) {
    throw std::invalid_argument("K9 has no specialisation for hidden " + std::to_string(hidden));
  }
  const dim3 grid((args.transitions + kTile - 1) / kTile);
  switch (args.layout.fixed()) {
    case 39:
      hipLaunchKernelGGL((rereadKernel<256, 39>), grid, dim3(kThreads), 0, stream, args);
      break;
    case 31:
      hipLaunchKernelGGL((rereadKernel<256, 31>), grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K9 is built for 31 or 39 keys per row, got " + std::to_string(args.layout.fixed()));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
