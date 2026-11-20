// 候補層ではワークグループ（256 スレッド = 8 ウェーブ）が 48 遷移を受け持つ。eager の CandidateLayer に当たる（静的な部分 t_s は
// 埋め込みが作る）。行のキーの数 F（局・席・牌）はテンプレート引数で、31 と 39 のカーネルを持つ。打牌の値 B を書くかもテンプレート引数で
// 決める。1 段の形は z の後に B（discard_value.h）を書き、2 段目のある形は z を 1 段目の出力 z1 として書いて終わる（B はイベント再読が書く）。
//
//   t0 = ⌊t_s + ⌊y_T[τ]·W_Eᵀ⌉⌉                  τ = 鳴いた後に捨てる牌、なければ主牌、なければ項なし
//   q  = ⌊⌊N_q(t0)⌉·W_qᵀ + b_q⌉
//   o  = ⌊Σ_j softmax_j((q·K_j + q·E_k[c_j])·¼)·(V_j + E_v[c_j])⌉   j は行の局・席・牌の F 個、FP32
//   z  = mlpStage(t0, o)                           （前段 FFNと同じ式、候補の重み）
//
// リダクションは全て遷移の中で閉じる（Cross-Attention はキーの順、B の和は列の決まった順）ので、同じ遷移の z と B は、タイルの組み方に
// 依らずビットで同じ。タイルは前段 FFNと同じ 48 行（64 行より共有メモリが小さく、1 つの計算ユニットに載るワークグループが多い）。
// 遷移ごとのリダクションの順（行列積の K のステップ、RMSNorm の 4 スレッドのツリー、B の halfSum とウェーブの順の和）は、タイルの行の
// 数に依らない。Cross-Attention は (ヘッド, 遷移) の項目をスレッドに配る（attendPipelined）。RMSNorm の受け持ち（4 スレッドで 1 行）は
// 64 行ぶんあるので、段 3 は行 48 以上（ウェーブ 6・7）を飛ばす（normalizeTile と同じ）。
// 行列積は全てソフトウェアパイプラインで回す（tileEntry・query・discardHidden は gemmTile、mlpStage は、計画が作ったフラグメントの順の
// コピーを読む gemmFragmentsFrom。feed_forward_tiles.h）。t_s を最初に読み、t0 の epilogue を 1/rms と同じ受け持ちの 1 回にまとめるので、
// tileEntry のパイプラインも spill しない。Cross-Attention は、キーを kAhead 個先まで device::Buffer から読むパイプラインのループ
// （attendPipelined）。FFN の h と B の silu は siluFast、活性値の丸めは bf16RoundActivation（NaN の選択なし）。バリアは全て
// tile::barrier で、共有メモリの順序だけをそろえる（読む領域と書く領域が別であることが前提）。
#include "kernel_tuning.h"
#include "bounded_buffer.h"
#include "collection_capabilities.h"
#include "candidate_offsets.h"
#include "inference_stages.h"
#include "discard_value.h"
#include "relation_encoding.h"
#include "static_shared_memory.h"

namespace prism {
namespace {

using candidate::KeySlot;
using tile::kChunk;
using tile::kThreads;
using tile::kWaves;

constexpr int kPreferredCandidateRows = KernelTuning::kPreferredCandidateRows;
constexpr int kDictionaryRows = KernelTuning::kDictionaryRows;
constexpr int kAhead = KernelTuning::kCandidateAhead;
static_assert(kPreferredCandidateRows > 0 && kDictionaryRows > 0, "candidate tiles need at least one row");

// 遷移のキーのコード F バイトを、16 バイトずつ何回で読むか（最後の 1 回は末尾に寄せて [F − 16, F) を読む）。
template <int F>
__host__ __device__ constexpr int codePieces() {
  static_assert(F >= 16 && F <= kMaxFixedTokens, "the key codes are read as 16-byte pieces");
  return (F + 15) / 16;
}

// コードを 16 バイトずつ読むとき、p 回目に読む先頭のバイト位置。
template <int F>
__host__ __device__ constexpr int codePieceStart(int p) {
  return p < codePieces<F>() - 1 ? 16 * p : F - 16;
}

template <int H, int kTile>
struct alignas(16) CandidateShared {
  tile::MlpShared<H, kTile> mlp;
  float inverse[kTile];
  float partial[kWaves][kTile];
  int row[kTile];
  int tile[kTile];
};

template <int H>
constexpr int kCandidateRows =
    sizeof(CandidateShared<H, kPreferredCandidateRows>) <= device::kStaticSharedBytes ? kPreferredCandidateRows
                                                                                      : kDictionaryRows;

// キー j のコードが指す E の行（A 個の FP32）のバイト位置。コードの F バイトは、codePieces 個の 16 バイトに分けて読んである
// （F = 39 なら [0, 16)・[16, 32)・[23, 39)、F = 31 なら [0, 16)・[15, 31)）。範囲は E の行ごとに確かめる。
template <int A, int F>
__device__ __forceinline__ uint32_t codeRow(const u32x4 (&codes)[codePieces<F>()], int j, size_t limit,
                                            Bounds& bounds) {
  const int part = j / 16 < codePieces<F>() - 1 ? j / 16 : codePieces<F>() - 1;
  return candidate::codeTableRow<A>(codes[part], j - codePieceStart<F>(part), limit, bounds);
}

// 段 5: Cross-Attention。(ヘッド, 遷移) の kHeads·kTile 個の項目を item = tid + 256u（u = 0, 1）でスレッドに配る。項目 item は
// ヘッド h = item / kTile と遷移 i = item % kTile を受け持つ（ウェーブ 0〜3 は 2 項目、ウェーブ 4〜7 は 1 項目）。
//   d_j = q·K_j + q·E_k[c_j]（2 要素の内積の連鎖と積和の連鎖）、ℓ_j = d_j·scale·log2e、
//   p_j = 2^(d_j·scale·log2e − max)·(1 / Σ_j 2^(…))（max は丸めた ℓ_j から選び、指数の引数は積和で 1 回だけ丸める）、
//   o = ⌊Σ_j p_j·(V_j + E_v[c_j])⌉
// 項目の式とリダクションの順は受け持ちに依らない。h は lane ごとに違うので、キーの中のヘッドの位置は rowByte に、E の行の中のヘッドの
// 位置は codeRow の結果に足す（lane ごとのレジスタ）。キー j + kAhead の読み込みを出してからキー j を使うパイプラインのループで、kAhead + 1 個の
// スロットを回す（device::scheduleLoads で区切らないと、コンパイラーが読み込みを使う直前へ寄せてしまう）。読み込みは device::Buffer から行い、キーごとの
// 64 bit のアドレスを作らない（作るとコンパイラーが 39 組のアドレスを項目のループの外へ持ち出して spill する）。項目の本体は別の関数に分けない。
// 分けると、先に展開されたキーのコードの行の位置をコンパイラーが項目のループの外へ持ち出し、ℓ のループと o のループで共有して spill する。
template <int H, int F, int kTile>
__device__ __forceinline__ void attendPipelined(const CandidateArgs& args, CandidateShared<H, kTile>& shared, int t0,
                                                int rows, int tid, Bounds& bounds) {
  constexpr int A = H / 2;
  constexpr int kHeads = A / 16;
  constexpr int kItems = kHeads * kTile;
  constexpr int kRounds = (kItems + kThreads - 1) / kThreads;
  constexpr int kKeyBytes = 2 * A * sizeof(uint16_t);   // MEMORY のキー 1 つ（K と V）
  constexpr int kValueBytes = A * sizeof(uint16_t);     // キーの中の V
  constexpr int kHeadKeyBytes = 16 * sizeof(uint16_t);  // キーの K・V の中のヘッド 1 つ
  constexpr int kHeadCodeBytes = 16 * sizeof(float);    // E の行の中のヘッド 1 つ
  constexpr int kSlots = kAhead + 1;
  constexpr int kKeys = F;
  static_assert(kAhead >= 0 && kAhead <= kKeys, "candidate prefetch exceeds the key tile");
  constexpr int kPieces = codePieces<F>();
  static_assert(kItems % kWave == 0, "a wave either has an item in a round or not");
  auto& inner = shared.mlp.inner;
  const float logitScale = args.scale * kLog2e;  // softmax は exp2 で計算する
  const device::Buffer memory = device::boundedBuffer(args.memory.data, args.memory.limit);
  const device::Buffer codeKey = device::boundedBuffer(args.codeKey.data, args.codeKey.limit);
  const device::Buffer codeValue = device::boundedBuffer(args.codeValue.data, args.codeValue.limit);
#pragma unroll 1
  for (int u = 0; u < kRounds; ++u) {
    const int item = tid + kThreads * u;
    const int i = item % kTile;
    if (item >= kItems || i >= rows) {
      continue;
    }
    const int h = item / kTile;
    // 行の kKeys 個のキー（K と V）は 1 つの領域として範囲を確かめ、キーの中のヘッドの位置を足す。
    const size_t keyAt =
        bounds.index(static_cast<size_t>(shared.row[i]) * kKeys * (2 * A), kKeys * (2 * A), args.memory.limit);
    const uint32_t rowByte = static_cast<uint32_t>(keyAt * sizeof(uint16_t)) + h * kHeadKeyBytes;
    // キーのコード（アラインしていない 16 バイトの読み込みを kPieces 回。どれもその遷移のフィールドの中）。
    const size_t codeAt = static_cast<size_t>(t0 + i) * kKeys;
    u32x4 codes[kPieces];
#pragma unroll
    for (int p = 0; p < kPieces; ++p) {
      codes[p] = loadUnaligned16(args.code.data + bounds.index(codeAt + codePieceStart<F>(p), 16, args.code.limit));
    }
    const uint32_t tableLane = h * kHeadCodeBytes;  // E の行の中のヘッドの位置
    u32x4 query[2];
    query[0] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16]);
    query[1] = *reinterpret_cast<const u32x4*>(&inner[i][h * 16 + 8]);
    float q[2][8];
    unpack8(query[0], q[0]);
    unpack8(query[1], q[1]);

    // 内積 d と ℓ = d·scale の最大値。指数は d·scale − 最大値を 1 回で丸めて求める（最大値は丸めた ℓ から選ぶ）。
    KeySlot ring[kSlots];
#pragma unroll
    for (int j = 0; j < kAhead; ++j) {
      ring[j].issue(memory, rowByte, j * kKeyBytes, codeKey,
                    codeRow<A, F>(codes, j, args.codeKey.limit, bounds) + tableLane);
    }
    float logits[kKeys];
    float maximum = -INFINITY;
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      const int ahead = j + kAhead;
      if (ahead < kKeys) {
        ring[ahead % kSlots].issue(memory, rowByte, ahead * kKeyBytes, codeKey,
                                   codeRow<A, F>(codes, ahead, args.codeKey.limit, bounds) + tableLane);
      }
      device::scheduleLoads();
      const KeySlot& slot = ring[j % kSlots];
      logits[j] = slot.keyDot(query) + slot.codeDot(q);
      maximum = fmaxf(maximum, roundedMultiply(logitScale, logits[j]));
      device::scheduleLoads();
    }
    float total = 0.0f;
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      logits[j] = device::exponent2(fusedMultiplyAdd(logitScale, logits[j], -maximum));
      total += logits[j];
    }
    const float inverse = 1.0f / total;

    // o = Σ_j p_j·(V_j + E_v[c_j])。
#pragma unroll
    for (int j = 0; j < kAhead; ++j) {
      ring[j].issue(memory, rowByte, j * kKeyBytes + kValueBytes, codeValue,
                    codeRow<A, F>(codes, j, args.codeValue.limit, bounds) + tableLane);
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
                                   codeRow<A, F>(codes, ahead, args.codeValue.limit, bounds) + tableLane);
      }
      device::scheduleLoads();
      ring[j % kSlots].accumulate(logits[j] * inverse, value);
      device::scheduleLoads();
    }
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}

// 牌のコードで各区分の表を引き、キーまたはバリューの補正を区分順に加える。
template <int A, bool kValue>
__device__ __forceinline__ void addSegmentOffsets(KeySlot& slot, const CandidateArgs& args,
                                                  const SegmentWeights& weights, device::Buffer table, size_t codes,
                                                  int tile, int head, Bounds& bounds) {
  // 固定位置から区分を読む。カーネル引数の配列を lane ごとの作業領域に写さずに使う。
#pragma unroll
  for (int segment = 0; segment < kMaxSegmentSpans; ++segment) {
    if (segment >= weights.layout.count) break;
    const auto& span = weights.layout.spans[segment];
    const size_t at = codes + span.column + tile;
    const int id = args.code.data[bounds.index(at, 1, args.code.limit)];
    const size_t row = span.firstRow + bounds.index(id, 1, span.rows);
    const size_t offset = bounds.index(row * (2 * A) + (kValue ? A : 0) + 16 * head, 16, weights.offsets.limit);
    const uint32_t bytes = static_cast<uint32_t>(offset * sizeof(float));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto add = device::loadWords(table, bytes, 16 * i);
      const auto terms = device::bits<f32x4>(add);
#pragma unroll
      for (int col = 0; col < 4; ++col) slot.code[i][col] += terms[col];
    }
  }
}

__device__ __forceinline__ int segmentCode(const CandidateArgs& args, HotCandidateOffsets hot, int transition,
                                           int firstTile, int key, size_t codeAt, Bounds& bounds) {
  if (key >= firstTile) {
    const size_t at = static_cast<size_t>(transition) * args.layout.tiles + key - firstTile;
    return hot.codes.data[bounds.index(at, 1, hot.codes.limit)];
  }
  return args.code.data[bounds.index(codeAt + key, 1, args.code.limit)];
}

template <int H, int F, bool dictionary, class Shared>
__device__ __forceinline__ void attendSegments(const CandidateArgs& args, const SegmentWeights& weights, Shared& shared,
                                               int t0, int rows, int tid, Bounds& bounds,
                                               HotCandidateOffsets hot = {}) {
  constexpr int A = H / 2;
  constexpr int kHeads = A / 16;
  constexpr int kItems = kHeads * std::extent_v<decltype(shared.row)>;
  constexpr int kKeyBytes = 2 * A * sizeof(uint16_t);
  constexpr int kRounds = (kItems + kThreads - 1) / kThreads;
  const int firstTile = F - args.layout.tiles;
  const int stride = weights.layout.codeWidth;
  const auto memory = device::boundedBuffer(args.memory.data, args.memory.limit);
  const auto codeKey = device::boundedBuffer(dictionary ? hot.keys.data : args.codeKey.data,
                                             dictionary ? hot.keys.limit : args.codeKey.limit);
  const auto codeValue = device::boundedBuffer(dictionary ? hot.values.data : args.codeValue.data,
                                               dictionary ? hot.values.limit : args.codeValue.limit);
  const auto segments = device::boundedBuffer(weights.offsets.data, weights.offsets.limit);
  auto& inner = shared.mlp.inner;
  const float scale = args.scale * kLog2e;
#pragma unroll 1
  for (int u = 0; u < kRounds; ++u) {
    const int item = tid + kThreads * u;
    // 同じ遷移のヘッドを隣の lane に置くと、各区分の同じ行の隣接範囲を読む。
    const int i = item / kHeads;
    if (item >= kItems || i >= rows) continue;
    const int h = item % kHeads;
    const size_t keyAt = bounds.index(static_cast<size_t>(shared.row[i]) * F * 2 * A, F * 2 * A, args.memory.limit);
    const uint32_t rowByte = static_cast<uint32_t>(keyAt * sizeof(uint16_t)) + h * 16 * sizeof(uint16_t);
    const size_t codeAt = static_cast<size_t>(t0 + i) * stride;
    u32x4 query[2] = {*reinterpret_cast<const u32x4*>(&inner[i][h * 16]),
                      *reinterpret_cast<const u32x4*>(&inner[i][h * 16 + 8])};
    float q[2][8];
    unpack8(query[0], q[0]);
    unpack8(query[1], q[1]);
    float value[2][8]{};
    if constexpr (dictionary) {
      float logits[F];
      float maximum = -INFINITY;
      const auto read = [&](KeySlot& slot, int j, bool value) {
        const int tagged = segmentCode(args, hot, t0 + i, firstTile, j, codeAt, bounds);
        const bool cached = (tagged & kHotCodeTag) != 0;
        const int code = tagged & ~kHotCodeTag;
        // キーと値の表は同じ行数・幅で作る。
        const size_t limit = cached ? hot.keys.limit : args.codeKey.limit;
        const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, limit);
        slot.issue(memory, rowByte, j * kKeyBytes + (value ? A * sizeof(uint16_t) : 0), value ? codeValue : codeKey,
                   static_cast<uint32_t>(offset * sizeof(float)));
        return j >= firstTile && !cached;
      };
      const auto logit = [&](KeySlot& slot, int j, bool miss) {
        if (miss) addSegmentOffsets<A, false>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
        logits[j] = (slot.keyDot(query) + slot.codeDot(q)) * scale;
        maximum = fmaxf(maximum, logits[j]);
      };
      // 次の表読みを発行してから、持っているキーを使う。内積と加算はキー順で進める。
      KeySlot slot0, slot1;
      bool miss0 = read(slot0, 0, false);
#pragma unroll 1
      for (int j = 0; j + 1 < F; j += 2) {
        const bool miss1 = read(slot1, j + 1, false);
        device::scheduleLoads();
        logit(slot0, j, miss0);
        miss0 = read(slot0, j + 2, false);
        device::scheduleLoads();
        logit(slot1, j + 1, miss1);
      }
      static_assert(F % 2 == 1);
      logit(slot0, F - 1, miss0);
      float total = 0.0f;
#pragma unroll 1
      for (int j = 0; j < F; ++j) {
        logits[j] = device::exponent2(logits[j] - maximum);
        total += logits[j];
      }
      const float inverse = 1.0f / total;
      const auto accumulate = [&](KeySlot& slot, int j, bool miss) {
        if (miss) addSegmentOffsets<A, true>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
        slot.accumulate(logits[j] * inverse, value);
      };
      miss0 = read(slot0, 0, true);
#pragma unroll 1
      for (int j = 0; j + 1 < F; j += 2) {
        const bool miss1 = read(slot1, j + 1, true);
        device::scheduleLoads();
        accumulate(slot0, j, miss0);
        miss0 = read(slot0, j + 2, true);
        device::scheduleLoads();
        accumulate(slot1, j + 1, miss1);
      }
      accumulate(slot0, F - 1, miss0);
    } else {
      float logits[F];
      float maximum = -INFINITY;
#pragma unroll 1
      for (int j = 0; j < F; ++j) {
        const int code = args.code.data[bounds.index(codeAt + j, 1, args.code.limit)];
        const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, args.codeKey.limit);
        KeySlot slot;
        slot.issue(memory, rowByte, j * kKeyBytes, codeKey, static_cast<uint32_t>(offset * sizeof(float)));
        if (j >= firstTile)
          addSegmentOffsets<A, false>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
        logits[j] = (slot.keyDot(query) + slot.codeDot(q)) * scale;
        maximum = fmaxf(maximum, logits[j]);
      }
      float total = 0.0f;
#pragma unroll 1
      for (int j = 0; j < F; ++j) {
        logits[j] = device::exponent2(logits[j] - maximum);
        total += logits[j];
      }
      const float inverse = 1.0f / total;
#pragma unroll 1
      for (int j = 0; j < F; ++j) {
        const int code = args.code.data[bounds.index(codeAt + j, 1, args.code.limit)];
        const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, args.codeValue.limit);
        KeySlot slot;
        slot.issue(memory, rowByte, j * kKeyBytes + A * sizeof(uint16_t), codeValue,
                   static_cast<uint32_t>(offset * sizeof(float)));
        if (j >= firstTile) addSegmentOffsets<A, true>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
        slot.accumulate(logits[j] * inverse, value);
      }
    }
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}

enum class CandidatePhase { kFull, kPrepare, kFinish };
struct BasicCandidate {};
struct SegmentedCandidate {
  Span<const uint16_t> count;
  SegmentWeights segments;
};
struct DictionaryCandidate : SegmentedCandidate {
  HotCandidateOffsets hot;
  Span<uint16_t> attentionState;
};

template <int H, int F, bool kValue, CandidatePhase Phase = CandidatePhase::kFull, class Extra>
__global__ __launch_bounds__(kThreads) void candidateKernel(CandidateArgs args, Extra extra) {
  constexpr bool kCoreLayout = !std::is_same_v<Extra, BasicCandidate>;
  constexpr bool dictionary = std::is_same_v<Extra, DictionaryCandidate>;
  constexpr int kTile = dictionary ? kDictionaryRows : kCandidateRows<H>;
  constexpr int kRowTiles = kTile / 16;
  constexpr int A = H / 2;
  constexpr int kColumns = H / 16 / kWaves;
  constexpr int kParts = H / 32;  // RMSNorm でスレッドが受け持つ H/4 列（4 スレッドで 1 行）に含まれる 16 バイトの数
  static_assert(kCoreLayout ? A >= kChunk : A == kChunk, "invalid candidate query workspace width");
  static_assert(H % kChunk == 0, "the query projection reads the stream in chunks");
  static_assert(sizeof(CandidateShared<H, kTile>) <= device::kStaticSharedBytes,
                "candidate tile exceeds static shared memory");
  __shared__ CandidateShared<H, kTile> shared;
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

  if constexpr (Phase != CandidatePhase::kFinish) {
    // 段 3 で使う t_s を最初に読んでおく（RMSNorm と同じ受け持ち。タイルの外の行は最後の行に寄せて読み、使わない）。受け持ちは 64 行
    // ぶんあり、行 kTile 以上（ウェーブ 6・7）には処理がない。
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

    // 0. 遷移ごとの行の番号と牌 τ の ID（0 は牌なし）を共有メモリに置く。
    if (tid < kTile) {
      int row = 0;
      int tileId = 0;
      if (tid < rows) {
        const int t = t0 + tid;
        row = candidate::transitionRow(args.rowColumn, args.rowWidth, args.rowFirstTransition, layout.rows, t, bounds);
        tileId = candidate::discardTile(args.category, args.categoryWidth, args.callDiscard, args.mainTile, t, bounds);
      }
      shared.row[tid] = row;
      shared.tile[tid] = tileId;
    }
    tile::barrier();

    // 1. y_T[τ] を stream へ置く（τ がない遷移とタイルの外は 0）。τ がなければ y の先頭の行を読み、読み込みを分岐なしで一度に出す。
    if constexpr (kCoreLayout) {
      constexpr int kVectors = H / 8;
      constexpr int kPasses = kTile * kVectors / kThreads;
      u32x4 values[kPasses];
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        const int r = (tid + pass * kThreads) / kVectors;
        const int col = ((tid + pass * kThreads) % kVectors) * 8;
        const int tileId = shared.tile[r];
        const size_t token = tileId > 0 ? static_cast<size_t>(1 + layout.seats) * layout.rows +
                                              static_cast<size_t>(layout.tiles) * shared.row[r] + tileId - 1
                                        : 0;
        values[pass] =
            *reinterpret_cast<const u32x4*>(args.state.data + bounds.index(token * H + col, 8, args.state.limit));
      }
      const u32x4 zero = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        const int r = (tid + pass * kThreads) / kVectors;
        const int col = ((tid + pass * kThreads) % kVectors) * 8;
        *reinterpret_cast<u32x4*>(&stream[r][col]) = shared.tile[r] > 0 ? values[pass] : zero;
      }
    } else {
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
        values[pass] =
            *reinterpret_cast<const u32x4*>(args.state.data + bounds.index(token * H + col, 8, args.state.limit));
      }
      const u32x4 zero = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        const int r = tile::tileRow<H, kTile>(pass, tid);
        *reinterpret_cast<u32x4*>(&stream[r][col]) = shared.tile[r] > 0 ? values[pass] : zero;
      }
    }
    tile::barrier();

    // 2. ⌊y_T·W_Eᵀ⌉（バイアスなし）を、C のフラグメントの並びのまま stream へ書く。
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

    // 3. t0 = ⌊t_s + ⌊E⌉⌉ を RMSNorm の受け持ちでその場で作り（タイルの外の行は t_s を 0 とする）、続けて 1/rms(t0) を inverse へ
    //    書く。行 kTile 以上の受け持ちは飛ばす（4 スレッドの組ごとに飛ぶので、inverseRms のツリーは欠けない）。
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

    // 4. q = ⌊⌊N_q(t0)⌉·W_qᵀ + b_q⌉。正規化した t0 を 128 列ずつ inner に置き、K の順に足し込む。
    if constexpr (kCoreLayout) {
      constexpr int kQueryTiles = (A / 16 + kWaves - 1) / kWaves;
      f32x8 q[kRowTiles][kQueryTiles]{};
      int units[kQueryTiles];
#pragma unroll
      for (int j = 0; j < kQueryTiles; ++j) {
        units[j] = (wave + j * kWaves) * 16 + column;
      }
      int safeUnits[kQueryTiles];
#pragma unroll
      for (int j = 0; j < kQueryTiles; ++j) {
        safeUnits[j] = kQueryTiles == 1 ? units[j] : min(units[j], A - 1);
      }
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
        if constexpr (Phase == CandidatePhase::kPrepare && kQueryTiles > 1) {
          // 最後の出力チャンクでは、存在する列を持つウェーブだけがタイル積和を行う。
          // 各ウェーブが出力タイルを 1 つずつ処理する。各列で K を足す順序は同じ。
#pragma unroll
          for (int j = 0; j < kQueryTiles; ++j) {
            if (units[j] >= A) continue;
            f32x8 accumulator[kRowTiles][1];
#pragma unroll
            for (int m = 0; m < kRowTiles; ++m) accumulator[m][0] = q[m][j];
            const int output[1] = {units[j]};
            tile::gemmTile<1, kChunk>(accumulator, inner, args.query.data, args.query.limit, H, k0, output, column,
                                      bounds);
#pragma unroll
            for (int m = 0; m < kRowTiles; ++m) q[m][j] = accumulator[m][0];
          }
        } else {
          tile::gemmTile<kQueryTiles, kChunk>(q, inner, args.query.data, args.query.limit, H, k0, safeUnits, column,
                                              bounds);
        }
        tile::barrier();
      }
#pragma unroll
      for (int j = 0; j < kQueryTiles; ++j) {
        if (kQueryTiles == 1 || units[j] < A) {
          const float bias = tile::load(args.queryBias.data, args.queryBias.limit, units[j], bounds);
#pragma unroll
          for (int m = 0; m < kRowTiles; ++m) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
              inner[m * 16 + 2 * v + half][units[j]] = bf16RoundActivation(q[m][j][v] + bias);
            }
          }
        }
      }
    } else {
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

    if constexpr (Phase == CandidatePhase::kPrepare) {
      // 後段が読むのは、融合経路の共有メモリにある丸め済みの t0 と q。
      tile::storeTile<H>(args.transition.data, args.transition.limit, stream, t0, rows, tid, bounds);
      tile::storeTile<A>(extra.attentionState.data, extra.attentionState.limit, inner, t0, rows, tid, bounds);
      bounds.trap();
      return;
    }
  } else {
    tile::stageTile<H>(stream, args.transition.data, args.transition.limit, t0, rows, tid, bounds);
    tile::stageTile<A>(inner, extra.attentionState.data, extra.attentionState.limit, t0, rows, tid, bounds);
    tile::barrier();
  }

  if constexpr (Phase == CandidatePhase::kFull) {
    if constexpr (dictionary)
      attendSegments<H, F, true>(args, extra.segments, shared, t0, rows, tid, bounds, extra.hot);
    else if constexpr (kCoreLayout)
      attendSegments<H, F, false>(args, extra.segments, shared, t0, rows, tid, bounds);
    else
      attendPipelined<H, F>(args, shared, t0, rows, tid, bounds);
    tile::barrier();
  }
  if constexpr (kCoreLayout)
    tile::mlpStage<H, kTile, true, Phase == CandidatePhase::kFinish>(shared.mlp, args.weights, args.ffn, args.epsilon,
                                                                     bounds, extra.count, t0, rows);
  else
    tile::mlpStage<H>(shared.mlp, args.weights, args.ffn, args.epsilon, bounds);
  if constexpr (kCoreLayout && H > kWave * 8) {
    for (int i = tid; i < rows * H / 8; i += kThreads) {
      const int r = i / (H / 8);
      const int c = i % (H / 8) * 8;
      const size_t at = bounds.index(static_cast<size_t>(t0 + r) * H + c, 8, args.transition.limit);
      *reinterpret_cast<u32x4*>(args.transition.data + at) = *reinterpret_cast<const u32x4*>(&stream[r][c]);
    }
  } else {
    tile::storeTile<H>(args.transition.data, args.transition.limit, stream, t0, rows, tid, bounds);
  }

  // 7. B を書く（1 段の形だけ）。
  if constexpr (kValue) {
    tile::barrier();
    discard::value<H>(stream, shared.partial, args.discard, args.discardExact, args.discardExactWidth, args.gate,
                      args.epsilon, args.value, t0, rows, bounds);
  }
  bounds.trap();
}

template <int H>
struct alignas(16) CandidateAttentionShared {
  struct {
    uint16_t inner[kDictionaryRows][H / 2 + tile::kPad];
  } mlp;
  int row[kDictionaryRows];
};

template <int H, int F>
__global__ __launch_bounds__(kThreads) void candidateAttentionKernel(CandidateArgs args, SegmentWeights segments,
                                                                     HotCandidateOffsets hot,
                                                                     Span<uint16_t> attentionState) {
  constexpr int A = H / 2;
  __shared__ CandidateAttentionShared<H> shared;
  const int tid = threadIdx.x;
  const int t0 = blockIdx.x * kDictionaryRows;
  const int rows = min(kDictionaryRows, args.transitions - t0);
  Bounds bounds;
  if (tid < rows) {
    shared.row[tid] = candidate::transitionRow(args.rowColumn, args.rowWidth, args.rowFirstTransition, args.layout.rows,
                                               t0 + tid, bounds);
  }
  tile::stageTile<A>(shared.mlp.inner, attentionState.data, attentionState.limit, t0, rows, tid, bounds);
  tile::barrier();
  attendSegments<H, F, true>(args, segments, shared, t0, rows, tid, bounds, hot);
  tile::barrier();
  // 1 遷移の q を読み終えてから、同じ場所に丸め済みの o を置く。
  tile::storeTile<A>(attentionState.data, attentionState.limit, shared.mlp.inner, t0, rows, tid, bounds);
  bounds.trap();
}

template <int F>
void launchKeys(int hidden, const CandidateArgs& args, Span<const uint16_t> count, SegmentWeights segments,
                HotCandidateOffsets hot, Span<uint16_t> attentionState, dim3 grid, gpu::Stream stream) {
  forCoreWidth(hidden, [&](auto width) {
    constexpr int H = decltype(width)::value;
    if (attentionState.data == nullptr) {
      candidateKernel<H, F, true, CandidatePhase::kFull>
          <<<grid, dim3(kThreads), 0, stream>>>(args, DictionaryCandidate{{count, segments}, hot, attentionState});
      return;
    }
    candidateKernel<H, F, true, CandidatePhase::kPrepare>
        <<<grid, dim3(kThreads), 0, stream>>>(args, DictionaryCandidate{{count, segments}, hot, attentionState});
    candidateAttentionKernel<H, F><<<grid, dim3(kThreads), 0, stream>>>(args, segments, hot, attentionState);
    candidateKernel<H, F, true, CandidatePhase::kFinish>
        <<<grid, dim3(kThreads), 0, stream>>>(args, DictionaryCandidate{{count, segments}, hot, attentionState});
  });
}
template <int F>
void launchKeys(int hidden, const CandidateArgs& args, Span<const uint16_t> count, SegmentWeights segments,
                gpu::Stream stream) {
  forCoreWidth(hidden, [&](auto width) {
    constexpr int H = decltype(width)::value;
    constexpr int rows = kCandidateRows<H>;
    const dim3 grid((args.transitions + rows - 1) / rows);
    candidateKernel<H, F, true><<<grid, dim3(kThreads), 0, stream>>>(args, SegmentedCandidate{count, segments});
  });
}
template <int F>
void launchKeys(const CandidateArgs& args, dim3 grid, gpu::Stream stream) {
  if (args.writeValue) {
    candidateKernel<kWidthFolded.width, F, true><<<grid, dim3(kThreads), 0, stream>>>(args, BasicCandidate{});
  } else {
    candidateKernel<kWidthFolded.width, F, false><<<grid, dim3(kThreads), 0, stream>>>(args, BasicCandidate{});
  }
}
}  // namespace

void launchCandidateWithHotOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                   SegmentWeights segments, HotCandidateOffsets hot, gpu::Stream stream,
                                   Span<uint16_t> attentionState) {
  if (!args.writeValue || args.keys != args.layout.fixed() || args.discardExactWidth > discard::kMaxExact) {
    throw std::invalid_argument("invalid candidate block layout");
  }
  const dim3 grid((args.transitions + kDictionaryRows - 1) / kDictionaryRows);
  forCandidateKeys(args.keys, [&](auto keys) {
    launchKeys<decltype(keys)::value>(hidden, args, count, segments, hot, attentionState, grid, stream);
  });
  PRISM_GPU(gpu::lastError());
}
void launchCandidateWithOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                SegmentWeights segments, gpu::Stream stream) {
  if (args.transitions == 0) return;
  if (!args.writeValue || args.keys != args.layout.fixed() || args.discardExactWidth > discard::kMaxExact) {
    throw std::invalid_argument("invalid candidate block layout");
  }
  forCandidateKeys(args.keys,
                   [&](auto keys) { launchKeys<decltype(keys)::value>(hidden, args, count, segments, stream); });
  PRISM_GPU(gpu::lastError());
}
void launchCandidate(int hidden, const CandidateArgs& args, gpu::Stream stream) {
  if (args.transitions <= 0) {
    return;
  }
  if (args.layout.fixed() != args.keys) {
    throw std::invalid_argument("candidate attention got " + std::to_string(args.keys) +
                                " keys per row but the layout has " + std::to_string(args.layout.fixed()));
  }
  if (args.writeValue && args.discardExactWidth > discard::kMaxExact) {
    throw std::invalid_argument("candidate attention reads at most " + std::to_string(discard::kMaxExact) +
                                " B path features per transition, the schema has " +
                                std::to_string(args.discardExactWidth));
  }
  if (hidden != kWidthFolded.width) {
    throw std::invalid_argument("candidate attention has no specialisation for hidden " + std::to_string(hidden));
  }
  constexpr int rows = kCandidateRows<kWidthFolded.width>;
  const dim3 grid((args.transitions + rows - 1) / rows);
  forCandidateKeys(args.keys, [&](auto keys) { launchKeys<decltype(keys)::value>(args, grid, stream); });
  PRISM_GPU(gpu::lastError());
}

}  // namespace prism
