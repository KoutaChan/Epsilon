// K6（candidate）: ワークグループ（256 スレッド = 8 ウェーブ）が 48 遷移を受け持つ。eager の CandidateLayer に当たる（静的な部分 t_s は
// K1 が作る）。行のキーの数 F（局・席・牌）はテンプレート引数で、31 と 39 のカーネルを持つ。打牌の値 B を書くかもテンプレート引数で
// 決める。1 段の形は z の後に B（prism_hip_value.h）を書き、2 段目のある形は z を 1 段目の出力 z1 として書いて終わる（B は K9 が書く）。
//
//   t0 = ⌊t_s + ⌊y_T[τ]·W_Eᵀ⌉⌉                  τ = 鳴いた後に捨てる牌、なければ主牌、なければ項なし
//   q  = ⌊⌊N_q(t0)⌉·W_qᵀ + b_q⌉
//   o  = ⌊Σ_j softmax_j((q·K_j + q·E_k[c_j])·¼)·(V_j + E_v[c_j])⌉   j は行の局・席・牌の F 個、FP32
//   z  = mlpStage(t0, o)                           （K3 と同じ式、候補の重み）
//
// リダクションは全て遷移の中で閉じる（Cross-Attention はキーの順、B の和は列の決まった順）ので、同じ遷移の z と B は、タイルの組み方に
// 依らずビットで同じ。タイルは K3 と同じ 48 行（64 行より LDS が小さく、1 WGP に載るワークグループが多い）。
// 遷移ごとのリダクションの順（行列積の K のステップ、RMSNorm の 4 スレッドのツリー、B の halfSum とウェーブの順の和）は、タイルの行の
// 数に依らない。Cross-Attention は (ヘッド, 遷移) の項目をスレッドに配る（attendPipelined）。RMSNorm の受け持ち（4 スレッドで 1 行）は
// 64 行ぶんあるので、段 3 は行 48 以上（ウェーブ 6・7）を飛ばす（normalizeTile と同じ）。
// 行列積は全てソフトウェアパイプラインで回す（tileEntry・query・discardHidden は gemmTile、mlpStage は、計画が作ったフラグメントの順の
// コピーを読む gemmFragmentsFrom。prism_hip_mlp.h）。t_s を最初に読み、t0 の epilogue を 1/rms と同じ受け持ちの 1 回にまとめるので、
// tileEntry のパイプラインも spill しない。Cross-Attention は、キーを kAhead 個先まで生の buffer で読むパイプラインのループ
// （attendPipelined）。FFN の h と B の silu は siluFast、活性値の丸めは bf16RoundActivation（NaN の選択なし）。バリアは全て
// tile::barrier で、LDS の順序だけをそろえる（読む領域と書く領域が別であることが前提）。
#include "prism_hip_candidate.h"
#include "prism_hip_value.h"

namespace prism {
namespace {

using candidate::KeySlot;
using tile::kChunk;
using tile::kThreads;
using tile::kWaves;

constexpr int kTile = 48;            // ワークグループが受け持つ遷移の数（MLP の段の行）
constexpr int kRowTiles = kTile / 16;
constexpr int kAhead = 3;            // Cross-Attention で読み込みを先に出すキーの数（5 にすると spill する）

// 遷移のキーのコード F バイトを、16 バイトずつ何回で読むか（最後の 1 回は末尾に寄せて [F − 16, F) を読む）。
template <int F>
constexpr int codePieces() {
  static_assert(F >= 16 && F <= kMaxFixedTokens, "the key codes are read as 16-byte pieces");
  return (F + 15) / 16;
}

// コードを 16 バイトずつ読むとき、p 回目に読む先頭のバイト位置。
template <int F>
constexpr int codePieceStart(int p) {
  return p < codePieces<F>() - 1 ? 16 * p : F - 16;
}

template <int H>
struct alignas(16) CandidateShared {
  tile::MlpShared<H, kTile> mlp;
  float inverse[kTile];
  float partial[kWaves][kTile];
  int row[kTile];
  int tile[kTile];
};

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
//   ℓ_j = (q·K_j + q·E_k[c_j])·scale·log2e（v_dot2 の連鎖と FMA の連鎖）、p_j = 2^(ℓ_j − max)·(1 / Σ_j 2^(ℓ_j − max))、
//   o = ⌊Σ_j p_j·(V_j + E_v[c_j])⌉
// 項目の式とリダクションの順は受け持ちに依らない。h は lane ごとに違うので、キーの中のヘッドの位置は rowByte に、E の行の中のヘッドの
// 位置は codeRow の結果に足す（VGPR）。キー j + kAhead の読み込みを出してからキー j を使うパイプラインのループで、kAhead + 1 個の
// スロットを回す（sched_barrier で区切らないと、コンパイラが読み込みを使う直前へ寄せてしまう）。読み込みは生の buffer で行い、キーごとの
// 64 bit のアドレスを作らない（作ると LLVM が 39 組のアドレスを項目のループの外へ持ち出して spill する）。項目の本体は別の関数に分けない。
// 分けると、先に展開されたキーのコードの行の位置を LLVM が項目のループの外へ持ち出し、ℓ のループと o のループで共有して spill する。
template <int H, int F>
__device__ __forceinline__ void attendPipelined(const CandidateArgs& args, CandidateShared<H>& shared, int t0, int rows,
                                                int tid, Bounds& bounds) {
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
  constexpr int kPieces = codePieces<F>();
  static_assert(kItems % kWave == 0, "a wave either has an item in a round or not");
  auto& inner = shared.mlp.inner;
  const float logitScale = args.scale * 1.4426950408889634f;  // softmax は exp2 で計算する
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

    // ℓ とその最大値。
    KeySlot ring[kSlots];
#pragma unroll
    for (int j = 0; j < kAhead; ++j) {
      ring[j].issue(memory, rowByte, j * kKeyBytes, codeKey, codeRow<A, F>(codes, j, args.codeKey.limit, bounds) + tableLane);
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
      __builtin_amdgcn_sched_barrier(0);
      ring[j % kSlots].accumulate(logits[j] * inverse, value);
      __builtin_amdgcn_sched_barrier(0);
    }
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}

// kValue は B を書くかどうか（1 段の形で true）。
template <int H, int F, bool kValue>
__global__ __launch_bounds__(kThreads) void candidateKernel(CandidateArgs args) {
  constexpr int A = H / 2;
  constexpr int kColumns = H / 16 / kWaves;
  constexpr int kParts = H / 32;  // RMSNorm でスレッドが受け持つ H/4 列（4 スレッドで 1 行）に含まれる 16 バイトの数
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

  // 0. 遷移ごとの行の番号と牌 τ の ID（0 は牌なし）を LDS に置く。
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

  // 5. Cross-Attention。⌊o⌉ を q に上書きする。
  attendPipelined<H, F>(args, shared, t0, rows, tid, bounds);
  tile::barrier();

  // 6. z = mlpStage(t0, o)（候補の O 射影と FFN）を書く。
  tile::mlpStage<H>(shared.mlp, args.weights, args.ffn, args.epsilon, bounds);
  tile::storeTile<H>(args.transition.data, args.transition.limit, stream, t0, rows, tid, bounds);

  // 7. B を書く（1 段の形だけ）。
  if constexpr (kValue) {
    tile::barrier();
    discard::value<H>(stream, shared.partial, args.discard, args.discardExact, args.discardExactWidth, args.gate,
                      args.epsilon, args.value, t0, rows, bounds);
  }
  bounds.trap();
}

template <int F>
void launchKeys(const CandidateArgs& args, dim3 grid, hipStream_t stream) {
  if (args.writeValue) {
    hipLaunchKernelGGL((candidateKernel<256, F, true>), grid, dim3(kThreads), 0, stream, args);
  } else {
    hipLaunchKernelGGL((candidateKernel<256, F, false>), grid, dim3(kThreads), 0, stream, args);
  }
}

}  // namespace

void launchCandidate(int hidden, const CandidateArgs& args, hipStream_t stream) {
  if (args.transitions <= 0) {
    return;
  }
  if (args.layout.fixed() != args.keys) {
    throw std::invalid_argument("K6 got " + std::to_string(args.keys) + " keys per row but the layout has " +
                                std::to_string(args.layout.fixed()));
  }
  if (args.writeValue && args.discardExactWidth > discard::kMaxExact) {
    throw std::invalid_argument("K6 reads at most " + std::to_string(discard::kMaxExact) +
                                " B path features per transition, the schema has " +
                                std::to_string(args.discardExactWidth));
  }
  if (hidden != 256) {
    throw std::invalid_argument("K6 has no specialisation for hidden " + std::to_string(hidden));
  }
  const dim3 grid((args.transitions + kTile - 1) / kTile);
  switch (args.keys) {
    case 39:
      launchKeys<39>(args, grid, stream);
      break;
    case 31:
      launchKeys<31>(args, grid, stream);
      break;
    default:
      throw std::invalid_argument("K6 is built for 31 or 39 keys per row, got " + std::to_string(args.keys));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
