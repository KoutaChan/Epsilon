// K7（value_graph）: eager の ValueGraphHead.scores と同じ値を推論用に計算する。アンカーの勾配は学習でしか使わないので扱わない。
// ワークグループは 16 行を受け持つ（256 スレッド = 8 ウェーブ）。
//
//   e_a  = Σ_t softmax_t(B(t)/τ)·z_t（手の代表の遷移について FP32 で計算する。遷移が 1 つなら z_t）
//   基準 = ダマ x の遷移（立直）、PASS の遷移（応答・槍槓）、ē = Σ_代表の打牌 softmax(B/τ)·z（手番の暗槓・加槓・九種・ツモ）
//   u    = float(⌊[⌊N(e_a)⌉; ⌊N(基準)⌉; h]·W_gᵀ + c⌉) + β_k + U_k·x^G     （ゲートの行列積は WMMA で、ゲートを 16 個ずつ）
//   Δ    = Σ_g silu(u_g)·w_k,g + b_k + u_k·x^G
//   s    = B（打牌）、B(ダマ) + Δ（立直）、Δ（応答）、M + Δ（手番のゲート。M は打牌と立直の s の最大値）、0（PASS）
//
// s には 0 を足して −0 を +0 にそろえ、要求の順に scores へ書く。値は eager の合成と同じで、NaN は 0x7FC00000 の 1 つにそろえる。
// リダクションは全て手か行の中で閉じるので、同じ行の s はバッチの組み方によらずビットまで同じ。
//
// 待ち時間を減らす工夫:
// - ゲートの行列積では、W_g のフラグメントを kGateDepth ステップ分リングに持ち、読み込みを常に先行させる（gateProduct）。
// - 行列積に依らない c・β_k・w_k,g の読み込みは、行列積の前に発行する。
// - U_k·x^G は x を外側のループにして、8 行分の読み込みを一度に発行する。
// - 合成ではスレッドが 1 手を受け持つ。手番のゲートだけが、LDS に置いた s から行の M を手の順に求める。
// ワークグループのバリアは LDS の順序だけを保証すればよい（tile::barrier）。K7 がグローバルメモリに書くのは scores だけで、
// カーネルの中では読まないため。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kRowsPerGroup = 16;
constexpr int kMaxActionsPerRow = 32;  // 1 行の合法手の上限（Limits.MAX_LEGAL_ACTIONS）
constexpr int kMaxActions = kRowsPerGroup * kMaxActionsPerRow;
constexpr int kGateTile = 16;
constexpr int kMaxExact = 16;
constexpr int kGateDepth = 12;  // ゲートの行列積で読み込みを先行させるステップ数（リングに持つ B のフラグメントの数）

template <int H>
struct alignas(16) ValueGraphShared {
  uint16_t input[kGateTile][3 * H + 8];
  float delta[kMaxActions];
  float score[kMaxActions];  // 合成した s（手番のゲートの M を求めるのに使う）
  float exact[kGateTile][kMaxExact];
  float partial[kWaves][kGateTile];
  int kind[kGateTile];
  short gateAction[kMaxActions];
  uint8_t composition[kMaxActions];  // 手の値の作り方（Composition）
  int gateCount;
};

// 表の値を 1 つ読むヘルパー（範囲検査つきのビルドでは上限を確かめる）。
__device__ __forceinline__ int cell(const Span<const int32_t>& table, int width, int index, int field, Bounds& bounds) {
  return table.data[bounds.index(static_cast<size_t>(index) * width + field, 1, table.limit)];
}

// 手の種類（role）を返す。範囲の外の値は、範囲検査つきのビルドでは止め、通常のビルドでは表の外を引かないように範囲の端に寄せる。
__device__ __forceinline__ int roleOf(const ValueGraphArgs& args, int action, Bounds& bounds) {
  const int role = cell(args.actionColumn, args.actionWidth, action, args.actionRole, bounds);
  (void)bounds.index(static_cast<size_t>(role), 1, static_cast<size_t>(args.roles));
  return min(max(role, 0), kMaxRoles - 1);
}

// M の対象になる手（打牌と立直）か。
__device__ __forceinline__ bool pivotal(int composition) {
  return composition == static_cast<int>(Composition::kDiscardValue) ||
         composition == static_cast<int>(Composition::kRiichiGate);
}

// scores に書く値。K7 の FP32 の演算（WMMA・和・最大値）が作る NaN は符号と中身が決まらない（x − NaN は 0xFFC00000 になりうる）ので、
// 呼び出し元に見える NaN は 0x7FC00000 の 1 つにそろえる（NaN でない値はそのまま）。
__device__ __forceinline__ float canonicalScore(float score) {
  return __builtin_isnan(score) ? __builtin_bit_cast(float, 0x7fc00000u) : score;
}

// lane が持つ 8 列について、遷移 [first, first + count) の z を softmax(B/τ) で重み付けして足す。FP32 で遷移の順に足し、積と和は
// 別々に丸める。
template <int H>
__device__ __forceinline__ void aggregate(const ValueGraphArgs& args, int first, int count, int col0, float (&e)[8],
                                          Bounds& bounds) {
  if (count == 1) {
    float values[8];
    unpack8(*reinterpret_cast<const u32x4*>(
                args.transition.data + bounds.index(static_cast<size_t>(first) * H + col0, 8, args.transition.limit)),
            values);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      e[c] = values[c];
    }
    return;
  }
  float maximum = -INFINITY;
  for (int i = 0; i < count; ++i) {
    maximum = fmaxf(maximum, args.value.data[bounds.index(first + i, 1, args.value.limit)] / args.tau);
  }
  float total = 0.0f;
  for (int i = 0; i < count; ++i) {
    total += expf(args.value.data[bounds.index(first + i, 1, args.value.limit)] / args.tau - maximum);
  }
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    e[c] = 0.0f;
  }
  for (int i = 0; i < count; ++i) {
    const float weight = expf(args.value.data[bounds.index(first + i, 1, args.value.limit)] / args.tau - maximum) / total;
    float values[8];
    unpack8(*reinterpret_cast<const u32x4*>(args.transition.data +
                                             bounds.index(static_cast<size_t>(first + i) * H + col0, 8,
                                                          args.transition.limit)),
            values);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      e[c] = __fadd_rn(e[c], __fmul_rn(weight, values[c]));
    }
  }
}

// ⌊N(e)⌉ をゲートの入力の行に書く。ウェーブの 32 lane が 8 列ずつ受け持ち、二乗平均とゲインの掛け算は FP32 で行う。
template <int H>
__device__ __forceinline__ void normalizeInto(uint16_t* dst, const float (&e)[8], int col0, const ValueGraphArgs& args,
                                              Bounds& bounds) {
  float sum = 0.0f;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    sum += e[c] * e[c];
  }
  sum = waveSum(sum);
  const float r = rsqrtf(sum / static_cast<float>(H) + args.epsilon);
  const size_t g = bounds.index(col0, 8, args.gateNorm.limit);
  float values[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    values[c] = (e[c] * r) * args.gateNorm.data[g + c];
  }
  *reinterpret_cast<u32x4*>(dst + col0) = pack8(values);
}

// ゲートの行列積 A·W_g[unit]ᵀ。A は LDS に置いたゲートの入力 16 行で、各 lane は lane & 15 の行を読む。W_g の B のフラグメントを
// kGateDepth ステップ分リングに持ち、ステップ k の積の直後に、同じ位置へステップ k + kGateDepth のフラグメントを読む。K の外に出る分は
// 最後のステップを読み直して分岐をなくし、その値は使わない。sched_barrier で読み込みを積の直後に留める（無いとループの中に vmcnt(0) が
// 入る）。外側のループは展開しない（全て展開すると spill する）。積は K の順のままなので、1 ステップずつ掛けた結果とビットまで同じ。
template <int H>
__device__ __forceinline__ f32x8 gateProduct(const uint16_t (*input)[3 * H + 8], const ValueGraphArgs& args, int unit,
                                             int lanes, Bounds& bounds) {
  constexpr int kInput = 3 * H;
  constexpr int kSteps = kInput / 16;
  static_assert(kSteps % kGateDepth == 0, "the ring covers the K steps in rounds of kGateDepth");
  bf16x16 b[kGateDepth];
#pragma unroll
  for (int d = 0; d < kGateDepth; ++d) {
    b[d] = tile::weightFragment(args.gateHidden.data, args.gateHidden.limit, kInput, unit, d * 16, bounds);
  }
  f32x8 acc = f32x8{};
#pragma unroll 1
  for (int k0 = 0; k0 < kSteps; k0 += kGateDepth) {
#pragma unroll
    for (int d = 0; d < kGateDepth; ++d) {
      const int k = k0 + d;
      acc = mma(fragment(&input[lanes][k * 16]), b[d], acc);
      b[d] = tile::weightFragment(args.gateHidden.data, args.gateHidden.limit, kInput, unit,
                                  min(k + kGateDepth, kSteps - 1) * 16, bounds);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  return acc;
}

template <int H>
__global__ __launch_bounds__(kThreads) void valueGraphKernel(ValueGraphArgs args) {
  static_assert(H / kWave == 8, "one lane holds 8 columns of a representation");
  static_assert(sizeof(ValueGraphShared<H>) <= 64 * 1024, "LDS of one work group is 64 KiB");
  __shared__ ValueGraphShared<H> shared;

  const int tid = threadIdx.x;
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int half = lane >> 4;
  const int lanes = lane & 15;
  const int col0 = lane * 8;
  const int row0 = blockIdx.x * kRowsPerGroup;
  const int rowsHere = min(kRowsPerGroup, args.rows - row0);
  Bounds bounds;

  const int firstAction = cell(args.rowColumn, args.rowWidth, row0, args.rowFirstAction, bounds);
  const int lastRow = row0 + rowsHere - 1;
  const int endAction = cell(args.rowColumn, args.rowWidth, lastRow, args.rowFirstAction, bounds) +
                        cell(args.rowColumn, args.rowWidth, lastRow, args.rowActions, bounds);
  const int actionCount = min(endAction - firstAction, kMaxActions);
  (void)bounds.index(static_cast<size_t>(endAction - firstAction), 0, kMaxActions);

  // 1. ゲートを使う手を、手の順に並べる。ウェーブ 0 が 32 手ずつ見て、ballot の累積和で位置を求める。
  if (wave == 0) {
    int running = 0;
    for (int base = 0; base < actionCount; base += kWave) {
      const int local = base + lane;
      bool isGate = false;
      if (local < actionCount) {
        const int role = cell(args.actionColumn, args.actionWidth, firstAction + local, args.actionRole, bounds);
        isGate = role >= 0 && role < args.roles && args.gateKind[role] >= 0;
      }
      const unsigned long long mask = __ballot(isGate);
      const int position = running + __popcll(mask & ((1ull << lane) - 1ull));
      if (isGate) {
        shared.gateAction[position] = static_cast<short>(local);
      }
      running += __popcll(mask);
    }
    if (lane == 0) {
      shared.gateCount = running;
    }
  }
  tile::barrier();
  const int gateCount = shared.gateCount;

  // 2. ゲートを 16 個ずつ処理する。ウェーブごとに入力の行を 2 行作り、W_g を WMMA で掛けて Δ を求める。
  for (int tile0 = 0; tile0 < gateCount; tile0 += kGateTile) {
    for (int gr = 2 * wave; gr < 2 * wave + 2; ++gr) {
      uint16_t* dst = shared.input[gr];
      const int gi = tile0 + gr;
      if (gi >= gateCount) {
        const u32x4 zero = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int part = 0; part < 3; ++part) {
          *reinterpret_cast<u32x4*>(dst + part * H + col0) = zero;
        }
        if (lane < kMaxExact) {
          shared.exact[gr][lane] = 0.0f;
        }
        if (lane == 0) {
          shared.kind[gr] = 0;
        }
        continue;
      }
      const int action = firstAction + shared.gateAction[gi];
      const int role = roleOf(args, action, bounds);
      const int composition = args.composition[role];
      const int representative =
          cell(args.actionColumn, args.actionWidth, action, args.actionRepresentative, bounds);
      float e[8];
      aggregate<H>(args, cell(args.actionColumn, args.actionWidth, representative, args.actionFirstTransition, bounds),
                   cell(args.actionColumn, args.actionWidth, representative, args.actionTransitions, bounds), col0, e,
                   bounds);
      normalizeInto<H>(dst, e, col0, args, bounds);
      // 基準
      const int row = cell(args.actionColumn, args.actionWidth, action, args.actionRow, bounds);
      if (composition == static_cast<int>(Composition::kTurnGate)) {
        // ē: 行の中の代表の打牌について、z を softmax(B/τ) で重み付けして足す（手の順）。
        const int rowFirst = cell(args.rowColumn, args.rowWidth, row, args.rowFirstAction, bounds);
        const int rowCount = cell(args.rowColumn, args.rowWidth, row, args.rowActions, bounds);
        float maximum = -INFINITY;
        for (int a = rowFirst; a < rowFirst + rowCount; ++a) {
          if (args.composition[roleOf(args, a, bounds)] == static_cast<int>(Composition::kDiscardValue) &&
              cell(args.actionColumn, args.actionWidth, a, args.actionRepresentative, bounds) == a) {
            const int t = cell(args.actionColumn, args.actionWidth, a, args.actionFirstTransition, bounds);
            maximum = fmaxf(maximum, args.value.data[bounds.index(t, 1, args.value.limit)] / args.tau);
          }
        }
        float total = 0.0f;
        for (int a = rowFirst; a < rowFirst + rowCount; ++a) {
          if (args.composition[roleOf(args, a, bounds)] == static_cast<int>(Composition::kDiscardValue) &&
              cell(args.actionColumn, args.actionWidth, a, args.actionRepresentative, bounds) == a) {
            const int t = cell(args.actionColumn, args.actionWidth, a, args.actionFirstTransition, bounds);
            total += expf(args.value.data[bounds.index(t, 1, args.value.limit)] / args.tau - maximum);
          }
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          e[c] = 0.0f;
        }
        for (int a = rowFirst; a < rowFirst + rowCount; ++a) {
          if (args.composition[roleOf(args, a, bounds)] == static_cast<int>(Composition::kDiscardValue) &&
              cell(args.actionColumn, args.actionWidth, a, args.actionRepresentative, bounds) == a) {
            const int t = cell(args.actionColumn, args.actionWidth, a, args.actionFirstTransition, bounds);
            const float weight = expf(args.value.data[bounds.index(t, 1, args.value.limit)] / args.tau - maximum) / total;
            float values[8];
            unpack8(*reinterpret_cast<const u32x4*>(args.transition.data +
                                                     bounds.index(static_cast<size_t>(t) * H + col0, 8,
                                                                  args.transition.limit)),
                    values);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
              e[c] = __fadd_rn(e[c], __fmul_rn(weight, values[c]));
            }
          }
        }
      } else {
        // 立直の基準はダマ x、応答の基準は PASS。基準の手の代表について、最初の遷移を使う。
        const int base = cell(args.actionColumn, args.actionWidth, action, args.actionBase, bounds);
        const int baseRepresentative =
            cell(args.actionColumn, args.actionWidth, max(base, 0), args.actionRepresentative, bounds);
        const int t =
            cell(args.actionColumn, args.actionWidth, baseRepresentative, args.actionFirstTransition, bounds);
        float values[8];
        unpack8(*reinterpret_cast<const u32x4*>(
                    args.transition.data + bounds.index(static_cast<size_t>(t) * H + col0, 8, args.transition.limit)),
                values);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          e[c] = values[c];
        }
      }
      normalizeInto<H>(dst + H, e, col0, args, bounds);
      // h = y_G[行]
      *reinterpret_cast<u32x4*>(dst + 2 * H + col0) = *reinterpret_cast<const u32x4*>(
          args.state.data + bounds.index(static_cast<size_t>(row) * H + col0, 8, args.state.limit));
      if (lane < kMaxExact) {
        shared.exact[gr][lane] =
            lane < args.actionExactWidth
                ? args.actionExact.data[bounds.index(static_cast<size_t>(action) * args.actionExactWidth + lane, 1,
                                                     args.actionExact.limit)]
                : 0.0f;
      }
      if (lane == 0) {
        shared.kind[gr] = args.gateKind[role];
      }
    }
    tile::barrier();

    // u を求め、silu(u)·w を足す。各ウェーブは G の列のフラグメントを、自分の番号から 8 個おきに受け持つ。C のフラグメントの要素 v は、
    // ゲート 2v + (lane >> 4) の値。
    float rowSum[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      rowSum[v] = 0.0f;
    }
    for (int n = wave; n < args.gate / 16; n += kWaves) {
      const int unit = n * 16 + lanes;
      // 行列積に依らない c・β_k・w_k,g の読み込みは先に発行する（積を計算する間に届く）。
      int kinds[8];
      float offsets[8];
      float outputs[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        kinds[v] = shared.kind[2 * v + half];
        offsets[v] = tile::load(args.kindOffset.data, args.kindOffset.limit, kinds[v] * args.gate + unit, bounds);
        outputs[v] = tile::load(args.gateOutput.data, args.gateOutput.limit, kinds[v] * args.gate + unit, bounds);
      }
      const float bias = tile::load(args.gateHiddenBias.data, args.gateHiddenBias.limit, unit, bounds);
      const f32x8 acc = gateProduct<H>(shared.input, args, unit, lanes, bounds);
      // U_k·x^G: x を外側のループにして、8 行分の読み込みを一度に発行する（行ごとの和は x の順のまま）。
      float projected[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        projected[v] = 0.0f;
      }
      for (int x = 0; x < args.actionExactWidth; ++x) {
        float weights[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          weights[v] = tile::load(args.exactProjection.data, args.exactProjection.limit,
                                  (kinds[v] * args.gate + unit) * args.actionExactWidth + x, bounds);
        }
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          projected[v] = __fadd_rn(projected[v], __fmul_rn(weights[v], shared.exact[2 * v + half][x]));
        }
      }
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const float u = (bf16Float(bf16Round(acc[v] + bias)) + offsets[v]) + projected[v];
        rowSum[v] += silu(u) * outputs[v];
      }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float sum = halfSum(rowSum[v]);
      if (lanes == 0) {
        shared.partial[wave][2 * v + half] = sum;
      }
    }
    tile::barrier();
    if (tid < kGateTile && tile0 + tid < gateCount) {
      const int kind = shared.kind[tid];
      float delta = 0.0f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w) {
        delta += shared.partial[w][tid];
      }
      float direct = 0.0f;
      for (int x = 0; x < args.actionExactWidth; ++x) {
        direct = __fadd_rn(
            direct, __fmul_rn(shared.exact[tid][x],
                              tile::load(args.exactOutput.data, args.exactOutput.limit,
                                         kind * args.actionExactWidth + x, bounds)));
      }
      delta = (delta + tile::load(args.gateOutputBias.data, args.gateOutputBias.limit, kind, bounds)) + direct;
      shared.delta[shared.gateAction[tile0 + tid]] = delta;
    }
    tile::barrier();
  }

  // 3. 合成。スレッドが 1 手ずつ受け持つ。手番のゲート以外の手は s を scores に書き、どの手も s と作り方を LDS に置く。手番のゲートは
  //    行の手を手の順に見て、M（打牌と立直の s の最大値）を求める。fmaxf を取る順は、1 スレッドが行を順に見たときと同じ。
  for (int local = tid; local < actionCount; local += kThreads) {
    const int a = firstAction + local;
    const int composition = args.composition[roleOf(args, a, bounds)];
    float score = 0.0f;
    if (composition == static_cast<int>(Composition::kDiscardValue)) {
      const int representative = cell(args.actionColumn, args.actionWidth, a, args.actionRepresentative, bounds);
      const int t = cell(args.actionColumn, args.actionWidth, representative, args.actionFirstTransition, bounds);
      score = args.value.data[bounds.index(t, 1, args.value.limit)];
    } else if (composition == static_cast<int>(Composition::kRiichiGate)) {
      const int base = cell(args.actionColumn, args.actionWidth, a, args.actionBase, bounds);
      const int representative =
          cell(args.actionColumn, args.actionWidth, max(base, 0), args.actionRepresentative, bounds);
      const int t = cell(args.actionColumn, args.actionWidth, representative, args.actionFirstTransition, bounds);
      score = args.value.data[bounds.index(t, 1, args.value.limit)] + shared.delta[local];
    } else if (composition == static_cast<int>(Composition::kResponseGate)) {
      score = shared.delta[local];
    }
    score = 0.0f + score;
    shared.score[local] = score;
    shared.composition[local] = static_cast<uint8_t>(composition);
    if (composition != static_cast<int>(Composition::kTurnGate)) {
      args.scores.data[bounds.index(a, 1, args.scores.limit)] = canonicalScore(score);
    }
  }
  tile::barrier();
  for (int local = tid; local < actionCount; local += kThreads) {
    if (shared.composition[local] != static_cast<int>(Composition::kTurnGate)) {
      continue;
    }
    const int a = firstAction + local;
    const int row = cell(args.actionColumn, args.actionWidth, a, args.actionRow, bounds);
    const int rowFirst = cell(args.rowColumn, args.rowWidth, row, args.rowFirstAction, bounds) - firstAction;
    const int rowEnd = min(rowFirst + cell(args.rowColumn, args.rowWidth, row, args.rowActions, bounds), actionCount);
    float pivot = 0.0f;
    bool havePivot = false;
    for (int b = max(rowFirst, 0); b < rowEnd; ++b) {
      if (pivotal(shared.composition[b])) {
        pivot = havePivot ? fmaxf(pivot, shared.score[b]) : shared.score[b];
        havePivot = true;
      }
    }
    args.scores.data[bounds.index(a, 1, args.scores.limit)] =
        canonicalScore(0.0f + ((havePivot ? pivot : 0.0f) + shared.delta[local]));
  }
  bounds.trap();
}

}  // namespace

void launchValueGraph(int hidden, const ValueGraphArgs& args, hipStream_t stream) {
  if (args.rows <= 0) {
    return;
  }
  if (args.actionExactWidth > kMaxExact) {
    throw std::invalid_argument("K7 holds at most " + std::to_string(kMaxExact) + " gate path features, got " +
                                std::to_string(args.actionExactWidth));
  }
  const dim3 grid((args.rows + kRowsPerGroup - 1) / kRowsPerGroup);
  switch (hidden) {
    case 256:
      hipLaunchKernelGGL(valueGraphKernel<256>, grid, dim3(kThreads), 0, stream, args);
      break;
    default:
      throw std::invalid_argument("K7 has no specialisation for hidden " + std::to_string(hidden));
  }
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism
