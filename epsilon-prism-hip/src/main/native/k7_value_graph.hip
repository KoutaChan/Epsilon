// K7（value_graph）: 16 行の作業群（256 スレッド = 8 波）。eager の ValueGraphHead.scores（推論。錨の勾配の扱いは学習だけなので無い）。
//
//   e_a  = Σ_t softmax_t(B(t)/τ)·z_t（手の代表の遷移の上、FP32。遷移が 1 つなら z_t）
//   基準 = ダマ x の遷移（立直）、PASS の遷移（応答・槍槓）、ē = Σ_代表の打牌 softmax(B/τ)·z（手番の暗槓・加槓・九種・ツモ）
//   u    = float(⌊[⌊N(e_a)⌉; ⌊N(基準)⌉; h]·W_gᵀ + c⌉) + β_k + U_k·x^G     （門の行列積は WMMA、門を 16 ずつ）
//   Δ    = Σ_g silu(u_g)·w_k,g + b_k + u_k·x^G
//   s    = B（打牌）、B(ダマ) + Δ（立直）、Δ（応答）、M + Δ（手番の門、M は打牌と立直の s の最大）、0（PASS）
//
// s は 0 に足して（−0 を +0 にして）要求の順に scores へ書く（eager の合成と同じ値。NaN は 0x7FC00000 の 1 つにそろえる）。縮約は全て
// 手か行の中で、同じ行の s は塊の組み方に依らずビットで同じ。
//
// 待ちを減らす形: 門の行列積は W_g の断片を kGateDepth 歩ぶん環に持って読み込みを飛ばし続け（gateProduct）、行列積に
// 依らない c・β_k・w_k,g の読み込みは行列積の前に出し、U_k·x^G は x を外の輪にして 8 行ぶんの読み込みを一度に飛ばす。合成は
// スレッドが 1 手を受け持ち、手番の門だけが LDS の s から行の M を手の順に取る。作業群の待ち合わせは LDS の順序だけ（tile::barrier）:
// K7 が全体メモリに書くのは scores だけで、カーネルの中では読まない。
#include "prism_hip_mlp.h"

namespace prism {
namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / kWave;
constexpr int kRowsPerGroup = 16;
constexpr int kMaxActionsPerRow = 32;  // 合法手の上限（Limits.MAX_LEGAL_ACTIONS）
constexpr int kMaxActions = kRowsPerGroup * kMaxActionsPerRow;
constexpr int kGateTile = 16;
constexpr int kMaxExact = 16;
constexpr int kGateDepth = 12;  // 門の行列積で読み込みが飛んでいる歩（環の B の断片）

template <int H>
struct alignas(16) ValueGraphShared {
  uint16_t input[kGateTile][3 * H + 8];
  float delta[kMaxActions];
  float score[kMaxActions];  // 合成の s（手番の門の M のため）
  float exact[kGateTile][kMaxExact];
  float partial[kWaves][kGateTile];
  int kind[kGateTile];
  short gateAction[kMaxActions];
  uint8_t composition[kMaxActions];  // 手の作り方（Composition）
  int gateCount;
};

// 行列の値を読む小さな道具（検査つきの組み立ての上限つき）。
__device__ __forceinline__ int cell(const Span<const int32_t>& table, int width, int index, int field, Bounds& bounds) {
  return table.data[bounds.index(static_cast<size_t>(index) * width + field, 1, table.limit)];
}

// 手の役割（範囲の外は検査つきの組み立てで止め、通常の組み立てでは表の外を引かないように詰める）。
__device__ __forceinline__ int roleOf(const ValueGraphArgs& args, int action, Bounds& bounds) {
  const int role = cell(args.actionColumn, args.actionWidth, action, args.actionRole, bounds);
  (void)bounds.index(static_cast<size_t>(role), 1, static_cast<size_t>(args.roles));
  return min(max(role, 0), kMaxRoles - 1);
}

// M に入る手（打牌と立直）。
__device__ __forceinline__ bool pivotal(int composition) {
  return composition == static_cast<int>(Composition::kDiscardValue) ||
         composition == static_cast<int>(Composition::kRiichiGate);
}

// scores に書く値。K7 の FP32 の演算（WMMA・和・最大）が作る NaN は符号と中身が決まらない（x − NaN は 0xFFC00000 になりうる）ので、
// 呼び手に見える NaN は 0x7FC00000 の 1 つにそろえる（NaN でない値はそのまま）。
__device__ __forceinline__ float canonicalScore(float score) {
  return __builtin_isnan(score) ? __builtin_bit_cast(float, 0x7fc00000u) : score;
}

// lane が持つ 8 列について、遷移 [first, first + count) の softmax(B/τ) で z を集める（FP32、遷移の順。積と和は分けて丸める）。
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

// ⌊N(e)⌉ を門の入力の行 dst に書く（波の 32 lane が 8 列ずつ。統計・利得は FP32）。
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

// 門の行列積 A·W_g[unit]ᵀ（A は LDS の門の入力 16 行、lane は行 lanes を読む）。W_g の B の断片を kGateDepth 歩ぶん環に持ち、歩 k の積の
// 直後に同じ枠へ歩 k + kGateDepth の断片を読む（K の外に出る分は最後の歩に寄せて分岐をなくし、使わない）。sched_barrier で読み込みを
// 積の直後に留め（無いと輪の中に vmcnt(0) が入る）、外の輪は畳む（全て展開すると spill する）。積は K の順のままなので、1 歩ずつ
// 掛けたものとビットで同じ。
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

  // 1. 門の手を手の順に並べる（波 0 が 32 手ずつ、ballot の前置和）。
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

  // 2. 門を 16 ずつ: 入力の行を作り（波が 2 行）、W_g を WMMA で掛け、Δ を出す。
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
        // ē: 行の代表の打牌（手の順）の softmax(B/τ) で z を集める。
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
        // 立直はダマ x、応答は PASS（基準の手の代表の最初の遷移）。
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

    // u と silu(u)·w の和。波は G の列の断片 n = wave, wave + 8, … を受け持つ。C の断片の行 v は門 2v + half。
    float rowSum[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      rowSum[v] = 0.0f;
    }
    for (int n = wave; n < args.gate / 16; n += kWaves) {
      const int unit = n * 16 + lanes;
      // 行列積に依らない c・β_k・w_k,g の読み込みは先に出す（積の間に届く）。
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
      // U_k·x^G: x を外の輪にして 8 行の読み込みを一度に飛ばす（行ごとの和は x の順のまま）。
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

  // 3. 合成（スレッドが 1 手）。手番の門のほかは s を書き、s と作り方を LDS に置く。手番の門は行の手を手の順に見て M（打牌と立直の
  //    s の最大）を取る（fmaxf を掛ける順は 1 スレッドが行を回したときと同じ）。
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
