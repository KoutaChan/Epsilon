// V4 candidate counting. All arithmetic before the output projection is local to one row or transition.
#include "prism_hip_capabilities.h"
#include "prism_hip_v4.h"
#include "prism_hip_candidate.h"
#include "prism_hip_mlp.h"

namespace prism::v4 {
namespace {

constexpr int kRows = 16;

struct CountArgs {
  Span<const uint16_t> state;
  Span<const uint8_t> codes;
  Span<const int32_t> rowColumn;
  CountWeights weights;
  uint16_t* tiles;
  uint16_t* delta;
  int rows, transitions, seats, tileKinds, keys, stride, rowWidth, rowFirstTransition;
};

// 16 tiles x 16 output channels per wave. The projection is shared by all transitions in the same row.
template <int H>
__global__ void countTileProjection(CountArgs args) {
  __shared__ uint16_t state[kRows][H + tile::kPad];
  const int lane = threadIdx.x;
  const int first = blockIdx.x * kRows;
  const int total = args.rows * args.tileKinds;
  Bounds bounds;
  for (int at = lane * 8; at < kRows * H; at += kWave * 8) {
    const int r = at / H;
    const int c = at % H;
    const int token = (1 + args.seats) * args.rows + min(first + r, total - 1);
    *reinterpret_cast<u32x4*>(&state[r][c]) = *reinterpret_cast<const u32x4*>(
        args.state.data + bounds.index(static_cast<size_t>(token) * H + c, 8, args.state.limit));
  }
  tile::barrier();
  f32x8 acc[1][1]{};
  const int column = lane & 15;
  const int output[1] = {column};
  tile::gemmTile<1, H>(acc, state, args.weights.tile.data, args.weights.tile.limit, H, 0, output, column, bounds);
  for (int i = 0; i < 8; ++i) {
    const int row = first + 2 * i + (lane >> 4);
    if (row < total) {
      args.tiles[row * kCountWidth + column] = bf16RoundActivation(acc[0][0][i]);
    }
  }
  bounds.trap();
}

template <int H>
__global__ void countTransitionProjection(CountArgs args) {
  __shared__ uint16_t count[kRows][kCountWidth + tile::kPad];
  __shared__ int rowIndex[kRows];
  const int tid = threadIdx.x;
  const int first = blockIdx.x * kRows;
  const int transition = first + tid / kCountWidth;
  const int channel = tid % kCountWidth;
  const int firstTile = args.keys - args.tileKinds;
  Bounds bounds;
  if (tid < kRows) {
    rowIndex[tid] = first + tid < args.transitions
                        ? candidate::transitionRow(args.rowColumn, args.rowWidth, args.rowFirstTransition, args.rows,
                                                   first + tid, bounds)
                        : 0;
  }
  tile::barrier();
  float sum = 0.0f;
  if (transition < args.transitions) {
    const size_t code = static_cast<size_t>(transition) * args.stride + firstTile;
    const size_t tile = static_cast<size_t>(rowIndex[tid / kCountWidth]) * args.tileKinds * kCountWidth;
    for (int j = 0; j < args.tileKinds; ++j) {
      const int id = args.codes.data[bounds.index(code + j, 1, args.codes.limit)] - args.weights.firstCode;
      const size_t at = bounds.index(static_cast<size_t>(id) * kCountWidth + channel, 1, args.weights.gate.limit);
      const float gate = 1.0f / (1.0f + expf(-args.weights.gate.data[at]));
      sum = __fadd_rn(sum, __fmul_rn(gate, bf16Float(args.tiles[tile + j * kCountWidth + channel])));
    }
  }
  count[tid / kCountWidth][channel] = bf16RoundActivation(sum);
  tile::barrier();
  constexpr int kColumns = H / 16 / tile::kWaves;
  const int lane = tid % kWave;
  const int column = lane & 15;
  const int wave = tid / kWave;
  int output[kColumns];
  for (int c = 0; c < kColumns; ++c) output[c] = (wave * kColumns + c) * 16 + column;
  f32x8 acc[1][kColumns]{};
  tile::gemmTile<kColumns, kCountWidth>(acc, count, args.weights.output.data, args.weights.output.limit, kCountWidth, 0,
                                        output, column, bounds);
  for (int i = 0; i < 8; ++i) {
    const int row = first + 2 * i + (lane >> 4);
    if (row < args.transitions) {
      for (int c = 0; c < kColumns; ++c) {
        args.delta[static_cast<size_t>(row) * H + output[c]] = bf16RoundActivation(acc[0][c][i]);
      }
    }
  }
  bounds.trap();
}

}  // namespace
}  // namespace prism::v4

namespace prism::v4 {

template <int H>
static Span<const uint16_t> launchCountProjection(const CandidateArgs& base, const CountWeights& weights,
                                                  uint16_t* workspace, int stride, hipStream_t stream) {
  const int tileRows = base.layout.rows * base.layout.tiles;
  uint16_t* delta = workspace + static_cast<size_t>(tileRows) * kCountWidth;
  CountArgs args{base.state,        base.code,
                 base.rowColumn,    weights,
                 workspace,         delta,
                 base.layout.rows,  base.transitions,
                 base.layout.seats, base.layout.tiles,
                 base.keys,         stride,
                 base.rowWidth,     base.rowFirstTransition};
  hipLaunchKernelGGL(countTileProjection<H>, dim3((tileRows + kRows - 1) / kRows), dim3(kWave), 0, stream, args);
  PRISM_HIP(hipGetLastError());
  hipLaunchKernelGGL(countTransitionProjection<H>, dim3((base.transitions + kRows - 1) / kRows), dim3(tile::kThreads),
                     0, stream, args);
  PRISM_HIP(hipGetLastError());
  return {delta, static_cast<size_t>(base.transitions) * H};
}

void launchCandidateCore(int hidden, const CandidateArgs& base, const CountWeights& count,
                         const SegmentWeights& segments, uint16_t* workspace, hipStream_t stream,
                         HotCandidateOffsets hot, Span<uint16_t> attentionState) {
  if (base.transitions == 0) return;
  const int stride = segments.layout.codeWidth;
  forCoreWidth(hidden, [&](auto width) {
    const auto projected = launchCountProjection<decltype(width)::value>(base, count, workspace, stride, stream);
    if (hot.codes.data != nullptr) {
      launchCandidateWithHotOffsets(hidden, base, projected, segments, hot, stream, attentionState);
    } else {
      launchCandidateWithOffsets(hidden, base, projected, segments, stream);
    }
  });
}

}  // namespace prism::v4
