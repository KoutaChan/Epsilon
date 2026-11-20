#pragma once

#include "gradients.h"

namespace prism {
namespace {

constexpr int kThreads = 128;

// 遷移ごとに、ゲートと射影済みの牌の積を牌番号順に FP32 の積和で足し、BF16 に丸める。
__global__ void countForward(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                             const int32_t* codes, uint16_t* output) {
  const int item = blockIdx.x * kThreads + threadIdx.x;
  const int transition = item / kCountWidth;
  if (transition >= shape.transitions) return;
#if PRISM_CHECKED
  // 逆伝播は行ごとの遷移の区間を二分探索で求めるので、行の番号は減らないこと。
  if (transition > 0 && rows[transition - 1] > rows[transition]) trainingTrap();
#endif
  const int col = item % kCountWidth;
  Bounds bounds;
  const size_t row = bounds.index(rows[transition], 1, shape.rows);
  float sum = 0.0f;
  for (int key = 0; key < shape.tiles; ++key) {
    const size_t code = bounds.index(codes[transition * shape.tiles + key], 1, shape.codes);
    const float gate = training::countGate(gates[code * kCountWidth + col]);
    const float tile = bf16Float(projected[(row * shape.tiles + key) * kCountWidth + col]);
    sum = fusedMultiplyAdd(gate, tile, sum);
  }
  output[item] = bf16RoundActivation(sum);
  bounds.trap();
}

__device__ int firstTransition(const int32_t* rows, int transitions, int row) {
  int first = 0;
  int last = transitions;
  while (first < last) {
    const int middle = first + (last - first) / 2;
    if (rows[middle] < row)
      first = middle + 1;
    else
      last = middle;
  }
  return first;
}

// 行ごとに 1 ブロックで、牌の勾配を上流の勾配とゲートの積の遷移順の積和で求める。ゲートの勾配は、FP32 に丸めた各項を共有メモリの
// 行の表に原子的に足し、0 でない成分を全体の表に原子的に足す（全体の表は先に 0 にしておく）。足す順はスレッドとブロックの到着順で決まる。
__global__ void countBackwardByRow(CountTraining shape, const uint16_t* projected, const float* gates,
                                   const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                   uint16_t* gradProjected, float* gradGate) {
  extern __shared__ float rowGate[];
  __shared__ int first, last;
  const int row = blockIdx.x;
  const int gateItems = shape.codes * kCountWidth;
  if (threadIdx.x == 0) {
    first = firstTransition(rows, shape.transitions, row);
    last = firstTransition(rows, shape.transitions, row + 1);
  }
  for (int g = threadIdx.x; g < gateItems; g += kThreads) rowGate[g] = 0.0f;
  shareGroupWrites();

  Bounds bounds;
  const int tileItems = shape.tiles * kCountWidth;
  for (int item = threadIdx.x; item < tileItems; item += kThreads) {
    const int key = item / kCountWidth;
    const int col = item % kCountWidth;
    const size_t tile = static_cast<size_t>(row) * tileItems + item;
    const float value = bf16Float(projected[tile]);
    float gradTile = 0.0f;
    for (int t = first; t < last; ++t) {
      const size_t code = bounds.index(codes[t * shape.tiles + key], 1, shape.codes);
      const size_t gi = code * kCountWidth + col;
      const float gate = training::countGate(gates[gi]);
      const float grad = bf16Float(gradOutput[t * kCountWidth + col]);
      gradTile = fusedMultiplyAdd(gate, grad, gradTile);
      atomicAdd(rowGate + gi, training::gateGradient(grad, value, gate));
    }
    gradProjected[tile] = bf16Round(gradTile);
  }
  shareGroupWrites();
  for (int g = threadIdx.x; g < gateItems; g += kThreads) {
    const float gradient = rowGate[g];
    if (gradient != 0.0f) atomicAdd(gradGate + g, gradient);
  }
  bounds.trap();
}
}  // namespace
}  // namespace prism
