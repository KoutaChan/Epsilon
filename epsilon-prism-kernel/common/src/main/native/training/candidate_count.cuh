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

// 各行の牌の勾配を、上流の勾配とゲートの積の遷移順の積和で求める。
__global__ void countProjectedGradient(CountTraining shape, const float* gates, const int32_t* rows,
                                       const int32_t* codes, const uint16_t* gradOutput, uint16_t* gradProjected) {
  __shared__ int first, last;
  const int row = blockIdx.x;
  if (threadIdx.x == 0) {
    first = firstTransition(rows, shape.transitions, row);
    last = firstTransition(rows, shape.transitions, row + 1);
  }
  shareGroupWrites();

  Bounds bounds;
  const int tileItems = shape.tiles * kCountWidth;
  for (int item = threadIdx.x; item < tileItems; item += kThreads) {
    const int key = item / kCountWidth;
    const int col = item % kCountWidth;
    const size_t tile = static_cast<size_t>(row) * tileItems + item;
    float gradTile = 0.0f;
    for (int t = first; t < last; ++t) {
      const size_t code = bounds.index(codes[t * shape.tiles + key], 1, shape.codes);
      const size_t gi = code * kCountWidth + col;
      const float gate = training::countGate(gates[gi]);
      const float grad = bf16Float(gradOutput[t * kCountWidth + col]);
      gradTile = fusedMultiplyAdd(gate, grad, gradTile);
    }
    gradProjected[tile] = bf16Round(gradTile);
  }
  bounds.trap();
}

// コードごとに 1 ブロックで、ゲートの各成分の勾配を求める。行ごとに牌を rowLanes 個ずつのまとまりに分け、まとまりの中は遷移・牌の順に、
// FP32 に丸めた各項を部分和に足す。行の部分和は行の番号の順に足し、0 の部分和は足さない。
__global__ void countGateGradient(CountTraining shape, const uint16_t* projected, const float* gates,
                                  const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                  float* gradGate) {
  constexpr int rowLanes = kThreads / kCountWidth;
  static_assert(kThreads % kCountWidth == 0);
  __shared__ float partial[kThreads];
  const int code = blockIdx.x;
  const int col = threadIdx.x % kCountWidth;
  const int lane = threadIdx.x / kCountWidth;
  const float gate = training::countGate(gates[code * kCountWidth + col]);
  float total = 0.0f;
  for (int begin = 0; begin < shape.rows; begin += rowLanes) {
    const int row = begin + lane;
    float sum = 0.0f;
    if (row < shape.rows) {
      const int first = firstTransition(rows, shape.transitions, row);
      const int last = firstTransition(rows, shape.transitions, row + 1);
      for (int start = 0; start < shape.tiles; start += rowLanes) {
        for (int t = first; t < last; ++t) {
          const float grad = bf16Float(gradOutput[t * kCountWidth + col]);
          for (int key = start; key < shape.tiles && key < start + rowLanes; ++key) {
            if (codes[t * shape.tiles + key] != code) continue;
            const size_t tile = static_cast<size_t>(row) * shape.tiles + key;
            sum = roundedAdd(sum, training::gateGradient(grad, bf16Float(projected[tile * kCountWidth + col]), gate));
          }
        }
      }
    }
    partial[threadIdx.x] = sum;
    shareGroupWrites();
    if (lane == 0) {
      for (int r = 0; r < rowLanes && begin + r < shape.rows; ++r) {
        const float value = partial[r * kCountWidth + col];
        if (value != 0.0f) total = roundedAdd(total, value);
      }
    }
    shareGroupWrites();
  }
  if (lane == 0) gradGate[code * kCountWidth + col] = total;
}
}  // namespace
}  // namespace prism
