// Training-time V4 count reduction (GEMMs still run on DJL's compute stream).
#include "prism_hip_relation.h"
#include "prism_hip_v4.h"

#include <stdexcept>

namespace prism::v4 {
namespace {

constexpr int kThreads = 128;

__global__ void countForward(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                             const int32_t* codes, uint16_t* output) {
  const int item = blockIdx.x * kThreads + threadIdx.x;
  const int transition = item / kCountWidth;
  if (transition >= shape.transitions) return;
  const int col = item % kCountWidth;
  Bounds bounds;
  const size_t row = bounds.index(rows[transition], 1, shape.rows);
  float sum = 0.0f;
  for (int key = 0; key < shape.tiles; ++key) {
    const size_t code = bounds.index(codes[transition * shape.tiles + key], 1, shape.codes);
    const float gate = 1.0f / (1.0f + expf(-gates[code * kCountWidth + col]));
    const float tile = bf16Float(projected[(row * shape.tiles + key) * kCountWidth + col]);
    sum = __fadd_rn(sum, __fmul_rn(gate, tile));
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

// BatchIndex groups transitions by row. Keep each tile gradient private and
// combine gate gradients in LDS before writing back to the shared table.
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
  __syncthreads();

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
      const float gate = 1.0f / (1.0f + expf(-gates[gi]));
      const float grad = bf16Float(gradOutput[t * kCountWidth + col]);
      gradTile = __fadd_rn(gradTile, __fmul_rn(grad, gate));
      atomicAdd(rowGate + gi, (grad * value) * (gate * (1.0f - gate)));
    }
    gradProjected[tile] = bf16Round(gradTile);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < gateItems; g += kThreads) {
    const float gradient = rowGate[g];
    if (gradient != 0.0f) atomicAdd(gradGate + g, gradient);
  }
  bounds.trap();
}

void validate(CountTraining shape) {
  if (shape.rows <= 0 || shape.tiles <= 0 || shape.tiles > kMaxTiles || shape.codes <= 0 || shape.codes >= kU8Codes ||
      shape.width != kCountWidth || shape.transitions <= 0 ||
      static_cast<int64_t>(shape.transitions) * shape.tiles * kCountWidth > INT32_MAX) {
    throw std::invalid_argument("invalid V4 candidate count training shape");
  }
}

}  // namespace

void launchCountTraining(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                         const int32_t* codes, uint16_t* output, hipStream_t stream) {
  validate(shape);
  const size_t items = static_cast<size_t>(shape.transitions) * kCountWidth;
  hipLaunchKernelGGL(countForward, dim3((items + kThreads - 1) / kThreads), dim3(kThreads), 0, stream, shape, projected,
                     gates, rows, codes, output);
  PRISM_HIP(hipGetLastError());
}

void launchCountTrainingBackward(CountTraining shape, const uint16_t* projected, const float* gates,
                                 const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                 uint16_t* gradProjected, float* gradGate, hipStream_t stream) {
  validate(shape);
  const size_t gateBytes = static_cast<size_t>(shape.codes) * kCountWidth * sizeof(float);
  PRISM_HIP(hipMemsetAsync(gradGate, 0, gateBytes, stream));
  hipLaunchKernelGGL(countBackwardByRow, dim3(shape.rows), dim3(kThreads), gateBytes, stream, shape, projected, gates,
                     rows, codes, gradOutput, gradProjected, gradGate);
  PRISM_HIP(hipGetLastError());
}

}  // namespace prism::v4
