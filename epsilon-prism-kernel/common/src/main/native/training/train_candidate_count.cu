#include "relation_encoding.h"
#include "training/training_launch.h"
#include "training_device.h"

#include <stdexcept>

#include "training/candidate_count.cuh"

namespace prism {
namespace {

void validate(CountTraining shape) {
  if (shape.rows <= 0 || shape.tiles <= 0 || shape.tiles > kMaxTiles || shape.codes <= 0 || shape.codes >= kU8Codes ||
      shape.width != kCountWidth || shape.transitions <= 0 ||
      static_cast<int64_t>(shape.transitions) * shape.tiles * kCountWidth > INT32_MAX) {
    throw std::invalid_argument("invalid candidate count training shape");
  }
}

}  // namespace

void launchCountTraining(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                         const int32_t* codes, uint16_t* output, gpu::Stream stream) {
  validate(shape);
  const size_t items = static_cast<size_t>(shape.transitions) * kCountWidth;
  countForward<<<dim3((items + kThreads - 1) / kThreads), dim3(kThreads), 0, stream>>>(shape, projected, gates, rows,
                                                                                       codes, output);
  PRISM_GPU(gpu::lastError());
}

void launchCountTrainingBackward(CountTraining shape, const uint16_t* projected, const float* gates,
                                 const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                 uint16_t* gradProjected, float* gradGate, gpu::Stream stream) {
  validate(shape);
  countProjectedGradient<<<dim3(shape.rows), dim3(kThreads), 0, stream>>>(shape, gates, rows, codes, gradOutput,
                                                                          gradProjected);
  PRISM_GPU(gpu::lastError());
  countGateGradient<<<dim3(shape.codes), dim3(kThreads), 0, stream>>>(shape, projected, gates, rows, codes, gradOutput,
                                                                      gradGate);
  PRISM_GPU(gpu::lastError());
}

}  // namespace prism
