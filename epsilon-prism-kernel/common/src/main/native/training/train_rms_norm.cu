#include <cstdint>
#include <stdexcept>
#include <string>

#include "training/training_launch.h"
#include "training_device.h"

#include "training/rms_norm.cuh"

namespace prism {
namespace {

int chunksOf(int width) { return (width / kChunk + kWave - 1) / kWave; }

template <class T>
void requireAligned(const T* data, const char* name) {
  if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
    throw std::invalid_argument(std::string("RMSNorm ") + name + " must be aligned to 16 bytes");
  }
}

}  // namespace

void checkNormShape(const NormShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(shape.rows > 0, "RMSNorm needs rows, got " + std::to_string(shape.rows));
  require(shape.width > 0 && shape.width % kChunk == 0 && shape.width <= kNormMaxWidth,
          "RMSNorm width must be a multiple of 8 in 8.." + std::to_string(kNormMaxWidth) + ", got " +
              std::to_string(shape.width));
  require(shape.epsilon > 0.0f, "RMSNorm epsilon must be positive");
}

void normForward(const NormShape& shape, const NormForward& arrays, gpu::Stream stream) {
  checkNormShape(shape);
  requireAligned(arrays.input, "input");
  requireAligned(arrays.gain, "gain");
  requireAligned(arrays.out, "output");
  const size_t elements = static_cast<size_t>(shape.rows) * shape.width;
  ForwardArgs args{};
  args.shape = shape;
  args.input = {arrays.input, elements};
  args.gain = {arrays.gain, static_cast<size_t>(shape.width)};
  args.out = {arrays.out, elements};
  args.inverseRms = {arrays.inverseRms, static_cast<size_t>(shape.rows)};
  const dim3 grid((shape.rows + kWaves - 1) / kWaves);
  switch (chunksOf(shape.width)) {
    case 1:
      normForwardKernel<1><<<grid, dim3(kThreads), 0, stream>>>(args);
      break;
    case 2:
      normForwardKernel<2><<<grid, dim3(kThreads), 0, stream>>>(args);
      break;
    case 3:
      normForwardKernel<3><<<grid, dim3(kThreads), 0, stream>>>(args);
      break;
    default:
      normForwardKernel<kMaxChunks><<<grid, dim3(kThreads), 0, stream>>>(args);
      break;
  }
  PRISM_GPU(gpu::lastError());
}

void normBackward(const NormShape& shape, const NormBackward& arrays, gpu::Stream stream) {
  checkNormShape(shape);
  requireAligned(arrays.input, "input");
  requireAligned(arrays.gain, "gain");
  requireAligned(arrays.gradOut, "output gradient");
  requireAligned(arrays.gradInput, "input gradient");
  const size_t elements = static_cast<size_t>(shape.rows) * shape.width;
  const int blocks = (shape.rows + kNormGainRows - 1) / kNormGainRows;
  BackwardArgs args{};
  args.shape = shape;
  args.input = {arrays.input, elements};
  args.gain = {arrays.gain, static_cast<size_t>(shape.width)};
  args.inverseRms = {arrays.inverseRms, static_cast<size_t>(shape.rows)};
  args.gradOut = {arrays.gradOut, elements};
  args.gradInput = {arrays.gradInput, elements};
  args.gradGain = {arrays.gradGain, static_cast<size_t>(blocks) * shape.width};
  switch (chunksOf(shape.width)) {
    case 1:
      normBackwardKernel<1><<<dim3(blocks), dim3(kThreads), 0, stream>>>(args);
      break;
    case 2:
      normBackwardKernel<2><<<dim3(blocks), dim3(kThreads), 0, stream>>>(args);
      break;
    case 3:
      normBackwardKernel<3><<<dim3(blocks), dim3(kThreads), 0, stream>>>(args);
      break;
    default:
      normBackwardKernel<kMaxChunks><<<dim3(blocks), dim3(kThreads), 0, stream>>>(args);
      break;
  }
  PRISM_GPU(gpu::lastError());
}

}  // namespace prism
