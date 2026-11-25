#pragma once

#include "inference_device.h"

namespace prism::device {
template <class T>
__device__ __forceinline__ Buffer boundedBuffer(const T* base, size_t elements) {
  return bufferResource(base, elements);
}
}
