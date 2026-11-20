#pragma once
#include "inference_device.h"

namespace prism {
__device__ __forceinline__ void trainingTrap() { __builtin_trap(); }
}  // namespace prism
