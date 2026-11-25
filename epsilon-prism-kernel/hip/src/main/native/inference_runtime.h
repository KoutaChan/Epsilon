#pragma once

#include "prism_hip_common.h"

namespace prism::gpu {
using Stream = hipStream_t;
using Event = hipEvent_t;
using Status = hipError_t;
inline constexpr auto outOfMemory = hipErrorOutOfMemory;
inline constexpr auto success = hipSuccess;
inline constexpr auto memoryInfo = hipMemGetInfo;
inline constexpr auto getDevice = hipGetDevice;
inline constexpr auto setDevice = hipSetDevice;
inline constexpr auto synchronize = hipStreamSynchronize;
inline constexpr auto destroyEvent = hipEventDestroy;
inline constexpr auto record = hipEventRecord;
inline constexpr auto release = hipFree;
template <class T>
inline Status allocate(T** p, size_t bytes) {
  return hipMalloc(p, bytes);
}
template <class T>
inline Status allocateHost(T** p, size_t bytes) {
  return hipHostMalloc(p, bytes);
}
inline constexpr auto releaseHost = hipHostFree;
inline constexpr auto createEventFlags = hipEventCreateWithFlags;
inline constexpr auto disableTiming = hipEventDisableTiming;
inline constexpr auto copy = hipMemcpyAsync;
inline constexpr auto clear = hipMemsetAsync;
inline constexpr auto lastError = hipGetLastError;
inline constexpr auto hostToDevice = hipMemcpyHostToDevice;
inline constexpr auto deviceToHost = hipMemcpyDeviceToHost;
inline constexpr auto deviceToDevice = hipMemcpyDeviceToDevice;
inline void check(Status result, const char* operation) { hipCheck(result, operation); }

}
#define PRISM_GPU(call) ::prism::gpu::check((call), #call)
