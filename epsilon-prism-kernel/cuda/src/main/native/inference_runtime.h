#pragma once
#include "device_stream.h"

namespace prism::gpu {
using Stream = cudaStream_t;
using Event = cudaEvent_t;
using Status = cudaError_t;
inline constexpr auto outOfMemory = cudaErrorMemoryAllocation;
inline constexpr auto success = cudaSuccess;
inline constexpr auto memoryInfo = cudaMemGetInfo;
inline constexpr auto getDevice = cudaGetDevice;
inline constexpr auto setDevice = cudaSetDevice;
inline constexpr auto synchronize = cudaStreamSynchronize;
inline constexpr auto destroyEvent = cudaEventDestroy;
inline constexpr auto record = cudaEventRecord;
inline constexpr auto release = cudaFree;
template <class T>
inline Status allocate(T** p, size_t bytes) {
  return cudaMalloc(p, bytes);
}
template <class T>
inline Status allocateHost(T** p, size_t bytes) {
  return cudaMallocHost(p, bytes);
}
inline constexpr auto releaseHost = cudaFreeHost;
inline constexpr auto createEventFlags = cudaEventCreateWithFlags;
inline constexpr auto disableTiming = cudaEventDisableTiming;
inline Status copy(void* destination, const void* source, size_t bytes, cudaMemcpyKind direction, Stream stream) {
  return cudaMemcpyAsync(destination, source, bytes, direction, stream);
}
inline constexpr auto clear = cudaMemsetAsync;
inline constexpr auto lastError = cudaGetLastError;
inline constexpr auto hostToDevice = cudaMemcpyHostToDevice;
inline constexpr auto deviceToHost = cudaMemcpyDeviceToHost;
inline constexpr auto deviceToDevice = cudaMemcpyDeviceToDevice;
inline void check(Status result, const char* operation) { runtime::checkCuda(result, operation); }
}
#define PRISM_GPU(call) ::prism::gpu::check((call), #call)
