#pragma once

#include "recorded_commands.h"
#include <cstdlib>
#include <stdexcept>

#define __host__
#define __device__
#define __forceinline__ inline

namespace prism::gpu {
using Stream = void*;
using Event = int*;
enum Status { success, outOfMemory, failed };

Status getDevice(int* value);
Status setDevice(int value);
Status memoryInfo(size_t* available, size_t* total);

template <class T>
Status allocate(T** p, size_t bytes) {
  test::append(test::Command::kAllocate, nullptr, nullptr, bytes);
  *p = test::allocationSucceeds() ? static_cast<T*>(std::malloc(bytes)) : nullptr;
  return *p ? success : outOfMemory;
}

template <class T>
Status allocateHost(T** p, size_t bytes) {
  return allocate(p, bytes);
}

inline Status release(void* p) {
  std::free(p);
  return success;
}

inline Status releaseHost(void* p) { return release(p); }

enum Direction { hostToDevice, deviceToHost, deviceToDevice };
Status copy(void* destination, const void* source, size_t bytes, Direction direction, Stream stream);
Status clear(void*, int, size_t bytes, Stream stream);
Status synchronize(Stream stream);
inline constexpr unsigned disableTiming = 2;
Status createEventFlags(Event* event, unsigned flags);
Status destroyEvent(Event event);
Status record(Event event, Stream stream);
inline Status lastError() { return success; }
inline void check(Status status, const char* operation) {
  if (status != success) throw std::runtime_error(operation);
}
}
#define PRISM_GPU(call) ::prism::gpu::check((call), #call)
