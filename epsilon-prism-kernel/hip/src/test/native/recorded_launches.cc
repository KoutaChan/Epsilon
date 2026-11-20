#include "recorded_commands.h"
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace {
using prism::test::append;
using prism::test::Command;
dim3 grid, block;
size_t shared;
hipStream_t stream;
}

extern "C" {
void** __hipRegisterFatBinary(const void*) {
  static void* image;
  return &image;
}
void __hipUnregisterFatBinary(void**) {}
void __hipRegisterFunction(void**, const void*, const char*, const char*, unsigned, void*, void*, void*, void*, void*) {
}
hipError_t hipGetDevice(int* value) {
  *value = prism::test::selectedDevice();
  return hipSuccess;
}
hipError_t hipSetDevice(int value) {
  assert(value >= 0);
  prism::test::selectDevice(value);
  return hipSuccess;
}
// 記録用のストリームは、それを使う計画が選んだデバイスのものとして答える。
hipError_t hipStreamGetDevice(hipStream_t, hipDevice_t* value) {
  *value = prism::test::selectedDevice();
  return hipSuccess;
}
hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int) {
  *p = {};
  std::strcpy(p->gcnArchName, PRISM_HIP_ARCH);
  p->warpSize = 32;
  p->sharedMemPerBlock = 65536;
  return hipSuccess;
}
hipError_t hipMalloc(void** p, size_t n) {
  append(Command::kAllocate, nullptr, nullptr, n);
  *p = std::malloc(n);
  return *p ? hipSuccess : hipErrorOutOfMemory;
}
hipError_t hipFree(void* p) {
  std::free(p);
  return hipSuccess;
}
hipError_t hipHostMalloc(void** p, size_t n, unsigned) { return hipMalloc(p, n); }
hipError_t hipHostFree(void* p) { return hipFree(p); }
hipError_t hipMemcpyAsync(void* destination, const void* source, size_t bytes, hipMemcpyKind kind, hipStream_t s) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      append(Command::kUpload, s, source, bytes);
      break;
    case hipMemcpyDeviceToHost:
      append(Command::kDownload, s, destination, bytes);
      break;
    case hipMemcpyDeviceToDevice:
      append(Command::kDeviceCopy, s, nullptr, bytes);
      break;
    default:
      assert(false);
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}
hipError_t hipMemsetAsync(void*, int, size_t bytes, hipStream_t s) {
  append(Command::kClear, s, nullptr, bytes);
  return hipSuccess;
}
hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  *free = *total = 1ULL << 32;
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t s) { return prism::test::completeStream(s) ? hipSuccess : hipErrorUnknown; }
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  append(Command::kAllocate);
  static int identity;
  *event = reinterpret_cast<hipEvent_t>(&identity);
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t s) {
  append(Command::kEvent, s);
  return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t) { return prism::test::eventComplete() ? hipSuccess : hipErrorNotReady; }
hipError_t hipGetLastError() { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "Recorded runtime error"; }
hipError_t __hipPushCallConfiguration(dim3 g, dim3 b, size_t n, hipStream_t s) {
  grid = g;
  block = b;
  shared = n;
  stream = s;
  return hipSuccess;
}
hipError_t __hipPopCallConfiguration(dim3* g, dim3* b, size_t* n, hipStream_t* s) {
  *g = grid;
  *b = block;
  *n = shared;
  *s = stream;
  return hipSuccess;
}
hipError_t hipLaunchKernel(const void*, dim3 g, dim3 b, void**, size_t, hipStream_t s) {
  assert(g.x && g.y && g.z && b.x && s);
  append(Command::kLaunch, s);
  return hipSuccess;
}
}
