#ifdef NDEBUG
#undef NDEBUG
#endif
#include "test_device.h"
#include "wave_tile.h"

#include <cassert>
#include <cstring>

namespace {
using namespace prism::cuda_tile;

__global__ void readWords(uint32_t* storage, bool valid, bool negative) {
  auto buffer = bufferResource(storage + 4, 4);
  if (negative) {
    storage[0] = bufferLoad4(buffer, 0, -4);
  } else if (!valid) {
    storage[0] = bufferLoad16(buffer, 16, 0)[0];
  } else {
    storage[0] = bufferLoad4(buffer, 16, -4) + bufferLoad16(buffer, 0, 0)[0];
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (!prism::test::hasDevice()) return 77;
  static_assert(PRISM_CHECKED);
  assert(argc == 2);
  const bool valid = !std::strcmp(argv[1], "valid");
  const bool negative = !std::strcmp(argv[1], "negative");
  assert(valid || negative || !std::strcmp(argv[1], "past-end"));
  uint32_t* storage;
  assert(cudaMalloc(&storage, 12 * sizeof(uint32_t)) == cudaSuccess);
  assert(cudaMemset(storage, 1, 12 * sizeof(uint32_t)) == cudaSuccess);
  readWords<<<1, 1>>>(storage, valid, negative);
  assert(cudaGetLastError() == cudaSuccess);
  const auto status = cudaDeviceSynchronize();
  if (valid) {
    assert(status == cudaSuccess);
    uint32_t value;
    assert(cudaMemcpy(&value, storage, sizeof(value), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(value == 0x02020202u);
    assert(cudaFree(storage) == cudaSuccess);
  } else {
    assert(status == cudaErrorIllegalInstruction || status == cudaErrorLaunchFailure);
  }
}
