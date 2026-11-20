#include "test_device.h"
#include "device_stream.h"
#include "wave_tile.h"

#include <array>
#include <stdexcept>

namespace {
constexpr int side = prism::cuda_tile::kMatrixTile;
constexpr int elements = side * side;
struct Matrices {
  uint16_t a[elements], b[elements];
  float c[elements];
};
__global__ void multiply(Matrices* matrices) {
  int lane = threadIdx.x, col = prism::cuda_tile::fragmentColumnOfLane(lane),
      parity = prism::cuda_tile::fragmentParityOfLane(lane);
  auto a = prism::cuda_tile::fragment(matrices->a + side * col);
  auto b = prism::cuda_tile::fragment(matrices->b + side * col);
  prism::cuda_tile::f32x8 c;
  for (int i = 0; i < 8; ++i) c[i] = matrices->c[side * prism::cuda_tile::accumulatorRow(i, parity) + col];
  c = prism::cuda_tile::mma(a, b, c);
  for (int i = 0; i < 8; ++i) matrices->c[side * prism::cuda_tile::accumulatorRow(i, parity) + col] = c[i];
}
void check(int device) {
  prism::runtime::DeviceScope selected(device);
  for (cudaStream_t invalid : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread}) {
    try {
      prism::runtime::DeviceStream lane(device, invalid);
    } catch (const std::invalid_argument&) {
      continue;
    }
    throw std::runtime_error("Accepted a default CUDA stream");
  }
  cudaStream_t stream;
  prism::runtime::checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  prism::runtime::DeviceStream lane(device, stream);
  Matrices *host, *gpu;
  prism::runtime::checkCuda(cudaMallocHost(&host, sizeof(*host)), "allocate host matrices");
  prism::runtime::checkCuda(cudaMalloc(&gpu, sizeof(*gpu)), "allocate device matrices");
  auto inputA = [](int i, int k) { return ((i + 3) * (k + 5) % 19) - 9; };
  auto inputB = [](int j, int k) { return ((j + 5) * (k + 1) % 23) - 11; };
  for (int i = 0; i < elements; ++i) {
    host->a[i] = cuda::std::bit_cast<uint32_t>(static_cast<float>(inputA(i / side, i % side))) >> 16;
    host->b[i] = cuda::std::bit_cast<uint32_t>(static_cast<float>(inputB(i / side, i % side))) >> 16;
    host->c[i] = static_cast<float>(i) / 2;
  }
  lane.upload(gpu, host, sizeof(*host));
  multiply<<<1, prism::cuda_tile::kWave, 0, stream>>>(gpu);
  prism::runtime::checkCuda(cudaGetLastError(), "launch matrix tile");
  lane.download(host, gpu, sizeof(*host));
  lane.synchronize();
  for (int i = 0; i < elements; ++i) {
    float expected = static_cast<float>(i) / 2;
    for (int k = 0; k < side; ++k) expected += inputA(i / side, k) * inputB(i % side, k);
    if (host->c[i] != expected) throw std::runtime_error("Wrong matrix tile element " + std::to_string(i));
  }
  prism::runtime::checkCuda(cudaFree(gpu), "free matrix");
  prism::runtime::checkCuda(cudaFreeHost(host), "free host matrix");
  prism::runtime::checkCuda(cudaStreamDestroy(stream), "destroy stream");
}
}
int main(int argc, char** argv) {
  if (!prism::test::hasDevice()) return 77;
  if (argc < 2) throw std::invalid_argument("Expected one or more CUDA device ordinals");
  for (int i = 1; i < argc; ++i) check(std::stoi(argv[i]));
}
