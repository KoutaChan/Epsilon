#include "test_device.h"
#include "inference_kernels.h"
#include "event_submission.h"
#include "relation_encoding.h"
#include "device_stream.h"
#include <cmath>
#include <stdexcept>

namespace {
constexpr int blocks = 4, words = blocks * blocks * 256, heads = 4;
struct Input {
  uint8_t attributes[64 * 9], tiles[34 * 34];
  int columns[2], mask;
  uint32_t codes[words];
  float rho[112 * heads], bias[2 * prism::kBiasEntries * 2];
};
void check(int device, int seats, int tiles) {
  prism::runtime::DeviceScope scope(device);
  cudaStream_t stream;
  PRISM_GPU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  prism::runtime::DeviceStream lane(device, stream);
  Input *host, *gpu;
  PRISM_GPU(cudaMallocHost(&host, sizeof(Input)));
  PRISM_GPU(cudaMalloc(&gpu, sizeof(Input)));
  *host = {};
  int fixed = 1 + seats + tiles;
  int tokens = 51;
  host->columns[1] = tokens - fixed;
  for (int i = 0; i < 112 * heads; ++i) host->rho[i] = float((37 * i) % 53 - 26) / 16;
  for (auto& word : host->codes) word = 0xffffffffu;
  lane.upload(gpu, host, sizeof(Input));
  prism::RelationArgs args{{gpu->attributes, sizeof(host->attributes)},
                           {gpu->columns, 2},
                           {gpu->tiles, sizeof(host->tiles)},
                           {gpu->codes, words},
                           {1, seats, tiles},
                           2,
                           0,
                           1,
                           blocks,
                           {9, 0, 1, 2, 3, 4, 5, 6, seats - 1}};
  auto type = [&](int p) { return p == 0 ? 0 : p <= seats ? 1 : p < fixed ? 2 : 3; };
  for (bool paged : {true, false}) {
    if (paged)
      prism::launchPageRelation(args, &gpu->mask, stream);
    else
      prism::launchRelation(args, stream);
    prism::launchRelationBias({{gpu->rho, 112 * heads}, {gpu->bias, heads * prism::kBiasEntries}, heads}, heads / 2,
                              stream);
    lane.download(host, gpu, sizeof(Input));
    lane.synchronize();
    for (int q = 0; q < 64; ++q)
      for (int k = 0; k < 64; ++k) {
        int segment[] = {4 * type(q) + type(k),
                         0,
                         0,
                         0,
                         (type(k) == 3) + (type(q) == 3 && type(k) == 1 ? 9 : 0),
                         type(k) == 3 ? (type(q) == 3 ? 5 : 10) : 0,
                         0};
        uint32_t expected = paged && (q / 16) * 16 >= fixed ? 0xffffffffu
                            : k >= tokens                   ? prism::kPaddedKeyCode
                            : q >= tokens                   ? 0u
                                                            : prism::attentionCode(segment);
        int offset = ((q / 16) * blocks + k / 16) * 256 + (q % 16 + 16 * (k % 2)) * 8 + (k % 16) / 2;
        if (host->codes[offset] != expected)
          throw std::runtime_error("Relation word at query " + std::to_string(q) + " key " + std::to_string(k));
      }
    for (int h = 0; h < heads; ++h)
      for (int i = 0; i < prism::kBiasEntries; ++i) {
        int entry = i % 256, base = i / 256 * 32;
        float value = base < 96     ? host->rho[(base + ((entry ^ (entry >> 4)) & 15)) * heads + h] +
                                          host->rho[(base + 16 + entry / 16) * heads + h]
                      : entry == 15 ? -INFINITY
                                    : host->rho[(96 + entry) * heads + h];
        if (host->bias[(h / 2 * prism::kBiasEntries + i) * 2 + h % 2] != value * prism::kLog2e)
          throw std::runtime_error("Relation bias mismatch");
      }
  }
  PRISM_GPU(cudaFree(gpu));
  PRISM_GPU(cudaFreeHost(host));
  PRISM_GPU(cudaStreamDestroy(stream));
}
}
int main(int argc, char** argv) {
  if (!prism::test::hasDevice()) return 77;
  if (argc < 2) throw std::invalid_argument("Expected CUDA device ordinals");
  for (int i = 1; i < argc; ++i) {
    check(std::stoi(argv[i]), 3, 27);
    check(std::stoi(argv[i]), 4, 34);
  }
}
