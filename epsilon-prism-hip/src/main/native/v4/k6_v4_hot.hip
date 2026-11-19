#define PRISM_V4_CORE
#define PRISM_V4_HOT

#include "prism_hip_hot_cache.h"
#include "k6_candidate.hip"

namespace prism::v4 {
namespace {

__global__ void buildHotKernel(const uint64_t* tuples, int width, Span<const float> keys, Span<const float> values,
                               SegmentWeights segments, float* outKeys, float* outValues) {
  const bool value = threadIdx.x >= width;
  const int column = threadIdx.x - (value ? width : 0);
  const uint64_t tuple = tuples[blockIdx.x];
  const int base = tuple & 255;
  const auto table = value ? values : keys;
  float sum = table.data[base * width + column];
  // 元の K6 と同じく、基本表の値に区分を順に足す。区分だけの和を後から足すと丸めの位置が変わる。
  for (int i = 0; i < segments.layout.count; ++i) {
    const int digit = (tuple >> (8 * (i + 1))) & 255;
    const int row = segments.layout.spans[i].firstRow + digit;
    const int at = (row * 2 + value) * width + column;
    sum = __fadd_rn(sum, segments.offsets.data[at]);
  }
  (value ? outValues : outKeys)[blockIdx.x * width + column] = sum;
}

template <int Tiles>
__global__ void classifyHotKernel(Span<const uint8_t> codes, int transitions, int keys, SegmentLayout layout,
                                  const HotEntry* dictionary, uint16_t* result) {
  const int at = blockIdx.x * blockDim.x + threadIdx.x;
  const int transition = at / Tiles;
  if (transition >= transitions) return;
  const int tile = at % Tiles;
  const size_t row = static_cast<size_t>(transition) * layout.codeWidth;
  const int baseCode = codes.data[row + keys - Tiles + tile];
  uint64_t key = baseCode;
#pragma unroll
  for (int i = 0; i < kMaxSegmentSpans; ++i) {
    if (i >= layout.count) break;
    key |= static_cast<uint64_t>(codes.data[row + layout.spans[i].column + tile]) << (8 * (i + 1));
  }
  const uint32_t bucket = hotBucket(key);
  // 見つからなければ基本コードを残し、K6 が元の区分表を読む。
  int code = baseCode;
#pragma unroll
  for (int i = 0; i < kHotProbes; ++i) {
    const auto& entry = dictionary[(bucket + i) & (kHotBuckets - 1)];
    const uint64_t found = entry.key;
    if (found == key) {
      code = entry.code;
      break;
    }
    if (found == kEmptyHotKey) break;
  }
  result[at] = static_cast<uint16_t>(code);
}

}  // namespace

void buildHotOffsets(const uint64_t* tuples, int count, int width, Span<const float> keys, Span<const float> values,
                     SegmentWeights segments, float* outKeys, float* outValues, hipStream_t stream) {
  hipLaunchKernelGGL(buildHotKernel, dim3(count), dim3(2 * width), 0, stream, tuples, width, keys, values, segments,
                     outKeys, outValues);
  PRISM_HIP(hipGetLastError());
}

template <int Tiles>
void launchClassify(Span<const uint8_t> codes, int transitions, int keys, SegmentLayout layout,
                    const HotEntry* dictionary, uint16_t* result, hipStream_t stream) {
  constexpr int threads = 256;
  dim3 grid((transitions * Tiles + threads - 1) / threads);
  hipLaunchKernelGGL(classifyHotKernel<Tiles>, grid, dim3(threads), 0, stream, codes, transitions, keys, layout,
                     dictionary, result);
  PRISM_HIP(hipGetLastError());
}

HotCodeClassifier hotCodeClassifier(int tiles) {
  switch (tiles) {
#define PRISM_WIDTH(...)
#define PRISM_RULES(seats, tiles) \
  case tiles:                     \
    return launchClassify<tiles>;
#define PRISM_EAGER_WIDTH(name, width)
#include "prism_hip_capabilities.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
    default:
      return nullptr;
  }
}

}  // namespace prism::v4
