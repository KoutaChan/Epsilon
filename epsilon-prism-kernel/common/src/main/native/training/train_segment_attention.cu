#include "training/training_launch.h"
#include "training_device.h"

#include <stdexcept>

#include "training/segment_attention.cuh"

namespace prism {
namespace {

template <int Width>
void launchTablePartials(SegmentAttention d, SegmentAttentionArrays a, SegmentAttentionBackward b, int tableRows,
                         gpu::Stream stream) {
  const dim3 grid(tableRows, kShards);
  const auto launch = [&](auto keys) {
    segmentAttentionPartials<Width, decltype(keys)::value>
        <<<grid, dim3(kThreads), 0, stream>>>(d, a, b.probs, b.score, b.upstream, b.offsets, b.occurrences, b.partial);
  };
  if (!dispatchCandidateKeys(d.input.keys, launch)) launch(std::integral_constant<int, 0>{});
}

void check(SegmentAttention d) {
  const auto s = d.input;
  const auto* capability = widthCapability(s.width);
  if (s.transitions <= 0 || s.keys <= 0 || s.keys > kMaxCandidateKeys || s.tiles < 1 || s.seats < 2 || s.seats > 4 ||
      s.keys != 1 + s.tiles + s.seats || s.baseCodes <= 0 || s.baseCodes > kU8Codes ||
      !s.layout.valid(s.keys, s.tiles) || (capability == nullptr || !capability->core) ||
      int64_t(d.heads) * 2 * kFeatures != s.width || d.rows <= 0 || !(d.scale > 0.0f && std::isfinite(d.scale)) ||
      int64_t(d.rows) * s.keys > INT32_MAX || int64_t(s.transitions) * d.heads + kWaves - 1 > INT32_MAX ||
      int64_t(s.transitions) * s.layout.codeWidth > INT32_MAX) {
    throw std::invalid_argument("invalid fused candidate attention shape");
  }
}

}  // namespace

void launchSegmentAttention(SegmentAttention d, SegmentAttentionArrays a, float* result, float* probs,
                            gpu::Stream stream) {
  check(d);
  const int blocks = (d.input.transitions * d.heads + kWaves - 1) / kWaves;
  segmentAttentionForward<<<dim3(blocks), dim3(kThreads), 0, stream>>>(d, a, result, probs);
  PRISM_GPU(gpu::lastError());
}

void launchSegmentAttentionBackward(SegmentAttention d, SegmentAttentionArrays a, SegmentAttentionBackward b,
                                    gpu::Stream stream) {
  check(d);
  const int queries = (d.input.transitions * d.heads + kWaves - 1) / kWaves;
  const int tableRows = d.input.baseCodes + d.input.layout.tableRows;
  const int pairs = (d.input.transitions * d.input.keys + kThreads - 1) / kThreads;
  segmentAttentionQueryGradient<<<dim3(queries), dim3(kThreads), 0, stream>>>(d, a, b.probs, b.upstream, b.query,
                                                                              b.score);
  segmentAttentionMemoryGradient<<<dim3(d.rows * d.input.keys), dim3(kThreads), 0, stream>>>(d, a, b.probs, b.score,
                                                                                             b.upstream, b.memory);
  PRISM_GPU(gpu::clear(b.cursor, 0, tableRows * sizeof(int32_t), stream));
  const size_t histogramBytes = tableRows * sizeof(int32_t);
  segmentAttentionOccurrences<false>
      <<<dim3(pairs), dim3(kThreads), histogramBytes, stream>>>(d, a.codes, b.cursor, nullptr, nullptr, tableRows);
  segmentAttentionOffsets<<<dim3(1), dim3(1), 0, stream>>>(tableRows, b.cursor, b.offsets);
  segmentAttentionOccurrences<true><<<dim3(pairs), dim3(kThreads), 2 * histogramBytes, stream>>>(
      d, a.codes, b.cursor, b.offsets, b.occurrences, tableRows);
  forCoreWidth(d.input.width,
               [&](auto width) { launchTablePartials<decltype(width)::value>(d, a, b, tableRows, stream); });
  segmentAttentionTables<<<dim3(tableRows), dim3(kThreads), 0, stream>>>(d, b.partial, b.base, b.segments);
  PRISM_GPU(gpu::lastError());
}

}  // namespace prism
