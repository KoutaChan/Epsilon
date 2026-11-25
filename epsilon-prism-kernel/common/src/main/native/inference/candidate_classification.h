#pragma once

#include "cache/candidate_dictionary.h"
#include "inference_stages.h"

namespace prism {
void buildHotOffsets(const uint64_t* tuples, int count, int width, Span<const float> keys, Span<const float> values,
                     SegmentWeights segments, float* outKeys, float* outValues, gpu::Stream stream);
using HotCodeClassifier = void (*)(Span<const uint8_t>, int, int, SegmentLayout, const HotEntry*, uint16_t*,
                                   gpu::Stream);

// 実体化済みの分類器を引く。用意していない入力には nullptr を返す。
HotCodeClassifier hotCodeClassifier(int tiles);

}  // namespace prism
