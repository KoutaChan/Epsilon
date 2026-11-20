#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../inference/device_inline.h"

namespace prism::training {
PRISM_DEVICE_INLINE float countGate(float logit) { return 1.0f / (1.0f + expf(-logit)); }

PRISM_DEVICE_INLINE float gateGradient(float upstream, float tile, float gate) {
  return (upstream * tile) * (gate * (1.0f - gate));
}

PRISM_DEVICE_INLINE float scoreGradient(float probability, float upstream, float weighted) {
  return probability * (upstream - weighted);
}

// 表の同じ成分を読んだ遷移とキーを公開順にたどり、担当する範囲の勾配を FP32 で足す。
template <int Width, int Features>
PRISM_DEVICE_INLINE float tablePartial(int keys, int column, float scale, const float* query, const float* upstream,
                                       const float* score, const float* probs, const int32_t* occurrences, int first,
                                       int last) {
  constexpr int half = Width / 2, heads = half / Features;
  const bool keyGradient = column < half;
  const int feature = keyGradient ? column : column - half;
  const float* coefficients = (keyGradient ? score : probs) + feature / Features * keys;
  const float* vectors = (keyGradient ? query : upstream) + feature;
  float sum = 0.0f;
  for (int i = first; i < last; ++i) {
    const int pair = occurrences[i], transition = pair / keys, key = pair % keys;
    float coefficient = coefficients[static_cast<size_t>(transition) * heads * keys + key];
    if (keyGradient) coefficient *= scale;
    sum = fmaf(vectors[static_cast<size_t>(transition) * half], coefficient, sum);
  }
  return sum;
}
}  // namespace prism::training
