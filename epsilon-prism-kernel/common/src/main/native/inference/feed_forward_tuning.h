#pragma once
#include "collection_capabilities.h"
#include "kernel_tuning.h"

namespace prism {

template <int Width>
struct MlpTileFor;
template <>
struct MlpTileFor<kWidthFolded.width> {
  static constexpr MlpTuning tile = KernelTuning::kFoldedMlp;
};
template <>
struct MlpTileFor<kWidthProjected.width> {
  static constexpr MlpTuning tile = KernelTuning::kProjectedMlp;
};

// イベントページの予算から射影表に配分する行数の上限。
inline constexpr int kProjectionRowBudget = kProjectionRows;

}  // namespace prism
