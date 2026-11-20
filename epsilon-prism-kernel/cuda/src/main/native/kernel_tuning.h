#pragma once

#include "inference/launch_tiles.h"

namespace prism {
struct KernelTuning : DefaultKernelTuning {
  static constexpr int kTokensPerBlock = 16;
  static constexpr bool kCanSplitShortPagedHeads = true;

  // 短い履歴のヘッドを 2 つのブロックに分けるのは、計測した compute capability 12.0 の GPU だけ。
  template <class Lane>
  static bool splitShortPagedHeads(const Lane& lane) {
    return lane.computeCapability() == 120;
  }
};
}  // namespace prism
