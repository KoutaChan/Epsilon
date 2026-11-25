#pragma once

#include "collection_capabilities.h"

#include <cstdint>

namespace prism {

struct SegmentSpan {
  int32_t column = 0;
  int32_t firstRow = 0;
  int32_t rows = 0;
};

// 推論と学習で使う区分表の配置。
struct SegmentLayout {
  int32_t codeWidth = 0;
  int32_t tableRows = 0;
  int32_t count = 0;
  SegmentSpan spans[kMaxSegmentSpans]{};

  bool valid(int fixedKeys, int tiles) const {
    if (count < 1 || count > kMaxSegmentSpans) return false;
    int column = fixedKeys;
    int row = 0;
    for (int i = 0; i < count; ++i) {
      const auto& span = spans[i];
      if (span.column != column || span.firstRow != row || span.rows < 1 || span.rows > kU8Codes) return false;
      column += tiles;
      row += span.rows;
    }
    return codeWidth == column && tableRows == row;
  }
};

constexpr int kSegmentLayoutInts = 3 + 3 * kMaxSegmentSpans;
static_assert(sizeof(SegmentLayout) == kSegmentLayoutInts * sizeof(int32_t));

}  // namespace prism
