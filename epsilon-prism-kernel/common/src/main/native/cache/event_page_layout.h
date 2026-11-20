#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace prism::cache {
// 各領域の先頭をバイトで表す。rows は射影表の行数。
struct PageLayout {
  size_t indices{}, raw{}, input{}, fixedKeys{}, extraKeys{}, scratch{}, projections{}, bytes{};
  int rows{};
};

// 全レーンの必須領域を確保した後、このレーンが使う配置と最大のページ数を求める。
struct PageBudget {
  size_t batchRows, batchEvents, basicCodes, inputBytes, hidden, attention, extraLayers;
  size_t keyBytes, alignment;
  int projectionLimit;
  bool folded;

  static size_t add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("event cache size exceeds address range");
    return a + b;
  }
  static size_t multiply(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
      throw std::overflow_error("event cache size exceeds address range");
    return a * b;
  }

  PageLayout layout(int slots) const {
    PageLayout out{};
    if (slots < 0 || !alignment || (alignment & (alignment - 1)))
      throw std::invalid_argument("invalid event cache dimensions");
    if (!slots) return out;
    const size_t pages = static_cast<size_t>(slots), batch = std::min(pages, batchEvents);
    auto reserve = [&](size_t bytes) {
      size_t offset = out.bytes;
      out.bytes = add(offset, add(bytes, alignment - 1) & ~(alignment - 1));
      return offset;
    };
    const size_t h = multiply(hidden, sizeof(uint16_t)), a = multiply(attention, sizeof(uint16_t));
    const size_t qkv = multiply(3, a);
    out.indices = reserve(add(
        add(multiply(add(batchRows, multiply(2, batch)), sizeof(int32_t)), multiply(batch, inputBytes)), basicCodes));
    out.raw = reserve(multiply(pages, inputBytes));
    out.input = reserve(folded ? multiply(pages, qkv) : 0);
    const size_t kv = multiply(pages, multiply(2, a));
    out.fixedKeys = reserve(kv);
    out.extraKeys = reserve(multiply(kv, extraLayers));
    out.scratch = reserve(multiply(batch, folded ? a : std::max(h, a)));
    int limit = std::min(slots, projectionLimit);
    if (limit > 0) {
      out.rows = 1;
      while (out.rows <= limit / 2) out.rows *= 2;
    }
    out.projections = reserve(multiply(out.rows, add(add(sizeof(uint64_t), keyBytes), add(h, qkv))));
    return out;
  }

  int fit(size_t budget) const {
    int lo = 0, hi = std::numeric_limits<int>::max();
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2 + 1;
      try {
        if (layout(mid).bytes <= budget) {
          lo = mid;
          continue;
        }
      } catch (const std::overflow_error&) {
      }
      hi = mid - 1;
    }
    return lo;
  }
};
}  // namespace prism::cache
