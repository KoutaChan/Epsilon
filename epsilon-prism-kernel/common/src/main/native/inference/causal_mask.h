#pragma once

namespace prism {

// 固定クエリは有効なキー、イベントは最初のイベントから自分までを読む。
__device__ __forceinline__ bool causalKey(bool present, int fixed, int query, int key) {
  return query < fixed ? present : key >= fixed && key <= query;
}

}  // namespace prism
