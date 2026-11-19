#pragma once

namespace prism::v4 {

// Fixed queries use the regular key mask. The event prefix starts at the first
// event, since a fixed token may already contain information from later events.
__device__ __forceinline__ bool causalKey(bool present, int fixed, int query, int key) {
  return query < fixed ? present : key >= fixed && key <= query;
}

}  // namespace prism::v4
