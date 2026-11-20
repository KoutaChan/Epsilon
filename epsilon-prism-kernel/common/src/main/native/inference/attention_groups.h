#pragma once
#include "device_inline.h"

#include "collection_capabilities.h"

namespace prism {

// 各行を 16 トークンずつタイルに分ける。クエリの範囲にかかわらず、行はタイル数でまとめる。
// 組 g は attentionWaves(g) タイルまでを受け持つ。実行時には固定トークンのヘッドを分担するウェーブを追加できる。
constexpr int kAttentionGroups = 4;

PRISM_HOST_DEVICE constexpr int attentionWaves(int group) {
  return group == 0 ? 4 : group == 1 ? 6 : group == 2 ? 8 : 10;
}

// 行の組を返す。T = min(F + max(イベント数, 0), kMaxRowTokens) は、Attention で有効なトークン数と同じ式。
PRISM_HOST_DEVICE constexpr int attentionGroup(int fixed, int events) {
  const int tokens = fixed + (events > 0 ? events : 0);
  const int blocks = ((tokens < kMaxRowTokens ? tokens : kMaxRowTokens) + 15) / 16;
  return blocks <= attentionWaves(0) ? 0 : blocks <= attentionWaves(1) ? 1 : blocks <= attentionWaves(2) ? 2 : 3;
}

static_assert(16 * attentionWaves(kAttentionGroups - 1) >= kMaxRowTokens, "the last row group holds the longest row");

}  // namespace prism
