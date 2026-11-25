#pragma once

#include <cstdint>

namespace prism {
// 牌ごとの射影と上流勾配は BF16、ゲートとその勾配は FP32。行とコードを逆伝播でも参照する。
struct CountTraining {
  int rows, transitions, tiles, codes, width;
};

// 区分は宣言順に足す。順伝播は確率を保存し、逆伝播はクエリ・記憶・基本表・区分表の FP32 勾配を書く。
struct SegmentTraining {
  int transitions, keys, tiles, seats, baseCodes, width;
  SegmentLayout layout;
};

struct SegmentAttention {
  SegmentTraining input;
  int rows, heads;
  float scale;
};

struct SegmentAttentionArrays {
  const float *query, *memory, *base, *segments;
  const uint8_t* codes;
  const int32_t* groups;
};

struct SegmentAttentionBackward {
  const float *probs, *upstream;
  float *query, *memory, *base, *segments, *score, *partial;
  int32_t *cursor, *offsets, *occurrences;
};

}  // namespace prism
