#pragma once

#include <cstdint>

namespace prism {

constexpr int kRelationSegments = 7;
constexpr int kRelationEntries = 16;
constexpr int kMaxAttributes = 12;  // 関係の属性の列の上限
constexpr int kCodeBlockWords = 256;
constexpr int kRelationOpponents = 3;

constexpr int kTileRelationFirstHalf = kRelationEntries - 1;

struct RelationColumns {
  int width;
  int tile;
  int owner;
  int event;
  int riichi;
  int lastDiscard;
  int link;
  int safety;
  int safeties;
};

}  // namespace prism
