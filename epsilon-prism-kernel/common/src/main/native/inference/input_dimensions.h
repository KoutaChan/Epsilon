#pragma once

#include "execution_plan.h"
#include "../model_schema.h"

namespace prism {

constexpr size_t kVectorAlignment = 16;  // 埋め込み表と入力をベクトルで読むときのバイト境界。

using schema::Axis;

inline Axis sectionAxis(Section section) { return schema::sections[static_cast<int>(section)].axis; }

constexpr int sectionElementBytes(Section section) {
  switch (schema::sections[static_cast<int>(section)].element) {
    case schema::Element::U8:
      return 1;
    case schema::Element::BF16:
      return 2;
    case schema::Element::I32:
    case schema::Element::FP32:
      return 4;
  }
  throw std::invalid_argument("Unknown section element");
}

inline Axis slotAxis(EmbeddingSlot slot) {
  switch (slot) {
    case EmbeddingSlot::kRound:
      return Axis::ROW;
    case EmbeddingSlot::kSeat:
      return Axis::SEAT;
    case EmbeddingSlot::kTile:
      return Axis::TILE;
    case EmbeddingSlot::kEvent:
      return Axis::EVENT;
    case EmbeddingSlot::kTransition:
      return Axis::TRANSITION;
    case EmbeddingSlot::kCount:
      break;
  }
  throw std::invalid_argument("unknown embedding slot");
}

inline size_t axisCount(Axis axis, const BatchShape& batch, int seats, int tiles) {
  switch (axis) {
    case Axis::ROW:
      return batch.rows;
    case Axis::SEAT:
      return static_cast<size_t>(seats) * batch.rows;
    case Axis::TILE:
      return static_cast<size_t>(tiles) * batch.rows;
    case Axis::EVENT:
      return batch.events;
    case Axis::TOKEN:
      return static_cast<size_t>(1 + seats + tiles) * batch.rows + batch.events;
    case Axis::TRANSITION:
      return batch.transitions;
    case Axis::ACTION:
      return batch.actions;
  }
  return 0;
}

}  // namespace prism
