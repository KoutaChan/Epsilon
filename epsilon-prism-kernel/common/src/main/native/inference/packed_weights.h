#pragma once

#include "execution_plan.h"
#include <string>

namespace prism {

// MLP の重みをフラグメントの順に並べた写し 1 つ: 重みのスロットの項目にある行列の、行 [firstRow, firstRow + rows) の [rows, columns]。
struct FragmentCopy {
  Weight weight;
  int entry;
  int firstRow;
  int rows;
  int columns;
  size_t offset;  // 写しの領域の中での BF16 の位置
};

struct PackedWeights {
  explicit PackedWeights(const PlanShape& shape);
  PackHeader inspect(const PlanShape& shape, const CandidatePackLayout& candidate, bool folded, const void* host,
                     size_t bytes, int64_t version) const;

  std::vector<FragmentCopy> fragments;
  std::array<int, kWeightEntries> index{};
  size_t fragmentElements = 0;
  PackHeader header{};
  int64_t version = -1;
  size_t bytes = 0;
  char* data = nullptr;
  float* relationBias = nullptr;
  uint16_t* qkv = nullptr;
  uint16_t* mlp = nullptr;
};

bool entryHeld(const Architecture& architecture, Weight weight, int entry);
size_t weightElements(Weight weight, int entry, const PackHeader& header);
std::string weightName(Weight weight, int entry);

}  // namespace prism
