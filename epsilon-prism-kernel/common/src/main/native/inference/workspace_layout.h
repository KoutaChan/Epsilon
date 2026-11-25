#pragma once

#include "execution_plan.h"

namespace prism {

struct WorkspaceLayout {
  WorkspaceLayout(const PlanShape& shape, const ExecutionPlan& plan, bool folded, bool sharedProjection);
  std::array<size_t, kRegions> offset{};
  std::array<size_t, kRegions> capacity{};
  size_t projectionIndexOffset = 0;
  size_t bytes = 0;
};

struct EventPageRequirements {
  size_t eventInputBytes;
  size_t candidateCodeBytes;
  int projectionLimit;
  int projectionKeyBytes;
  bool folded;
};

struct EventPageLayout {
  EventPageLayout(const PlanShape& shape, EventPageRequirements requirements, int slots);
  static int fittingSlots(const PlanShape& shape, EventPageRequirements requirements, size_t budget);
  size_t map, features, embedding, fixedKeys, extraKeys, scratch, projections;
  size_t bytes = 0;
  int projectionRows = 0;
};

}  // namespace prism
