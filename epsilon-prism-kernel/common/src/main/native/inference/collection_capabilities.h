#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace prism {

// JNI の両側で使う上限とタイル幅。
#define PRISM_LIMIT(name, value) inline constexpr int k##name = value;
#include "collection_limits.def"
#undef PRISM_LIMIT

// 列は、幅、層 0 の折り込み、基本候補層、読み直し、ΣC と K を持つ候補層の順。
struct WidthCapability {
  int width;
  bool folded;
  bool standard;
  bool reread;
  bool core;
};

#define PRISM_WIDTH(name, width, ...) inline constexpr WidthCapability kWidth##name{width, __VA_ARGS__};
#define PRISM_RULES(seats, tiles)
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH

inline constexpr auto kWidthCapabilities = std::array{
#define PRISM_WIDTH(name, ...) kWidth##name,
#define PRISM_RULES(seats, tiles)
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
};

struct RuleCapability {
  int seats;
  int tiles;
  constexpr int fixed() const { return 1 + seats + tiles; }
};

inline constexpr auto kRuleCapabilities = std::array{
#define PRISM_WIDTH(...)
#define PRISM_RULES(seats, tiles) RuleCapability{seats, tiles},
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
};

inline constexpr auto kMaxCandidateKeys = [] {
  int keys = 0;
  for (const auto rule : kRuleCapabilities) {
    if (rule.fixed() > keys) keys = rule.fixed();
  }
  return keys;
}();

template <class Launch>
bool dispatchCandidateKeys(int keys, Launch launch) {
  switch (keys) {
#define PRISM_WIDTH(...)
#define PRISM_RULES(seats, tiles)                                                \
  case RuleCapability{seats, tiles}.fixed():                                     \
    launch(std::integral_constant<int, RuleCapability{seats, tiles}.fixed()>{}); \
    return true;
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
    default:
      return false;
  }
}

template <class Launch>
void forCandidateKeys(int keys, Launch launch) {
  if (!dispatchCandidateKeys(keys, launch)) throw std::invalid_argument("unsupported number of candidate keys");
}

inline constexpr const WidthCapability* widthCapability(int hidden) {
  for (const auto& capability : kWidthCapabilities) {
    if (capability.width == hidden) return &capability;
  }
  return nullptr;
}

template <class Launch>
void forCoreWidth(int hidden, Launch launch) {
  switch (hidden) {
#define PRISM_WIDTH(name, width, folded, standard, reread, core, ...) \
  case width:                                                         \
    if constexpr (core) {                                             \
      launch(std::integral_constant<int, width>{});                   \
      return;                                                         \
    }                                                                 \
    break;
#define PRISM_RULES(seats, tiles)
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
    default:
      break;
  }
  throw std::invalid_argument("no candidate kernel for this width");
}

inline constexpr int kU8Codes = std::numeric_limits<uint8_t>::max() + 1;

}  // namespace prism
