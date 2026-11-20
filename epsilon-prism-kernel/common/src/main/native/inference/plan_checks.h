#pragma once

#include <stdexcept>
#include <string>

namespace prism {

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

template <class E>
int ordinal(E value) {
  return static_cast<int>(value);
}

}  // namespace prism
