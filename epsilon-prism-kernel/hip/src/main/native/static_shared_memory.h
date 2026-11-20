#pragma once

#include <cstddef>

namespace prism::device {

// 一つのブロックで静的に確保できる共有メモリの上限。
inline constexpr std::size_t kStaticSharedBytes = 64 * 1024;

}
