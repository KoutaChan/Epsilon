#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../inference/device_inline.h"
namespace prism::cache {
// カテゴリ列の行幅は 4 の倍数とは限らない。整列していないワードも memcpy で読む。
PRISM_DEVICE_INLINE uint32_t readWord(const uint8_t* p) {
  uint32_t word;
  memcpy(&word, p, sizeof(word));
  return word;
}

PRISM_DEVICE_INLINE uint32_t hashBytes(const uint8_t* p, int bytes, uint32_t hash) {
  int i = 0;
  for (; i + 4 <= bytes; i += 4) hash = (hash ^ readWord(p + i)) * 16777619u;
  for (; i < bytes; ++i) hash = (hash ^ p[i]) * 16777619u;
  return hash;
}

PRISM_DEVICE_INLINE bool sameBytes(const uint8_t* left, const uint8_t* right, int bytes) {
  int i = 0;
  for (; i + 4 <= bytes; i += 4) {
    if (readWord(left + i) != readWord(right + i)) return false;
  }
  for (; i < bytes; ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

struct RowBytes {
  const uint8_t* data;
  int width;
  PRISM_DEVICE_INLINE const uint8_t* row(int i) const { return width ? data + static_cast<size_t>(i) * width : data; }
};

struct InputKey {
  RowBytes columns[3];  // 埋め込みが参照するカテゴリ・スカラー・厳密値。省略された列の幅は 0。
};

// 空きの行を取った提出だけが値を書く。現在の提出で確保した行は元の入力を参照できるので、
// キーを書き終える前の別ウェーブとも照合できる。次の提出の読み出しはストリーム順で後になる。
template <int Kinds, class Cache, class Tags>
PRISM_DEVICE_INLINE int cachedInput(InputKey key, int kind, int row, uint32_t hash, Cache cache, bool& compute,
                                    Tags tags) {
  int width = 0;
  if (!cache.rows) return -1;
  for (const auto& col : key.columns) {
    if (col.width > cache.kKeyBytes - width) return -1;
    width += col.width;
  }
  const int slot = hash & (cache.rows - 1);
  const unsigned stamp = cache.generation * Kinds + kind;
  const unsigned long long claim = (static_cast<unsigned long long>(stamp) << 32) | static_cast<unsigned>(row);
  auto* tag = cache.tags + slot;
  auto prior = tags.load(tag);
  if (!prior) prior = tags.claim(tag, claim);
  auto* saved = cache.keys + size_t(slot) * cache.kKeyBytes;
  if (!prior) {
    for (const auto& col : key.columns) {
      for (int i = 0; i < col.width; ++i) saved[i] = col.row(row)[i];
      saved += col.width;
    }
    compute = true;
    return slot;
  }
  const unsigned when = static_cast<unsigned>(prior >> 32);
  if (when % Kinds != kind) return -1;
  for (const auto& col : key.columns) {
    const auto* other = when == stamp ? col.row(static_cast<int>(static_cast<uint32_t>(prior))) : saved;
    if (!sameBytes(col.row(row), other, col.width)) return -1;
    saved += col.width;
  }
  compute = false;
  return slot;
}

}  // namespace prism::cache
