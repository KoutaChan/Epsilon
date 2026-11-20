#include "cache/event_page_layout.h"
#include "cache/projection_key.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
void pageCapacity() {
  for (const size_t fixed : {31, 39}) {
    prism::cache::PageBudget spec{2, 19, 7 * fixed, 17, 64, 48, 2, 32, 256, 64, false};
    for (const bool folded : {false, true}) {
      spec.folded = folded;
      spec.projectionLimit = folded ? 0 : 64;
      assert(spec.layout(0).bytes == 0 && spec.layout(0).rows == 0);
      assert(spec.fit(spec.layout(1).bytes - 1) == 0);
      for (int n = 19; n <= 129; ++n) {
        const auto lay = spec.layout(n);
        assert(lay.indices == 0 && lay.raw % 256 == 0 && lay.projections % 256 == 0 && lay.bytes % 256 == 0);
        assert(lay.bytes >= lay.projections && lay.projections >= lay.scratch);
        assert(lay.raw >=
               sizeof(int32_t) * (2 + 2 * spec.batchEvents) + spec.batchEvents * spec.inputBytes + spec.basicCodes);
        assert(lay.fixedKeys - lay.input >= (folded ? n * 3 * spec.attention * sizeof(uint16_t) : 0));
        assert(lay.extraKeys - lay.fixedKeys >= n * 2 * spec.attention * sizeof(uint16_t));
        assert(lay.scratch - lay.extraKeys >= n * spec.extraLayers * 2 * spec.attention * sizeof(uint16_t));
        assert(lay.bytes - lay.projections >=
               lay.rows * (sizeof(uint64_t) + spec.keyBytes + sizeof(uint16_t) * (spec.hidden + 3 * spec.attention)));
        int fit = spec.fit(lay.bytes);
        assert(fit >= n && spec.layout(fit).bytes <= lay.bytes && spec.layout(fit + 1).bytes > lay.bytes);
        assert(spec.fit(lay.bytes - 1) < n);
        if (n == 19) assert(spec.fit(lay.bytes - 1) > 0);
        if (n == 63 || n == 64 || n == 129) assert(lay.rows == (folded ? 0 : n == 63 ? 32 : 64));
      }
      spec.basicCodes = std::numeric_limits<size_t>::max();
      bool overflow = false;
      try {
        (void)spec.layout(1);
      } catch (const std::overflow_error&) {
        overflow = true;
      }
      assert(overflow && spec.fit(1024) == 0);
      spec.basicCodes = 7 * fixed;
    }
  }
}

struct Tags {
  unsigned long long load(unsigned long long* tag) const { return *tag; }
  unsigned long long claim(unsigned long long* tag, unsigned long long value) const {
    const auto prior = *tag;
    if (!prior) *tag = value;
    return prior;
  }
};

struct Cache {
  unsigned long long* tags;
  uint8_t* keys;
  unsigned generation;
  int rows;
  static constexpr int kKeyBytes = 32;
};

void projectionIdentity() {
  static constexpr int rows = 64, kinds = 5;
  std::array<unsigned long long, rows> tags{};
  std::array<uint8_t, rows * Cache::kKeyBytes> saved{};
  Cache cache{tags.data(), saved.data(), 1, rows};
  std::array<uint8_t, 6> category{0x80, 0, 3, 0x80, 0, 3};
  std::array<uint8_t, 4> scalar{0, 0, 0, 0};
  std::array<uint8_t, 8> exact{1, 2, 3, 4, 1, 2, 3, 4};
  prism::cache::InputKey key{{{category.data(), 3}, {scalar.data(), 2}, {exact.data(), 4}}};
  auto lookup = [&](int kind, int row, int hash, bool compute) {
    bool actual = true;
    const int slot = prism::cache::cachedInput<kinds>(key, kind, row, hash, cache, actual, Tags{});
    assert(compute == actual);
    return slot;
  };
  assert(lookup(3, 0, 61, true) == 61 && lookup(3, 1, 61, false) == 61);
  scalar[3] = 0x80;
  assert(lookup(3, 1, 61, true) == -1 && lookup(3, 1, 29, true) == 29);
  scalar[3] = 0;
  ++cache.generation;
  // 上書きした元行とは独立に、保存した全列で照合する。
  category[0] = 4;
  assert(lookup(3, 1, 61, false) == 61 && lookup(2, 1, 61, true) == -1);
  assert(lookup(3, 0, 61, true) == -1);
  exact[7] ^= 1;
  assert(lookup(3, 1, 61, true) == -1);
  exact[7] ^= 1;
  scalar[3] = 0x80;
  assert(lookup(3, 1, 29, false) == 29 && lookup(3, 1, 61, true) == -1);
  key.columns[0].width = Cache::kKeyBytes + 1;
  assert(lookup(3, 1, 61, true) == -1);
  key.columns[0].width = 3;
  tags.fill(0);
  assert(lookup(3, 1, 29, true) == 29);
  cache.rows = 0;
  assert(lookup(3, 1, 29, true) == -1);
}
}

int main() {
  pageCapacity();
  projectionIdentity();
}
