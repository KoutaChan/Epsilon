#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "inference/device_inline.h"

namespace prism {

inline constexpr int kHotRows = 512;
inline constexpr int kHotBuckets = 4096;
inline constexpr int kHotProbes = 4;
inline constexpr uint64_t kEmptyHotKey = ~uint64_t{0};

struct HotEntry {
  uint64_t key = kEmptyHotKey;
  uint16_t code = 0;
};

PRISM_HOST_DEVICE inline uint32_t hotBucket(uint64_t key) {
  const uint32_t lo = static_cast<uint32_t>(key);
  const uint32_t hi = static_cast<uint32_t>(key >> 32);
  uint32_t x = lo ^ (hi * 0x9e3779b9u);
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  return (x ^ (x >> 13)) & (kHotBuckets - 1);
}

struct CandidateDictionary {
  HotEntry entries[kHotBuckets];
  uint64_t tuples[kHotRows];
};

// 辞書の標本と入れ替えの決まり。
struct DictionaryTuning {
  size_t maximumSamples;  // 1 回に調べる組の数の上限
  int stableInterval;     // 辞書を保った後、次に調べるまでの提出の数
  int changeInterval;     // 辞書を作り直した後、次に調べるまでの提出の数
  int minimumCoverage;    // 辞書を使う命中率の下限（%）
  int coverageDrop;       // 前の辞書の命中率がこれだけ下がったら作り直す（ポイント）
  int minimumGain;        // 作り直しに求める命中率の改善（ポイント）
};

struct DictionarySelection {
  int count = 0;
  int coverage = 0;
};

template <class Layout>
inline uint64_t candidateKey(const uint8_t* row, int firstTile, int tile, int baseCodes, const Layout& layout) {
  uint64_t key = row[firstTile + tile];
  if (key >= static_cast<uint64_t>(baseCodes)) return kEmptyHotKey;
  for (int i = 0; i < layout.count; ++i) {
    const auto& span = layout.spans[i];
    const uint64_t code = row[span.column + tile];
    if (code >= static_cast<uint64_t>(span.rows)) return kEmptyHotKey;
    key |= code << (8 * (1 + i));
  }
  return key;
}

inline bool containsCandidateKey(const HotEntry* dictionary, uint64_t key) {
  if (dictionary == nullptr || key == kEmptyHotKey) return false;
  const auto bucket = hotBucket(key);
  for (int probe = 0; probe < kHotProbes; ++probe) {
    const auto found = dictionary[(bucket + probe) & (kHotBuckets - 1)].key;
    if (found == key) return true;
    if (found == kEmptyHotKey) break;
  }
  return false;
}

// 遷移と牌の組から区分コードを採り、同じ頻度ならキーの順で辞書に登録する。
class DictionarySampling {
 public:
  bool due() {
    if (skip_ == 0) return true;
    --skip_;
    return false;
  }

  template <class Layout>
  DictionarySelection select(const uint8_t* codes, size_t pairs, int keys, int tiles, int baseCodes, uint16_t tag,
                             const Layout& layout, const DictionaryTuning& tuning, const CandidateDictionary* previous,
                             int lastCoverage, CandidateDictionary& upload) {
    if (pairs == 0) return {};
    // 牌の種類数と互いに素な間隔を使い、一部の牌に偏らずに上限内で調べる。
    size_t stride = (pairs + tuning.maximumSamples - 1) / tuning.maximumSamples;
    while (std::gcd(stride, static_cast<size_t>(tiles)) != 1) ++stride;
    std::vector<uint64_t> sample;
    sample.reserve((pairs + stride - 1) / stride);
    size_t previousHits = 0;
    for (size_t i = 0; i < pairs; i += stride) {
      const uint8_t* row = codes + (i / tiles) * layout.codeWidth;
      const uint64_t key = candidateKey(row, keys - tiles, i % tiles, baseCodes, layout);
      sample.push_back(key);
      previousHits += containsCandidateKey(previous ? previous->entries : nullptr, key);
    }
    const int previousCoverage = static_cast<int>(100 * previousHits / sample.size());
    if (previous && previousCoverage >= std::max(tuning.minimumCoverage, lastCoverage - tuning.coverageDrop)) {
      skip_ = tuning.stableInterval - 1;
      return {};
    }

    skip_ = tuning.changeInterval - 1;
    std::unordered_map<uint64_t, int> counts;
    for (const auto key : sample) {
      if (key != kEmptyHotKey) ++counts[key];
    }
    std::vector<std::pair<uint64_t, int>> ranked(counts.begin(), counts.end());
    const size_t limit = std::min(ranked.size(), static_cast<size_t>(kHotRows));
    std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end(), [](const auto& a, const auto& b) {
      return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    std::fill_n(upload.entries, kHotBuckets, HotEntry{});
    int count = 0;
    size_t hits = 0;
    for (size_t n = 0; n < limit; ++n) {
      const auto [key, frequency] = ranked[n];
      const uint32_t bucket = hotBucket(key);
      for (int probe = 0; probe < kHotProbes; ++probe) {
        auto& entry = upload.entries[(bucket + probe) & (kHotBuckets - 1)];
        if (entry.key != kEmptyHotKey) continue;
        entry = {key, static_cast<uint16_t>(tag | (baseCodes + count))};
        upload.tuples[count++] = key;
        hits += frequency;
        break;
      }
    }
    const int coverage = static_cast<int>(100 * hits / sample.size());
    if (coverage < tuning.minimumCoverage || (previous && coverage < previousCoverage + tuning.minimumGain)) return {};

    return {count, coverage};
  }

 private:
  int skip_ = 0;
};

}  // namespace prism
