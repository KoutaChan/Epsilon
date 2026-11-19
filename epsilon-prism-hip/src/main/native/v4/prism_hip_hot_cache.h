#pragma once

#include "prism_hip_v4.h"

#include <vector>

namespace prism::v4 {

inline constexpr int kHotRows = 512;
inline constexpr int kHotBuckets = 4096;
inline constexpr int kHotProbes = 4;
inline constexpr uint64_t kEmptyHotKey = ~uint64_t{0};

struct HotEntry {
  uint64_t key = kEmptyHotKey;
  uint16_t code = 0;
};

__host__ __device__ inline uint32_t hotBucket(uint64_t key) {
  const uint32_t lo = static_cast<uint32_t>(key);
  const uint32_t hi = static_cast<uint32_t>(key >> 32);
  uint32_t x = lo ^ (hi * 0x9e3779b9u);
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  return (x ^ (x >> 13)) & (kHotBuckets - 1);
}

void buildHotOffsets(const uint64_t* tuples, int count, int width, Span<const float> keys, Span<const float> values,
                     SegmentWeights segments, float* outKeys, float* outValues, hipStream_t stream);
using HotCodeClassifier = void (*)(Span<const uint8_t>, int, int, SegmentLayout, const HotEntry*, uint16_t*,
                                   hipStream_t);

// 実体化済みの分類器を引く。用意していない入力には nullptr を返す。
HotCodeClassifier hotCodeClassifier(int tiles);

// 辞書のキーはモデルから独立する。重みを公開するたび、辞書に沿った重み表を作り直す。
class HotOffsetCache {
 public:
  HotOffsetCache(const Architecture& architecture, int transitionCapacity, SegmentLayout layout,
                 HotCodeClassifier classifier);
  ~HotOffsetCache();

  HotOffsetCache(const HotOffsetCache&) = delete;
  HotOffsetCache& operator=(const HotOffsetCache&) = delete;

  bool active() const { return offsets_ != nullptr; }
  // 次の順伝播の入力を調べる。辞書を差し替えた場合に true を返すので、呼び出し側は合算表を作り直す。
  bool observe(const uint8_t* codes, int transitions, hipStream_t stream);
  void publish(Span<const float> keys, Span<const float> values, SegmentWeights segments, hipStream_t stream);
  HotCandidateOffsets prepare(const CandidateArgs& args, hipStream_t stream);

 private:
  size_t tableSize() const { return static_cast<size_t>(baseCodes_ + kHotRows) * width_; }
  void release() noexcept;

  int baseCodes_;
  int width_;
  int tiles_;
  int keys_;
  int transitionCapacity_;
  SegmentLayout layout_;
  HotCodeClassifier classifier_;
  std::vector<HotEntry> hostDictionary_;
  int tupleCount_ = 0;
  int coveragePercent_ = 0;
  int skipBatches_ = 0;
  HotEntry* dictionary_ = nullptr;
  uint64_t* deviceTuples_ = nullptr;
  float* offsets_ = nullptr;
  uint16_t* codes_ = nullptr;
};

}  // namespace prism::v4
