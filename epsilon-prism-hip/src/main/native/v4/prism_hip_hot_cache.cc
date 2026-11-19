#include "prism_hip_hot_cache.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace prism::v4 {
namespace {

// 512 枠に対して少なくとも 4 標本/枠を確保し、短い呼び出しのために辞書を確保しない。
constexpr size_t kMinSamplePairs = 4 * kHotRows;
constexpr size_t kMaxSamplePairs = 8 * kHotRows;
// 更新直後は分布の移り変わりを早めに拾い、安定している間はホストの走査を減らす。
constexpr int kReviewAfterChange = 8;
constexpr int kReviewWhileStable = 32;
// 命中率が低い辞書は追加の読み込みに見合わない。測定の揺れで辞書を入れ替えないよう、更新には 4 ポイント以上の改善を求める。
constexpr int kMinimumCoverage = 75;
constexpr int kCoverageDrop = 5;
constexpr int kMinimumGain = 4;

uint64_t readKey(const uint8_t* row, int firstTile, int tile, int baseCodes, const SegmentLayout& layout) {
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

bool contains(const std::vector<HotEntry>& dictionary, uint64_t key) {
  if (dictionary.empty() || key == kEmptyHotKey) return false;
  const auto bucket = hotBucket(key);
  for (int probe = 0; probe < kHotProbes; ++probe) {
    const auto found = dictionary[(bucket + probe) & (kHotBuckets - 1)].key;
    if (found == key) return true;
    if (found == kEmptyHotKey) break;
  }
  return false;
}

}  // namespace

HotOffsetCache::HotOffsetCache(const Architecture& a, int transitionCapacity, SegmentLayout layout,
                               HotCodeClassifier classifier)
    : baseCodes_(a.keyCodes),
      width_(a.attention()),
      tiles_(a.tiles),
      keys_(a.fixed()),
      transitionCapacity_(transitionCapacity),
      layout_(layout),
      classifier_(classifier) {}

bool HotOffsetCache::observe(const uint8_t* codes, int transitions, hipStream_t stream) {
  const size_t pairs = static_cast<size_t>(transitions) * tiles_;
  if (pairs < kMinSamplePairs) return false;
  if (skipBatches_ > 0) {
    --skipBatches_;
    return false;
  }
  // 牌の種類数と互いに素な間隔を使い、一部の牌に偏らずに上限内で調べる。
  size_t stride = (pairs + kMaxSamplePairs - 1) / kMaxSamplePairs;
  while (std::gcd(stride, static_cast<size_t>(tiles_)) != 1) ++stride;
  std::vector<uint64_t> sample;
  sample.reserve((pairs + stride - 1) / stride);
  size_t previousHits = 0;
  for (size_t i = 0; i < pairs; i += stride) {
    const uint8_t* row = codes + (i / tiles_) * layout_.codeWidth;
    const uint64_t key = readKey(row, keys_ - tiles_, i % tiles_, baseCodes_, layout_);
    sample.push_back(key);
    previousHits += contains(hostDictionary_, key);
  }
  const size_t previousCoverage = 100 * previousHits / sample.size();
  if (active() &&
      previousCoverage >= static_cast<size_t>(std::max(kMinimumCoverage, coveragePercent_ - kCoverageDrop))) {
    skipBatches_ = kReviewWhileStable - 1;
    return false;
  }

  skipBatches_ = kReviewAfterChange - 1;
  std::unordered_map<uint64_t, int> counts;
  for (const auto key : sample) {
    if (key != kEmptyHotKey) ++counts[key];
  }
  std::vector<std::pair<uint64_t, int>> ranked(counts.begin(), counts.end());
  const size_t limit = std::min(ranked.size(), static_cast<size_t>(kHotRows));
  std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::vector<HotEntry> dictionary(kHotBuckets);
  std::vector<uint64_t> tuples;
  tuples.reserve(limit);
  size_t hits = 0;
  for (size_t n = 0; n < limit; ++n) {
    const auto [key, count] = ranked[n];
    const uint32_t bucket = hotBucket(key);
    for (int probe = 0; probe < kHotProbes; ++probe) {
      auto& entry = dictionary[(bucket + probe) & (kHotBuckets - 1)];
      if (entry.key != kEmptyHotKey) continue;
      entry = {key, static_cast<uint16_t>(kHotCodeTag | (baseCodes_ + tuples.size()))};
      tuples.push_back(key);
      hits += count;
      break;
    }
  }
  const size_t coverage = 100 * hits / sample.size();
  if (coverage < kMinimumCoverage || (active() && coverage < previousCoverage + kMinimumGain)) return false;

  try {
    if (!active()) {
      PRISM_HIP(hipMalloc(&dictionary_, dictionary.size() * sizeof(HotEntry)));
      PRISM_HIP(hipMalloc(&deviceTuples_, kHotRows * sizeof(uint64_t)));
      PRISM_HIP(hipMalloc(&offsets_, 2 * tableSize() * sizeof(float)));
      PRISM_HIP(hipMalloc(&codes_, static_cast<size_t>(transitionCapacity_) * tiles_ * sizeof(uint16_t)));
    }
    PRISM_HIP(hipMemcpyAsync(dictionary_, dictionary.data(), dictionary.size() * sizeof(HotEntry),
                             hipMemcpyHostToDevice, stream));
    PRISM_HIP(
        hipMemcpyAsync(deviceTuples_, tuples.data(), tuples.size() * sizeof(uint64_t), hipMemcpyHostToDevice, stream));
    // ホストの辞書を破棄する前に転送を終える。
    PRISM_HIP(hipStreamSynchronize(stream));
  } catch (...) {
    release();
    throw;
  }
  hostDictionary_ = std::move(dictionary);
  tupleCount_ = static_cast<int>(tuples.size());
  coveragePercent_ = static_cast<int>(coverage);
  return true;
}

HotOffsetCache::~HotOffsetCache() { release(); }

void HotOffsetCache::release() noexcept {
  if (codes_ != nullptr) (void)hipFree(std::exchange(codes_, nullptr));
  if (offsets_ != nullptr) (void)hipFree(std::exchange(offsets_, nullptr));
  if (deviceTuples_ != nullptr) (void)hipFree(std::exchange(deviceTuples_, nullptr));
  if (dictionary_ != nullptr) (void)hipFree(std::exchange(dictionary_, nullptr));
  hostDictionary_.clear();
}

void HotOffsetCache::publish(Span<const float> keys, Span<const float> values, SegmentWeights segments,
                             hipStream_t stream) {
  const size_t first = static_cast<size_t>(baseCodes_) * width_;
  const size_t bytes = first * sizeof(float);
  float* outValues = offsets_ + tableSize();
  PRISM_HIP(hipMemcpyAsync(offsets_, keys.data, bytes, hipMemcpyDeviceToDevice, stream));
  PRISM_HIP(hipMemcpyAsync(outValues, values.data, bytes, hipMemcpyDeviceToDevice, stream));
  buildHotOffsets(deviceTuples_, tupleCount_, width_, keys, values, segments, offsets_ + first, outValues + first,
                  stream);
}

HotCandidateOffsets HotOffsetCache::prepare(const CandidateArgs& args, hipStream_t stream) {
  if (!active()) return {};
  classifier_(args.code, args.transitions, keys_, layout_, dictionary_, codes_, stream);
  const size_t used = static_cast<size_t>(baseCodes_ + tupleCount_) * width_;
  return {{offsets_, used}, {offsets_ + tableSize(), used}, {codes_, static_cast<size_t>(args.transitions) * tiles_}};
}

}  // namespace prism::v4
