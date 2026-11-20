#include "cache/candidate_offset_cache.h"

#include "device_allocation.h"
#include "kernel_tuning.h"
#include "lane_resources.h"

#include <new>
#include <utility>

namespace prism {
namespace {

// 辞書の 1 行あたり 4 組の標本が取れる遷移の容量がなければ、辞書を使わない。
constexpr size_t kMinSamplePairs = 4 * kHotRows;
constexpr DictionaryTuning kTuning{KernelTuning::kDictionarySamplesPerRow * size_t{kHotRows},
                                   KernelTuning::kDictionaryStableInterval,
                                   KernelTuning::kDictionaryChangeInterval,
                                   KernelTuning::kDictionaryMinimumCoverage,
                                   KernelTuning::kDictionaryCoverageDrop,
                                   KernelTuning::kDictionaryMinimumGain};

}  // namespace

CandidateOffsetCache::CandidateOffsetCache(const Architecture& a, int transitionCapacity, SegmentLayout layout,
                                           HotCodeClassifier classifier)
    : baseCodes_(a.keyCodes),
      width_(a.attention()),
      tiles_(a.tiles),
      keys_(a.fixed()),
      transitionCapacity_(transitionCapacity),
      layout_(layout),
      classifier_(classifier) {}

void CandidateOffsetCache::reserve() {
  if (static_cast<size_t>(transitionCapacity_) * tiles_ < kMinSamplePairs) return;
  try {
    for (auto& bank : dictionaries_) {
      requireAllocated(gpu::allocateHost(&bank.upload, sizeof(CandidateDictionary)), sizeof(CandidateDictionary),
                       "the candidate dictionary upload");
      new (bank.upload) CandidateDictionary;
      requireAllocated(gpu::allocate(&bank.entries, kHotBuckets * sizeof(HotEntry)), kHotBuckets * sizeof(HotEntry),
                       "the candidate dictionary");
      requireAllocated(gpu::allocate(&bank.tuples, kHotRows * sizeof(uint64_t)), kHotRows * sizeof(uint64_t),
                       "the candidate dictionary tuples");
      requireAllocated(gpu::allocate(&bank.offsets, 2 * tableSize() * sizeof(float)), 2 * tableSize() * sizeof(float),
                       "the candidate dictionary offsets");
      PRISM_GPU(gpu::createEventFlags(&bank.copied, gpu::disableTiming));
    }
    const size_t codes = static_cast<size_t>(transitionCapacity_) * tiles_ * sizeof(uint16_t);
    requireAllocated(gpu::allocate(&codes_, codes), codes, "the candidate dictionary codes");
  } catch (...) {
    release();
    throw;
  }
}

bool CandidateOffsetCache::observe(const uint8_t* codes, int transitions, gpu::Stream stream) {
  const size_t pairs = static_cast<size_t>(transitions) * tiles_;
  if (codes_ == nullptr || pairs < kMinSamplePairs || !sampling_.due()) return false;
  auto& bank = dictionaries_[current_ == &dictionaries_[0] ? 1 : 0];
  if (bank.uploadFailed) return false;
  if (bank.count != 0 && !eventComplete(bank.copied)) return false;
  auto& upload = *bank.upload;
  const auto selected =
      sampling_.select(codes, pairs, keys_, tiles_, baseCodes_, kHotCodeTag, layout_, kTuning,
                       current_ ? current_->upload : nullptr, current_ ? current_->coverage : 0, upload);
  if (!selected.count) return false;

  // 同じストリームで表の作成と参照を続ける。転送中のホスト領域は完了通知まで書き換えない。
  bank.uploadFailed = true;
  PRISM_GPU(gpu::copy(bank.entries, upload.entries, sizeof(upload.entries), gpu::hostToDevice, stream));
  PRISM_GPU(gpu::copy(bank.tuples, upload.tuples, selected.count * sizeof(uint64_t), gpu::hostToDevice, stream));
  PRISM_GPU(gpu::record(bank.copied, stream));
  bank.uploadFailed = false;
  bank.count = selected.count;
  bank.coverage = selected.coverage;
  pending_ = &bank;
  return true;
}

CandidateOffsetCache::~CandidateOffsetCache() { release(); }

void CandidateOffsetCache::release() noexcept {
  if (codes_ != nullptr) (void)gpu::release(std::exchange(codes_, nullptr));
  for (auto& bank : dictionaries_) {
    if (bank.copied) (void)gpu::destroyEvent(std::exchange(bank.copied, nullptr));
    if (bank.offsets) (void)gpu::release(std::exchange(bank.offsets, nullptr));
    if (bank.tuples) (void)gpu::release(std::exchange(bank.tuples, nullptr));
    if (bank.entries) (void)gpu::release(std::exchange(bank.entries, nullptr));
    if (bank.upload) (void)gpu::releaseHost(std::exchange(bank.upload, nullptr));
  }
  current_ = pending_ = nullptr;
}

void CandidateOffsetCache::publish(Span<const float> keys, Span<const float> values, SegmentWeights segments,
                                   gpu::Stream stream) {
  const size_t first = static_cast<size_t>(baseCodes_) * width_;
  const size_t bytes = first * sizeof(float);
  auto* bank = pending_ ? std::exchange(pending_, nullptr) : current_;
  float* outValues = bank->offsets + tableSize();
  PRISM_GPU(gpu::copy(bank->offsets, keys.data, bytes, gpu::deviceToDevice, stream));
  PRISM_GPU(gpu::copy(outValues, values.data, bytes, gpu::deviceToDevice, stream));
  buildHotOffsets(bank->tuples, bank->count, width_, keys, values, segments, bank->offsets + first, outValues + first,
                  stream);
  current_ = bank;
}

HotCandidateOffsets CandidateOffsetCache::prepare(const CandidateArgs& args, gpu::Stream stream) {
  if (!active()) return {};
  classifier_(args.code, args.transitions, keys_, layout_, current_->entries, codes_, stream);
  const size_t used = static_cast<size_t>(baseCodes_ + current_->count) * width_;
  return {{current_->offsets, used},
          {current_->offsets + tableSize(), used},
          {codes_, static_cast<size_t>(args.transitions) * tiles_}};
}

}  // namespace prism
