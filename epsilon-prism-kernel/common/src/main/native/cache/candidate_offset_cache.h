#pragma once

#include "cache/candidate_dictionary.h"
#include "candidate_classification.h"

#include <array>

namespace prism {

// 頻出する候補の区分コードの組の辞書と、辞書の組ごとに基本表へ区分の補正を足しておいた表をデバイスに持つ。辞書のキーはモデルから
// 独立し、重みを公開するたびに足しておいた表を作り直す。
class CandidateOffsetCache {
 public:
  CandidateOffsetCache(const Architecture& architecture, int transitionCapacity, SegmentLayout layout,
                       HotCodeClassifier classifier);
  ~CandidateOffsetCache();

  CandidateOffsetCache(const CandidateOffsetCache&) = delete;
  CandidateOffsetCache& operator=(const CandidateOffsetCache&) = delete;

  // 計画のデバイスで、辞書と非同期転送の領域を確保する。確保できなければ、要るバイト数と空きを付けて例外を投げる。遷移の容量が辞書の
  // 標本に足りない形では何も確保せず、辞書を使わない。
  void reserve();
  bool available() const { return codes_ != nullptr; }
  bool active() const { return current_ != nullptr; }
  // 次の順伝播の入力を調べる。辞書を差し替えた場合に true を返すので、呼び出し側は合算表を作り直す。
  bool observe(const uint8_t* codes, int transitions, gpu::Stream stream);
  void publish(Span<const float> keys, Span<const float> values, SegmentWeights segments, gpu::Stream stream);
  HotCandidateOffsets prepare(const CandidateArgs& args, gpu::Stream stream);

 private:
  size_t tableSize() const { return static_cast<size_t>(baseCodes_ + kHotRows) * width_; }
  void release() noexcept;

  struct Dictionary {
    CandidateDictionary* upload = nullptr;
    HotEntry* entries = nullptr;
    uint64_t* tuples = nullptr;
    float* offsets = nullptr;
    gpu::Event copied = nullptr;
    int count = 0;
    int coverage = 0;
    bool uploadFailed = false;
  };

  int baseCodes_;
  int width_;
  int tiles_;
  int keys_;
  int transitionCapacity_;
  SegmentLayout layout_;
  HotCodeClassifier classifier_;
  DictionarySampling sampling_;
  std::array<Dictionary, 2> dictionaries_{};
  Dictionary* current_ = nullptr;
  Dictionary* pending_ = nullptr;
  uint16_t* codes_ = nullptr;
};

}  // namespace prism
