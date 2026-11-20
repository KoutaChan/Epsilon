// スロットごとの作業領域と重みを管理し、計画の順にバッチを提出する。
#pragma once

#include "cache/event_page_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "inference_device.h"
#include "execution_plan.h"
#include "workspace_layout.h"
#include "packed_weights.h"
#include "inference_kernels.h"
#include "event_submission.h"
#include "candidate_segments.h"

namespace prism {

class CandidateOffsetCache;

struct WidthKernels;

// 開いたレーンの実効の容量。
struct LaneCapacity {
  int64_t freeBefore = 0;       // イベントのページを確保する前の空き（バイト）
  int64_t freeAfter = 0;        // 確保した後の空き（バイト）
  int64_t total = 0;            // デバイスの総量（バイト）
  int64_t pageSlots = 0;        // イベントのページの数（0 はページを使わない）
  int64_t projectionRows = 0;   // 層 0 の射影表の行数
  int64_t stagingBytes = 0;     // 転送と作業の領域（バイト）
  int64_t pageBytes = 0;        // イベントのページ（バイト）
  int64_t projectionBytes = 0;  // 射影表（バイト）
  bool dictionary = false;      // 頻出辞書を使うか
  bool splitCandidate = false;  // 候補の Attention を前段と後段に分けるか
};

// 一つのスロットが、公開された重みと専用の作業領域を使って計画を実行する。
class Plan {
 public:
  // スロットを作った呼び出し元が、計画の破棄までストリームを保持する。
  Plan(int device, gpu::Stream stream, const PlanShape& shape);
  ~Plan();

  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  int device() const { return device_; }
  const PlanShape& shape() const { return shape_; }
  const ExecutionPlan& executionPlan() const { return execution_; }
  // 提出に使える重みの世代。重みが無いか、公開の途中で失敗した後は -1。
  int64_t publishedVersion() const { return packed_.version; }

  // 領域の容量（バイト）。
  size_t regionBytes(Region region) const;

  // WeightPack（ヘッダーつき）を読んで照合する（ヘッダーの形の記述が計画のものとビットで同じこと、BF16・FP32 の部分に NaN と ±∞ がないこと）。
  // そのあとデバイスへ写し、バイアスの表と、QKV（層 1 から）・MLP の重みをフラグメントの順に並べた写しを作り直して、バージョンを覚える。
  // 呼ぶのは公開のときだけで、ストリームを同期する。重みを替える手段はこれだけ。途中で失敗したらバージョンを忘れ、次に写し終えるまで
  // forward は例外を投げる。
  void uploadWeights(const void* host, size_t bytes, int64_t version);

  // 入力全体か公開イベントの差分を転送して、形の記述の順にカーネルを流し、スコアを読み出す（同期しない）。Attention に渡す行の組は、ホスト側で入力の行インデックスから求める。
  void forward(const void* input, const BatchShape& batch, void* scores, int64_t version, const int* eventMap = nullptr,
               int misses = 0, int slots = 0, size_t transferBytes = 0);

  // 全レーンの作業領域を確保したあと、空き VRAM を残りのレーンで分ける。
  void configureCandidateInput(const void* descriptor, size_t count);
  // イベントのページを確保して、レーンの実効の容量を返す。events が −1 ならページを確保しない。0 なら空きのうち reserveFraction を
  // 残して remaining 個のレーンで分けた予算に収まる数を確保し、1 バッチのイベントの数に足りなければ確保しない。正なら events 個を
  // 確保し、確保できなければ要るバイト数と空きを付けて例外を投げる。
  LaneCapacity configureEventPages(int remaining, int events, double reserveFraction);

 private:
  CandidateInputLayout candidateInput_{};
  bool restoreCandidates_ = false;

  using Submit = void (Plan::*)(const PlanStep&, const BatchShape&);
  static Submit submitFunction(Kernel kernel);
  std::vector<Submit> submits_;
  void submitEmbed(const PlanStep&, const BatchShape& batch);
  void submitRelation(const PlanStep&, const BatchShape& batch);
  void submitAttention(const PlanStep&, const BatchShape& batch);
  void submitMlpAll(const PlanStep&, const BatchShape& batch);
  void submitMlpFixed(const PlanStep&, const BatchShape& batch);
  void submitMlpLast(const PlanStep&, const BatchShape& batch);
  void submitRereadKeyValue(const PlanStep&, const BatchShape& batch);
  void submitCandidate(const PlanStep&, const BatchShape& batch);
  void submitReread(const PlanStep&, const BatchShape& batch);
  void submitValueGraph(const PlanStep&, const BatchShape& batch);
  // ホストの行のインデックス [行, 幅]（int32、アラインメントは問わない）から、Attention で同じタイル数の組に入る行を数える。起動の前に、
  // 行のイベントの範囲がバッチの中にあることと、行の遷移が行の順に隙間なく並び、総数がバッチの遷移の数と同じことを確かめる。
  void countGroupRows(const char* rowColumn, const BatchShape& batch);
  void validate(const BatchShape& batch) const;
  void chooseHotOffsets(const void* input, const BatchShape& batch);
  void publishHotOffsets(CandidateOffsetCache& cache);

  // デバイスのメモリを解放する（今のデバイスが device_ であること）。
  void release() noexcept;

  template <class T>
  Span<T> regionSpan(Region region, size_t elementOffset) const;
  template <class T>
  Span<const T> weight(Weight weight, int entry) const;
  template <class T>
  Span<const T> section(const BatchShape& batch, Section section) const;
  template <class T>
  Span<T> relationBias(int layer) const;
  template <class T>
  Span<T> qkvFragments(int layer) const;
  Span<const uint16_t> mlpFragments(Weight weight, int entry) const;
  MlpWeights mlpWeights(int layer, TokenGroup group) const;
  MlpWeights candidateWeights() const;
  DiscardValueWeights discardWeights() const;
  // 局・席・牌の層の FFN の区間を segment に入れ、区間の数を返す（型を分けなければ局・席・牌の 1 つ、型ごとなら局・席と牌の 2 つ）。
  // last なら最後の正規化のゲインも入れる。
  int segments(int layer, const BatchShape& batch, bool last, MlpSegment (&segment)[2]) const;
  // 最後の全トークンの層の前段 FFNが作る、イベントのキーとバリューの組（局・席・牌の層ごとの分と、2 段目があれば 2 段目のイベントの記憶の分）
  // を args に入れる。
  void eventKeyValues(const PlanStep& step, const BatchShape& batch, MlpArgs& args) const;
  TokenLayout layout(const BatchShape& batch) const;
  size_t weightBytes(Weight weight, int entry) const;

  gpu::Status allocateEventPages(int slots);
  size_t eventInputWidth() const;
  EventPageRequirements pageRequirements() const;
  size_t basicCodeBytes(int transitions) const;
  void uploadPageMap(const BatchShape& batch, const int* map, int misses, int slots, size_t transferBytes);
  void uploadInputs(const void* input, const BatchShape& batch, bool sparse);
  // 埋め込みと同時に射影する幅では QKV も保存する。広い形では全トークンの層の射影を使う。
  bool saveEventProjection() const;
  void gatherEventFeatures(const BatchShape& compact, EmbedArgs& args);
  void exchangePages(const BatchShape& batch, bool embedding);
  void compactEventInputs(const BatchShape& batch);

  char* pageStorage_ = nullptr;
  int* pageMap_ = nullptr;
  uint8_t* pageFeatures_ = nullptr;
  char* pageInput_ = nullptr;
  char* pageFixedKeys_ = nullptr;
  char* pageExtraKeys_ = nullptr;
  uint16_t* compactAttention_ = nullptr;
  ProjectionCache projectionCache_{};
  int pageSlots_ = 0;
  int denseEvents_ = 0;
  bool paged_ = false;
  bool pagesUsable_ = true;

  int device_;
  gpu::Stream stream_;
  const PlanShape shape_;
  const WidthKernels& kernels_;
  const ExecutionPlan execution_;
  const WorkspaceLayout layout_;
  PackedWeights packed_;
  void launchRereadKeys(const BatchShape& batch);
  uint16_t* countWorkspace_ = nullptr;
  uint16_t* candidateAttention_ = nullptr;
  std::unique_ptr<CandidateOffsetCache> candidateOffsets_;
  char* workspace_ = nullptr;
  int32_t* rowOrder_ =
      nullptr;  // 全トークンの層と固定クエリの層が読む、組の順に並べた行の番号（関係のコードの起動で rowOrderKernel がバッチごとに書く。RowOrderArgs）
  int groupRows_[kAttentionGroups] = {};  // 今のバッチの組ごとの行の数（forward がホストで数える）
  size_t sharedBytes_ = 0;
  bool splitShortPagedHeads_ = false;
  size_t embedShared_ =
      0;  // 埋め込みの動的な共有メモリ（uploadWeights がヘッダーの特徴の数から決める。embedSharedBytes）
};

}  // namespace prism
