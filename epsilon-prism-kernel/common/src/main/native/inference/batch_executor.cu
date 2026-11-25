#include "input_dimensions.h"
#include "plan_checks.h"
#include "batch_executor.h"
#include "kernel_selection.h"
#include "feed_forward_tuning.h"
#include "lane_resources.h"
#include "cache/candidate_offset_cache.h"
#include "device_allocation.h"
#include "event_submission.h"
#include "inference_stages.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "collection_capabilities.h"
#include "inference_kernels.h"
#include "relation_encoding.h"

namespace prism {

namespace {

constexpr size_t kSlack =
    kPackAlignment;  // 領域ごとの余白（境界検査つきのビルドが、外れた読み書きを領域の先頭へ向けても外に出ない）
constexpr size_t kBufferBytes =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());  // device::Buffer で読む領域の上限

// 1 層の QKV をフラグメントの順に並べた写しの、BF16 の数（W_qkv を並べ替えただけなので、同じ 3A·H）。
size_t qkvFragmentElements(const PlanShape& shape) {
  return static_cast<size_t>(3) * shape.attention() * shape.hidden();
}

// 計画のデバイスに確保する。足りなければ、要るバイト数と空きを付けて例外を投げる。
template <class T>
void allocateRequired(T** address, size_t bytes, const char* what) {
  requireAllocated(gpu::allocate(address, bytes), bytes, what);
}

}  // namespace

Plan::Plan(int device, gpu::Stream stream, const PlanShape& shape)
    : device_(device),
      stream_(stream),
      shape_(checkedPlanShape(shape)),
      kernels_(widthKernels(shape_.hidden())),
      execution_(shape_, kernels_.capability),
      layout_(shape_, execution_, kernels_.capability.folded, kernels_.sharedProjection != nullptr),
      packed_(shape_) {
  for (const auto& step : execution_.steps) submits_.push_back(submitFunction(step.kernel));
  const Architecture& a = shape_.architecture;
  // 前段 FFN・最終 FFN・候補の Attention ・再読射影・イベント再読は、MLP の重みの写しを device::Buffer の 32 bit のバイトオフセットで読む。
  const size_t mlpFragmentBytes = packed_.fragmentElements * sizeof(uint16_t);
  require(mlpFragmentBytes <= kBufferBytes, "the MLP weight copies of " + std::to_string(mlpFragmentBytes) +
                                                " bytes exceed the 2 GiB that the kernels address with buffer offsets");

  gpu::SelectedDevice scope(device_);
  const gpu::Lane lane(device_, stream_);
  sharedBytes_ = lane.sharedBytes();
  splitShortPagedHeads_ = KernelTuning::splitShortPagedHeads(lane);
  try {
    allocateRequired(&workspace_, layout_.bytes, "the workspace");
    allocateRequired(&packed_.data, shape.weightBytes + kSlack, "the weight pack");
    if (execution_.candidate.countWorkspace != 0) {
      allocateRequired(&countWorkspace_, execution_.candidate.countWorkspace * sizeof(uint16_t),
                       "the candidate count workspace");
    }
    allocateRequired(&packed_.relationBias, a.depth * relationBiasFloats(a.headPairs()) * sizeof(float) + kSlack,
                     "the relation bias tables");
    allocateRequired(&packed_.qkv, a.depth * qkvFragmentElements(shape) * sizeof(uint16_t) + kSlack,
                     "the QKV weight fragments");
    allocateRequired(&packed_.mlp, mlpFragmentBytes + kSlack, "the MLP weight fragments");
    allocateRequired(&rowOrder_, static_cast<size_t>(shape.rows) * sizeof(int32_t) + kSlack, "the row order");
    if (shape_.options.has(PlanOptions::kSegments)) {
      if (auto classifier = hotCodeClassifier(a.tiles)) {
        candidateOffsets_ =
            std::make_unique<CandidateOffsetCache>(a, shape_.transitions, shape_.options.segments, classifier);
        candidateOffsets_->reserve();
        if (candidateOffsets_->available() && kernels_.splitCandidate) {
          allocateRequired(&candidateAttention_, size_t(shape_.transitions) * a.attention() * sizeof(uint16_t),
                           "the split candidate attention");
        }
      }
    }
  } catch (...) {
    release();
    throw;
  }
}

Plan::~Plan() {
  int previous = -1;
  const bool switched =
      gpu::getDevice(&previous) == gpu::success && previous != device_ && gpu::setDevice(device_) == gpu::success;
  release();
  if (switched) {
    (void)gpu::setDevice(previous);
  }
}

void Plan::release() noexcept {
  // 解放の前に、この計画が積んだ処理が終わるのを待つ（作業領域を使うカーネルが残らないように）。
  if (workspace_ != nullptr || packed_.data != nullptr || countWorkspace_ != nullptr ||
      packed_.relationBias != nullptr || packed_.qkv != nullptr || packed_.mlp != nullptr || rowOrder_ != nullptr) {
    (void)gpu::synchronize(stream_);
  }
  candidateOffsets_.reset();
  if (pageStorage_ != nullptr) {
    (void)gpu::release(pageStorage_);
    pageStorage_ = nullptr;
  }
  if (rowOrder_ != nullptr) {
    (void)gpu::release(rowOrder_);
    rowOrder_ = nullptr;
  }
  if (packed_.mlp != nullptr) {
    (void)gpu::release(packed_.mlp);
    packed_.mlp = nullptr;
  }
  if (packed_.qkv != nullptr) {
    (void)gpu::release(packed_.qkv);
    packed_.qkv = nullptr;
  }
  if (packed_.relationBias != nullptr) {
    (void)gpu::release(packed_.relationBias);
    packed_.relationBias = nullptr;
  }
  if (packed_.data != nullptr) {
    (void)gpu::release(packed_.data);
    packed_.data = nullptr;
  }
  if (countWorkspace_ != nullptr) {
    (void)gpu::release(countWorkspace_);
    countWorkspace_ = nullptr;
  }
  if (candidateAttention_ != nullptr) {
    (void)gpu::release(candidateAttention_);
    candidateAttention_ = nullptr;
  }
  if (workspace_ != nullptr) {
    (void)gpu::release(workspace_);
    workspace_ = nullptr;
  }
}

size_t Plan::regionBytes(Region region) const {
  require(ordinal(region) >= 0 && ordinal(region) < kRegions, "unknown workspace region");
  return layout_.capacity[ordinal(region)];
}

void Plan::uploadWeights(const void* host, size_t bytes, int64_t version) {
  PackHeader header = packed_.inspect(shape_, execution_.candidate, kernels_.capability.folded, host, bytes, version);
  const size_t embedShared = kernels_.embedBytes ? kernels_.embedBytes(shape_.hidden(), header.embedding) : 0;
  require(embedShared + sizeof(EmbeddingSpec) <= sharedBytes_,
          "embedding needs " + std::to_string(embedShared) + " bytes of shared memory for the dense weights");
  if (pageStorage_) {
    require(std::equal(std::begin(header.sectionWidth), std::end(header.sectionWidth),
                       std::begin(packed_.header.sectionWidth)),
            "published weights changed the event page layout");
  }
  gpu::SelectedDevice scope(device_);
  // 写しと、重みから作るものが済むまでは、重みのない計画として扱う（途中で失敗しても、新しい重みと古い表・写しの組み合わせで
  // forward が回らない）。
  packed_.version = -1;
  if (projectionCache_.rows)
    PRISM_GPU(
        gpu::clear(projectionCache_.tags, 0, size_t(projectionCache_.rows) * sizeof(*projectionCache_.tags), stream_));
  packed_.header = header;
  packed_.bytes = bytes;
  embedShared_ = embedShared;
  PRISM_GPU(gpu::copy(packed_.data, host, bytes, gpu::hostToDevice, stream_));
  // バイアスの表と、QKV・MLP の重みをフラグメントの順に並べた写しは重みだけで決まる。重みを替える手段はここだけなので、公開のたびに
  // 作り直す（古いものは残らない）。pack にない層・重みの分は作らず、カーネルが引くときに止める（relationBias・qkvFragments・
  // mlpFragments）。層 0 の W_qkv は埋め込みの表に折り込んであり（全トークンをクエリにする層 0 は qkv0 を読む）、Attention 用の重みを別に複製する必要はない。
  const Architecture& a = shape_.architecture;
  for (int l = 0; l < a.depth; ++l) {
    if (packed_.header.offset(Weight::kRelationBias, l) >= 0) {
      const RelationBiasArgs args{weight<float>(Weight::kRelationBias, l), relationBias<float>(l), a.heads()};
      launchRelationBias(args, a.headPairs(), stream_);
    }
    if ((l > 0 || !kernels_.capability.folded) && packed_.header.offset(Weight::kQkv, l) >= 0) {
      const QkvFragmentArgs args{weight<uint16_t>(Weight::kQkv, l), qkvFragments<uint16_t>(l)};
      kernels_.qkv(a.hidden, args, a.headPairs(), stream_);
    }
  }
  for (const FragmentCopy& copy : packed_.fragments) {
    if (packed_.header.offset(copy.weight, copy.entry) < 0) {
      continue;
    }
    Span<const uint16_t> source = weight<uint16_t>(copy.weight, copy.entry);
    const size_t elements = static_cast<size_t>(copy.rows) * copy.columns;
    const size_t first = static_cast<size_t>(copy.firstRow) * copy.columns;
    require(first + elements <= source.limit, weightName(copy.weight, copy.entry) + " is smaller than its copy");
    const MlpFragmentArgs args{Span<const uint16_t>{source.data + first, elements},
                               Span<uint16_t>{packed_.mlp + copy.offset, elements}, copy.rows, copy.columns};
    launchMlpFragments(args, stream_);
  }
  if (candidateOffsets_ && candidateOffsets_->active()) publishHotOffsets(*candidateOffsets_);
  PRISM_GPU(gpu::synchronize(stream_));
  packed_.version = version;
  pagesUsable_ = true;
}

void Plan::chooseHotOffsets(const void* input, const BatchShape& batch) {
  if (!candidateOffsets_) return;
  const auto* codes = static_cast<const uint8_t*>(input) + batch.section[ordinal(Section::kTransitionCode)];
  if (candidateOffsets_->observe(codes, batch.transitions, stream_)) publishHotOffsets(*candidateOffsets_);
}

void Plan::publishHotOffsets(CandidateOffsetCache& cache) {
  const auto& options = shape_.options;
  const auto* segments = reinterpret_cast<const float*>(packed_.data + options.segmentsOffset);
  cache.publish(weight<float>(Weight::kCodeKey, 0), weight<float>(Weight::kCodeValue, 0),
                {{segments, execution_.candidate.segments}, options.segments}, stream_);
}

void Plan::validate(const BatchShape& batch) const {
  require(packed_.version >= 0, "weights have not been uploaded to the plan");
  require(batch.rows > 0 && batch.rows <= shape_.rows && batch.events >= 0 && batch.events <= shape_.events &&
              batch.transitions >= batch.rows && batch.transitions <= shape_.transitions &&
              batch.actions >= batch.rows && batch.actions <= shape_.actions,
          "batch counts exceed the plan capacity");
  require(batch.bytes > 0 && batch.bytes <= shape_.inputBytes, "batch blob exceeds the input capacity");
  const int seats = shape_.architecture.seats;
  const int tiles = shape_.architecture.tiles;
  for (int s = 0; s < kSections; ++s) {
    const Section section = static_cast<Section>(s);
    const size_t extent = axisCount(sectionAxis(section), batch, seats, tiles) * packed_.header.sectionWidth[s] *
                          sectionElementBytes(section);
    if (batch.section[s] % kVectorAlignment != 0 || batch.section[s] > batch.bytes ||
        extent > batch.bytes - batch.section[s]) {
      throw std::invalid_argument("input section " + std::to_string(s) + " at " + std::to_string(batch.section[s]) +
                                  " (+" + std::to_string(extent) + ") is outside the blob of " +
                                  std::to_string(batch.bytes) + " bytes");
    }
  }
}

template <class T>
Span<T> Plan::regionSpan(Region region, size_t elementOffset) const {
  const size_t capacity = layout_.capacity[ordinal(region)] / sizeof(T);
  require(elementOffset <= capacity, "region offset out of range");
  return Span<T>{reinterpret_cast<T*>(workspace_ + layout_.offset[ordinal(region)]) + elementOffset,
                 capacity - elementOffset};
}

size_t Plan::weightBytes(Weight weight, int entry) const {
  return weightElements(weight, entry, packed_.header) * weightElementBytes(weight);
}

template <class T>
Span<const T> Plan::weight(Weight weight, int entry) const {
  const int32_t offset = packed_.header.offset(weight, entry);
  if (offset < 0) {
    throw std::invalid_argument(weightName(weight, entry) + " is absent from the weight pack");
  }
  return Span<const T>{reinterpret_cast<const T*>(packed_.data + offset), weightBytes(weight, entry) / sizeof(T)};
}

template <class T>
Span<const T> Plan::section(const BatchShape& batch, Section section) const {
  const int s = ordinal(section);
  const size_t elements = axisCount(sectionAxis(section), batch, shape_.architecture.seats, shape_.architecture.tiles) *
                          packed_.header.sectionWidth[s];
  return Span<const T>{
      reinterpret_cast<const T*>(workspace_ + layout_.offset[ordinal(Region::kInput)] + batch.section[s]), elements};
}

template <class T>
Span<T> Plan::relationBias(int layer) const {
  require(packed_.header.offset(Weight::kRelationBias, layer) >= 0, "relation bias is absent from the weight pack");
  const size_t floats = relationBiasFloats(shape_.architecture.headPairs());
  return Span<T>{packed_.relationBias + layer * floats, floats};
}

template <class T>
Span<T> Plan::qkvFragments(int layer) const {
  require(layer > 0 || !kernels_.capability.folded,
          "the layer-0 QKV weights are folded into the embedding tables and have no fragment copy");
  require(packed_.header.offset(Weight::kQkv, layer) >= 0, "QKV projection is absent from the weight pack");
  const size_t elements = qkvFragmentElements(shape_);
  return Span<T>{packed_.qkv + layer * elements, elements};
}

Span<const uint16_t> Plan::mlpFragments(Weight weight, int entry) const {
  const int index = packed_.index[weightFirstEntry(weight) + entry];
  if (index < 0 || packed_.header.offset(weight, entry) < 0) {
    throw std::invalid_argument(weightName(weight, entry) + " has no MLP fragment copy");
  }
  const FragmentCopy& copy = packed_.fragments[index];
  return Span<const uint16_t>{packed_.mlp + copy.offset, static_cast<size_t>(copy.rows) * copy.columns};
}

MlpWeights Plan::mlpWeights(int layer, TokenGroup group) const {
  const int entry = ffnEntry(layer, group);
  return MlpWeights{mlpFragments(Weight::kOutput, layer),   weight<float>(Weight::kOutputBias, layer),
                    weight<float>(Weight::kFfnNorm, entry), mlpFragments(Weight::kUp, entry),
                    weight<float>(Weight::kUpBias, entry),  mlpFragments(Weight::kDown, entry),
                    weight<float>(Weight::kDownBias, entry)};
}

MlpWeights Plan::candidateWeights() const {
  return MlpWeights{
      mlpFragments(Weight::kCandidateOutput, 0),           weight<float>(Weight::kCandidateOutputBias, 0),
      weight<float>(Weight::kFfnNorm, kCandidateFfnEntry), mlpFragments(Weight::kUp, kCandidateFfnEntry),
      weight<float>(Weight::kUpBias, kCandidateFfnEntry),  mlpFragments(Weight::kDown, kCandidateFfnEntry),
      weight<float>(Weight::kDownBias, kCandidateFfnEntry)};
}

DiscardValueWeights Plan::discardWeights() const {
  return DiscardValueWeights{weight<float>(Weight::kDiscardNorm, 0), weight<uint16_t>(Weight::kDiscardHidden, 0),
                             weight<float>(Weight::kDiscardHiddenBias, 0), weight<float>(Weight::kDiscardOutput, 0),
                             weight<float>(Weight::kDiscardExact, 0)};
}

int Plan::segments(int layer, const BatchShape& batch, bool last, MlpSegment (&segment)[2]) const {
  const Architecture& a = shape_.architecture;
  const auto finalNorm = [&](TokenGroup group) {
    return last ? weight<float>(Weight::kFinalNorm, ordinal(group)) : Span<const float>{nullptr, 0};
  };
  const int fixed = a.fixed() * batch.rows;
  if (!a.typed(layer)) {
    segment[0] = MlpSegment{0, fixed, mlpWeights(layer, TokenGroup::kAll), a.ffn(layer, TokenGroup::kAll),
                            finalNorm(TokenGroup::kAll)};
    return 1;
  }
  const int context = (1 + a.seats) * batch.rows;
  segment[0] = MlpSegment{0, context, mlpWeights(layer, TokenGroup::kContext), a.ffn(layer, TokenGroup::kContext),
                          finalNorm(TokenGroup::kContext)};
  segment[1] = MlpSegment{context, fixed - context, mlpWeights(layer, TokenGroup::kTile),
                          a.ffn(layer, TokenGroup::kTile), finalNorm(TokenGroup::kTile)};
  return 2;
}

void Plan::eventKeyValues(const PlanStep& step, const BatchShape& batch, MlpArgs& args) const {
  const Architecture& a = shape_.architecture;
  const size_t width = static_cast<size_t>(2) * shape_.attention();
  for (int l = step.layer + 2; l < a.depth; ++l) {
    const Span<const float> bias = weight<float>(Weight::kQkvBias, l);
    args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{
        weight<float>(Weight::kAttentionNorm, l), mlpFragments(Weight::kQkv, l),
        Span<const float>{bias.data + shape_.attention(), bias.limit - shape_.attention()},
        regionSpan<uint16_t>(Region::kEventKeyValue, static_cast<size_t>(l - step.layer - 2) * batch.events * width)};
  }
  if (step.writes(Region::kRereadMemory)) {
    const size_t fixed = static_cast<size_t>(a.fixed()) * batch.rows;
    args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{
        weight<float>(Weight::kRereadEventNorm, 0), mlpFragments(Weight::kRereadKeyValue, 0),
        weight<float>(Weight::kRereadKeyValueBias, 0), regionSpan<uint16_t>(Region::kRereadMemory, fixed * width)};
  }
}

TokenLayout Plan::layout(const BatchShape& batch) const {
  return TokenLayout{batch.rows, shape_.architecture.seats, shape_.architecture.tiles};
}

void Plan::countGroupRows(const char* rowColumn, const BatchShape& batch) {
  const int width = packed_.header.width(Section::kRowColumn);
  const int countColumn = packed_.header.value(Schema::kRowEvents);
  const int firstColumn = packed_.header.value(Schema::kRowFirstEvent);
  const int transitionFirstColumn = packed_.header.value(Schema::kRowFirstTransition);
  const int transitionCountColumn = packed_.header.value(Schema::kRowTransitions);
  const int fixed = shape_.architecture.fixed();
  std::fill(std::begin(groupRows_), std::end(groupRows_), 0);
  auto cell = [&](const char* row, int column) {
    int32_t value;
    std::memcpy(&value, row + column * sizeof(int32_t), sizeof(value));
    return value;
  };
  int transitions = 0;
  for (int r = 0; r < batch.rows; ++r) {
    const char* row = rowColumn + static_cast<size_t>(r) * width * sizeof(int32_t);
    const int32_t first = cell(row, firstColumn);
    const int32_t events = cell(row, countColumn);
    if (events < 0 || events > shape_.architecture.maxEvents || first < 0 || first > batch.events ||
        events > batch.events - first) {
      throw std::invalid_argument("event span outside row or batch at row " + std::to_string(r));
    }
    // 遷移は行の順に隙間なく並ぶ（候補のカーネルは行の最初の遷移を二分探索し、行の遷移の範囲に書く）。
    const int32_t firstTransition = cell(row, transitionFirstColumn);
    const int32_t rowTransitions = cell(row, transitionCountColumn);
    if (firstTransition != transitions || rowTransitions < 0 || rowTransitions > batch.transitions - transitions) {
      throw std::invalid_argument("transition span of row " + std::to_string(r) + " does not follow the previous row");
    }
    transitions += rowTransitions;
    ++groupRows_[attentionGroup(fixed, events)];
  }
  if (transitions != batch.transitions) {
    throw std::invalid_argument("rows hold " + std::to_string(transitions) + " transitions but the batch has " +
                                std::to_string(batch.transitions));
  }
}

Plan::Submit Plan::submitFunction(Kernel kernel) {
  static constexpr std::array<Submit, kKernels> functions{
      &Plan::submitEmbed,     &Plan::submitRelation, &Plan::submitAttention, &Plan::submitMlpAll,
      &Plan::submitAttention, &Plan::submitMlpFixed, &Plan::submitMlpLast,   &Plan::submitRereadKeyValue,
      &Plan::submitCandidate, &Plan::submitReread,   &Plan::submitValueGraph};
  return functions.at(static_cast<size_t>(kernel));
}

void Plan::submitEmbed(const PlanStep&, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const int attention = shape_.attention();
  const TokenLayout tokens = layout(batch);
  const int fixed = tokens.fixed() * batch.rows;
  const int allTokens = fixed + batch.events;
  EmbedArgs args{};
  args.pack = Span<const uint8_t>{reinterpret_cast<const uint8_t*>(packed_.data), packed_.bytes};
  args.spec = reinterpret_cast<const EmbeddingSpec*>(packed_.data + offsetof(PackHeader, embedding));
  args.input =
      Span<const uint8_t>{reinterpret_cast<const uint8_t*>(workspace_ + layout_.offset[ordinal(Region::kInput)]),
                          layout_.capacity[ordinal(Region::kInput)]};
  for (int s = 0; s < kSections; ++s) {
    args.section[s] = batch.section[s];
    args.sectionWidth[s] = packed_.header.sectionWidth[s];
  }
  const size_t starts[kEmbeddingSlots] = {0, static_cast<size_t>(batch.rows),
                                          static_cast<size_t>(1 + tokens.seats) * batch.rows,
                                          static_cast<size_t>(fixed), 0};
  const int counts[kEmbeddingSlots] = {batch.rows, tokens.seats * batch.rows, tokens.tiles * batch.rows, batch.events,
                                       batch.transitions};
  int blocks = 0;
  for (int e = 0; e < kEmbeddingSlots; ++e) {
    if (packed_.header.embedding[e].tables == 0) {
      throw std::invalid_argument("embedding slot " + std::to_string(e) + " is absent from the weight pack");
    }
    args.count[e] = counts[e];
    args.blockStart[e] = blocks;
    blocks += (counts[e] + 63) / 64;
    const bool transition = static_cast<EmbeddingSlot>(e) == EmbeddingSlot::kTransition;
    // 前段 FFNは代表から読んだ残差を各行へ書く。固定トークンの代表は、最終 FFNまで空いている STATE に置く。
    Region residual =
        paged_ && kernels_.pageMlp && e < ordinal(EmbeddingSlot::kEvent) ? Region::kState : Region::kResidual;
    args.out[e] = regionSpan<uint16_t>(transition ? Region::kStatic : residual, starts[e] * hidden);
    args.qkv0[e] =
        transition ? Span<uint16_t>{nullptr, 0} : regionSpan<uint16_t>(Region::kQkv0, starts[e] * 3 * attention);
  }
  args.blockStart[kEmbeddingSlots] = blocks;
  args.epsilon = a.epsilon();
  if (paged_ && saveEventProjection()) gatherEventFeatures(batch, args);
  if (kernels_.projectedEmbed) {
    Span<uint16_t> normalized[kEmbeddingSlots]{};
    for (int e = 0; e < ordinal(EmbeddingSlot::kTransition); ++e) {
      normalized[e] = regionSpan<uint16_t>(Region::kNormalized, starts[e] * hidden);
    }
    ProjectionIndex index{};
    if (paged_ && kernels_.sharedProjection) {
      // 同じ提出の途中では入力を直接照合し、先行の提出では保存したキーを照合する。
      if (++projectionCache_.generation >= std::numeric_limits<unsigned>::max() / kEmbeddingSlots) {
        PRISM_GPU(gpu::clear(projectionCache_.tags, 0, size_t(projectionCache_.rows) * sizeof(*projectionCache_.tags),
                             stream_));
        projectionCache_.generation = 1;
      }
      index = {reinterpret_cast<int*>(workspace_ + layout_.projectionIndexOffset), allTokens, projectionCache_};
      indexLayerInputs(args, packed_.header.embedding, index, stream_);
    }
    kernels_.projectedEmbed(args, weight<float>(Weight::kAttentionNorm, 0), normalized, index, stream_);
  } else {
    kernels_.embed(hidden, args, blocks, embedShared_, stream_);
  }
  if (paged_ && saveEventProjection()) exchangePages(batch, true);
}

void Plan::submitRelation(const PlanStep&, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const TokenLayout tokens = layout(batch);
  RelationArgs args{};
  args.relation = section<uint8_t>(batch, Section::kRelation);
  args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
  args.tileRelation = weight<uint8_t>(Weight::kTileRelation, 0);
  args.codes = regionSpan<uint32_t>(Region::kRelationCode, 0);
  args.layout = tokens;
  args.rowWidth = packed_.header.width(Section::kRowColumn);
  args.rowFirstEvent = packed_.header.value(Schema::kRowFirstEvent);
  args.rowEvents = packed_.header.value(Schema::kRowEvents);
  args.codeBlocks = a.codeBlocks();
  args.columns = RelationColumns{packed_.header.width(Section::kRelation),
                                 packed_.header.value(Schema::kRelationTile),
                                 packed_.header.value(Schema::kRelationOwner),
                                 packed_.header.value(Schema::kRelationEvent),
                                 packed_.header.value(Schema::kRelationRiichi),
                                 packed_.header.value(Schema::kRelationLastDiscard),
                                 packed_.header.value(Schema::kRelationLink),
                                 packed_.header.value(Schema::kRelationSafety),
                                 a.opponents()};
  // 同じ起動で、行の番号も組ごとに並べる。関係のコードと行の順序はバッチごとに 1 回作り、各層が共有する。
  RowOrderArgs order{};
  order.rowColumn = args.rowColumn;
  order.rowOrder = Span<int32_t>{rowOrder_, static_cast<size_t>(batch.rows)};
  order.rows = batch.rows;
  order.fixed = tokens.fixed();
  order.rowWidth = args.rowWidth;
  order.rowEvents = args.rowEvents;
  for (int g = 0, start = 0; g < kAttentionGroups; start += groupRows_[g++]) {
    order.groupStart[g] = start;
    order.groupRows[g] = groupRows_[g];
  }
  if (paged_)
    launchPageRelation(args, pageMap_, stream_);
  else
    launchRelation(args, stream_);
  launchRowOrder(order, stream_);
}

void Plan::submitAttention(const PlanStep& step, const BatchShape& batch) {
  const auto layer = step.layer;
  const auto kernel = step.kernel;
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const int attention = shape_.attention();
  const TokenLayout tokens = layout(batch);
  const int fixed = tokens.fixed() * batch.rows;
  const int allTokens = fixed + batch.events;
  // 全トークンの層 0 は射影をせず、埋め込みが書いた qkv0 を読む。ほかの層は n と W_qkv の写しで射影する。
  const bool fixedLayer = kernel == Kernel::kAttentionFixed;
  const bool folded = !step.reads(Region::kNormalized);
  AttentionArgs args{};
  args.o = regionSpan<uint16_t>(Region::kAttention, 0);
  args.codes = regionSpan<const uint32_t>(Region::kRelationCode, 0);
  args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
  args.rowOrder = Span<const int32_t>{rowOrder_, static_cast<size_t>(batch.rows)};
  if (folded) {
    args.qkv0 = regionSpan<const uint16_t>(Region::kQkv0, 0);
  } else {
    args.normalized = regionSpan<const uint16_t>(Region::kNormalized, 0);
    args.qkv = qkvFragments<const uint16_t>(layer);
    args.qkvBias = weight<float>(Weight::kQkvBias, layer);
  }
  args.bias = relationBias<const float>(layer);
  args.layout = tokens;
  args.rowWidth = packed_.header.width(Section::kRowColumn);
  args.rowFirstEvent = packed_.header.value(Schema::kRowFirstEvent);
  args.rowEvents = packed_.header.value(Schema::kRowEvents);
  args.scale = a.scale();
  args.role = fixedLayer ? LayerRole::kFixedTokens : LayerRole::kAllTokens;
  args.codeBlocks = a.codeBlocks();
  // 最初の局・席・牌の層は、最後の全トークンの層の前段 FFNが全てのトークンについて書いた n から、イベントの k・v も射影する。
  // 後の局・席・牌の層は、その前段 FFNが作ったイベントの k・v を写す。
  args.eventKeys = step.reads(Region::kEventKeyValue) ? EventKeys::kCopied : EventKeys::kProjected;
  const bool firstFixed = paged_ && fixedLayer && step.reads(Region::kQkv0);
  if (firstFixed) {
    args.eventKeys = EventKeys::kCopied;
    args.eventKeyValue = regionSpan<const uint16_t>(Region::kQkv0, 0);
  } else if (args.eventKeys == EventKeys::kCopied) {
    args.eventKeyValue = regionSpan<const uint16_t>(
        Region::kEventKeyValue, static_cast<size_t>(step.copiedKeysIndex) * batch.events * 2 * attention);
  }
  if (firstFixed && kernels_.pageAttention) {
    args.eventKeyValue = {reinterpret_cast<const uint16_t*>(pageFixedKeys_),
                          size_t(pageSlots_) * 2 * shape_.attention()};
    kernels_.pageAttention(args, {pageMap_ + batch.rows, size_t(batch.events)}, a.headPairs(), groupRows_,
                           splitShortPagedHeads_, stream_);
  } else if (step.causal) {
    if (paged_) args.eventKeyValue = {reinterpret_cast<const uint16_t*>(pageMap_), size_t(batch.rows) * 2};
    if (paged_ && kernels_.sharedProjection) {
      ProjectionIndex index{reinterpret_cast<int*>(workspace_ + layout_.projectionIndexOffset), allTokens,
                            projectionCache_};
      kernels_.sharedProjection(args, index, regionSpan<uint16_t>(Region::kQkv0, 0), a.headPairs(), groupRows_,
                                stream_);
    } else {
      kernels_.causal(args, a.headPairs(), groupRows_, stream_);
    }
  } else {
    kernels_.attention(hidden, folded, args, a.headPairs(), groupRows_, stream_);
  }
}

void Plan::submitMlpAll(const PlanStep& step, const BatchShape& batch) {
  const auto layer = step.layer;
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const int attention = shape_.attention();
  const TokenLayout tokens = layout(batch);
  const int fixed = tokens.fixed() * batch.rows;
  const int allTokens = fixed + batch.events;
  MlpArgs args{};
  args.x = regionSpan<uint16_t>(Region::kResidual, 0);
  args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
  args.normalized = regionSpan<uint16_t>(Region::kNormalized, 0);
  args.nextNorm = weight<float>(Weight::kAttentionNorm, layer + 1);
  args.segments = 1;
  args.segment[0] = MlpSegment{0, allTokens, mlpWeights(layer, TokenGroup::kAll), a.ffn(layer, TokenGroup::kAll),
                               Span<const float>{nullptr, 0}};
  // 最後の全トークンの層は、x を局・席・牌の分だけ書く（局・席・牌の層はイベントの残差を読まない）。n は全てのトークンについて書く
  // （最初の局・席・牌の層の Attention がイベントの k・v も射影する）。後の局・席・牌の層と 2 段目が使うイベントのキーとバリューも、
  // その場で作る。
  args.normalizedTokens = allTokens;
  if (!step.lastAllTokenLayer) {
    args.residualTokens = allTokens;
  } else {
    args.residualTokens = fixed;
    args.firstEvent = fixed;
    eventKeyValues(step, batch, args);
    if (paged_) {
      // 最初の固定クエリ層の K/V も同じ丸めで作る。因果マスク付きの Attention が読み終えた QKV0 に詰める。
      auto bias = weight<float>(Weight::kQkvBias, layer + 1);
      args.eventKeyValue[args.eventKeyValues++] = EventKeyValue{args.nextNorm,
                                                                mlpFragments(Weight::kQkv, layer + 1),
                                                                {bias.data + attention, bias.limit - attention},
                                                                regionSpan<uint16_t>(Region::kQkv0, 0)};
      args.normalizedTokens = fixed;
    }
  }
  args.epsilon = a.epsilon();
  if (paged_ && kernels_.pageMlp) {
    MlpSources sources{
        regionSpan<const uint16_t>(Region::kState, 0),
        {projectionCache_.residual.data, projectionCache_.residual.limit},
        {reinterpret_cast<const int*>(workspace_ + layout_.projectionIndexOffset), size_t(fixed + denseEvents_)},
        {pageMap_ + batch.rows + denseEvents_, size_t(batch.events)}};
    kernels_.pageMlp(hidden, args, sources, stream_);
  } else {
    if (paged_) compactEventInputs(batch);
    kernels_.mlp(hidden, args, stream_);
  }
  if (paged_) exchangePages(batch, false);
}

void Plan::submitMlpFixed(const PlanStep& step, const BatchShape& batch) {
  const auto layer = step.layer;
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const TokenLayout tokens = layout(batch);
  const int fixed = tokens.fixed() * batch.rows;
  MlpArgs args{};
  args.x = regionSpan<uint16_t>(Region::kResidual, 0);
  args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
  args.normalized = regionSpan<uint16_t>(Region::kNormalized, 0);
  args.nextNorm = weight<float>(Weight::kAttentionNorm, layer + 1);
  args.segments = segments(layer, batch, false, args.segment);
  args.residualTokens = fixed;
  args.normalizedTokens = fixed;
  args.epsilon = a.epsilon();
  kernels_.mlp(hidden, args, stream_);
}

void Plan::submitMlpLast(const PlanStep& step, const BatchShape& batch) {
  const auto layer = step.layer;
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const TokenLayout tokens = layout(batch);
  MlpKvArgs args{};
  args.x = regionSpan<const uint16_t>(Region::kResidual, 0);
  args.o = regionSpan<const uint16_t>(Region::kAttention, 0);
  args.segments = segments(layer, batch, true, args.segment);
  args.keyValue = mlpFragments(Weight::kKeyValue, 0);
  args.keyValueBias = weight<float>(Weight::kKeyValueBias, 0);
  args.state = regionSpan<uint16_t>(Region::kState, 0);
  args.memory = regionSpan<uint16_t>(Region::kMemory, 0);
  args.layout = tokens;
  args.epsilon = a.epsilon();
  kernels_.lastMlp(hidden, args, stream_);
}

void Plan::submitRereadKeyValue(const PlanStep&, const BatchShape& batch) { launchRereadKeys(batch); }

void Plan::submitCandidate(const PlanStep& step, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const TokenLayout tokens = layout(batch);
  CandidateArgs args{};
  args.statics = regionSpan<const uint16_t>(Region::kStatic, 0);
  args.state = regionSpan<const uint16_t>(Region::kState, 0);
  args.memory = regionSpan<const uint16_t>(Region::kMemory, 0);
  args.category = section<uint8_t>(batch, Section::kTransitionCategory);
  args.code = section<uint8_t>(batch, Section::kTransitionCode);
  args.discardExact = section<float>(batch, Section::kDiscardExact);
  args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
  args.tileEntry = weight<uint16_t>(Weight::kTileEntry, 0);
  args.queryNorm = weight<float>(Weight::kQueryNorm, 0);
  args.query = weight<uint16_t>(Weight::kQuery, 0);
  args.queryBias = weight<float>(Weight::kQueryBias, 0);
  args.codeKey = weight<float>(Weight::kCodeKey, 0);
  args.codeValue = weight<float>(Weight::kCodeValue, 0);
  args.weights = candidateWeights();
  args.discard = discardWeights();
  args.transition = regionSpan<uint16_t>(Region::kTransition, 0);
  args.value = regionSpan<float>(Region::kValue, 0);
  args.layout = tokens;
  args.transitions = batch.transitions;
  args.categoryWidth = packed_.header.width(Section::kTransitionCategory);
  args.callDiscard = packed_.header.value(Schema::kTransitionCallDiscard);
  args.mainTile = packed_.header.value(Schema::kTransitionMainTile);
  args.codes = a.keyCodes;
  args.discardExactWidth = packed_.header.width(Section::kDiscardExact);
  args.rowWidth = packed_.header.width(Section::kRowColumn);
  args.rowFirstTransition = packed_.header.value(Schema::kRowFirstTransition);
  args.ffn = a.candidateFfn;
  args.gate = a.gate;
  args.keys = a.fixed();
  args.writeValue = step.writes(Region::kValue);
  args.epsilon = a.epsilon();
  args.scale = a.scale();
  const auto& options = shape_.options;
  const auto segmentWeights = [&] {
    const auto* offsets = reinterpret_cast<const float*>(packed_.data + options.segmentsOffset);
    return SegmentWeights{{offsets, execution_.candidate.segments}, options.segments};
  };
  if (execution_.candidate.countWorkspace) {
    const size_t gateCount = execution_.candidate.gate;
    const size_t matrixCount = execution_.candidate.projection;
    const auto* gate = reinterpret_cast<const float*>(packed_.data + options.countOffset);
    const auto* tile = reinterpret_cast<const uint16_t*>(gate + gateCount);
    const CountWeights extra{
        {gate, gateCount}, {tile, matrixCount}, {tile + matrixCount, matrixCount}, options.countFirstCode};
    const auto hot = candidateOffsets_ ? candidateOffsets_->prepare(args, stream_) : HotCandidateOffsets{};
    const Span<uint16_t> attentionState{candidateAttention_, static_cast<size_t>(batch.transitions) * a.attention()};
    launchCandidateCore(hidden, args, extra, segmentWeights(), countWorkspace_, stream_, hot, attentionState);
  } else {
    kernels_.standard(hidden, args, stream_);
  }
}

void Plan::submitReread(const PlanStep&, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  const TokenLayout tokens = layout(batch);
  RereadArgs args{};
  args.transition = regionSpan<uint16_t>(Region::kTransition, 0);
  args.value = regionSpan<float>(Region::kValue, 0);
  args.memory = regionSpan<const uint16_t>(Region::kRereadMemory, 0);
  args.category = section<uint8_t>(batch, Section::kTransitionCategory);
  args.code = section<uint8_t>(batch, Section::kTransitionCode);
  args.relation = section<uint8_t>(batch, Section::kRelation);
  args.discardExact = section<float>(batch, Section::kDiscardExact);
  args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
  args.tileRelation = weight<uint8_t>(Weight::kTileRelation, 0);
  args.queryNorm = weight<float>(Weight::kRereadQueryNorm, 0);
  args.query = weight<uint16_t>(Weight::kRereadQuery, 0);
  args.queryBias = weight<float>(Weight::kRereadQueryBias, 0);
  args.codeKey = weight<float>(Weight::kRereadCodeKey, 0);
  args.codeValue = weight<float>(Weight::kRereadCodeValue, 0);
  args.output = mlpFragments(Weight::kRereadOutput, 0);
  args.outputBias = weight<float>(Weight::kRereadOutputBias, 0);
  args.discard = discardWeights();
  args.layout = tokens;
  args.transitions = batch.transitions;
  args.categoryWidth = packed_.header.width(Section::kTransitionCategory);
  args.callDiscard = packed_.header.value(Schema::kTransitionCallDiscard);
  args.mainTile = packed_.header.value(Schema::kTransitionMainTile);
  args.relationWidth = packed_.header.width(Section::kRelation);
  args.relationTile = packed_.header.value(Schema::kRelationTile);
  args.keyCodes = a.keyCodes;
  args.rowWidth = packed_.header.width(Section::kRowColumn);
  args.rowFirstTransition = packed_.header.value(Schema::kRowFirstTransition);
  args.rowFirstEvent = packed_.header.value(Schema::kRowFirstEvent);
  args.rowEvents = packed_.header.value(Schema::kRowEvents);
  args.discardExactWidth = packed_.header.width(Section::kDiscardExact);
  args.gate = a.gate;
  args.epsilon = a.epsilon();
  args.scale = a.scale();
  kernels_.reread(hidden, args, stream_);
}

void Plan::submitValueGraph(const PlanStep&, const BatchShape& batch) {
  const Architecture& a = shape_.architecture;
  const int hidden = shape_.hidden();
  ValueGraphArgs args{};
  args.transition = regionSpan<const uint16_t>(Region::kTransition, 0);
  args.value = regionSpan<const float>(Region::kValue, 0);
  args.state = regionSpan<const uint16_t>(Region::kState, 0);
  args.rowColumn = section<int32_t>(batch, Section::kRowColumn);
  args.actionColumn = section<int32_t>(batch, Section::kActionColumn);
  args.actionExact = section<float>(batch, Section::kActionExact);
  args.gateNorm = weight<float>(Weight::kGateNorm, 0);
  args.gateHidden = weight<uint16_t>(Weight::kGateHidden, 0);
  args.gateHiddenBias = weight<float>(Weight::kGateHiddenBias, 0);
  args.kindOffset = weight<float>(Weight::kGateKindOffset, 0);
  args.exactProjection = weight<float>(Weight::kGateExactProjection, 0);
  args.gateOutput = weight<float>(Weight::kGateOutput, 0);
  args.gateOutputBias = weight<float>(Weight::kGateOutputBias, 0);
  args.exactOutput = weight<float>(Weight::kGateExactOutput, 0);
  args.scores = regionSpan<float>(Region::kScores, 0);
  args.rows = batch.rows;
  args.rowWidth = packed_.header.width(Section::kRowColumn);
  args.rowFirstAction = packed_.header.value(Schema::kRowFirstAction);
  args.rowActions = packed_.header.value(Schema::kRowActions);
  args.actionWidth = packed_.header.width(Section::kActionColumn);
  args.actionFirstTransition = packed_.header.value(Schema::kActionFirstTransition);
  args.actionTransitions = packed_.header.value(Schema::kActionTransitions);
  args.actionRole = packed_.header.value(Schema::kActionRole);
  args.actionBase = packed_.header.value(Schema::kActionBase);
  args.actionRepresentative = packed_.header.value(Schema::kActionRepresentative);
  args.actionRow = packed_.header.value(Schema::kActionRow);
  args.actionExactWidth = packed_.header.width(Section::kActionExact);
  args.roles = a.roles;
  for (int r = 0; r < kMaxRoles; ++r) {
    args.composition[r] = packed_.header.composition[r];
    args.gateKind[r] = packed_.header.gate[r];
  }
  args.gateKinds = a.gateKinds;
  args.gate = a.gate;
  args.epsilon = a.epsilon();
  args.tau = a.tau();
  kernels_.valueGraph(hidden, args, stream_);
}

void Plan::launchRereadKeys(const BatchShape& batch) {
  KeyValueArgs args{regionSpan<const uint16_t>(Region::kState, 0), mlpFragments(Weight::kRereadKeyValue, 0),
                    weight<float>(Weight::kRereadKeyValueBias, 0), regionSpan<uint16_t>(Region::kRereadMemory, 0),
                    layout(batch)};
  kernels_.rereadKeys(shape_.hidden(), args, stream_);
}

bool Plan::saveEventProjection() const { return kernels_.capability.folded; }

void Plan::configureCandidateInput(const void* descriptor, size_t count) {
  const auto& segments = shape_.options.segments;
  require(
      shape_.options.has(PlanOptions::kCausal) && shape_.options.has(PlanOptions::kSegments) && packed_.version >= 0,
      "candidate input restoration requires published causal segment weights");
  require(count * sizeof(int) ==
              offsetof(CandidateInputLayout, slots) + segments.count * sizeof(CandidateInputLayout::Slot),
          "candidate input descriptor length mismatch");
  CandidateInputLayout layout{};
  std::memcpy(&layout, descriptor, count * sizeof(int));
  auto column = [&](int index, Section section) {
    require(index >= 0 && index < packed_.header.width(section), "candidate input column out of range");
  };
  for (int index : {layout.kind, layout.author, layout.eventTile, layout.declaration})
    column(index, Section::kEventCategory);
  column(layout.river, Section::kTileCategory);
  column(layout.riichi, Section::kSeatCategory);
  for (auto digit : {layout.ukeire, layout.wait, layout.discarded})
    require(digit.weight > 0 && digit.radix > 0, "invalid candidate digit");
  for (int i = 0; i < segments.count; i++) {
    auto slot = layout.slots[i];
    require(slot.seat >= -1 && slot.seat < shape_.architecture.seats, "candidate input seat outside rule");
    require(slot.bonusTile >= -1 && slot.bonusTile < shape_.architecture.tiles, "candidate bonus tile outside rule");
    if (slot.seat != 0) column(slot.tileColumn, Section::kTileCategory);
  }
  candidateInput_ = layout;
  restoreCandidates_ = true;
}

size_t Plan::eventInputWidth() const {
  return packed_.header.width(Section::kEventCategory) +
         sizeof(uint16_t) * packed_.header.width(Section::kEventScalar) + packed_.header.width(Section::kRelation);
}

size_t Plan::basicCodeBytes(int transitions) const {
  return restoreCandidates_ ? size_t(transitions) * shape_.architecture.fixed() : 0;
}

EventPageRequirements Plan::pageRequirements() const {
  // 射影表の行は、1 バッチのトークンの数より多くは使わない。
  const int projectionLimit =
      kernels_.sharedProjection ? static_cast<int>(std::min<size_t>(kProjectionRowBudget, shape_.tokens())) : 0;
  return {eventInputWidth(),
          shape_.options.has(PlanOptions::kSegments) ? size_t(shape_.transitions) * shape_.architecture.fixed() : 0,
          projectionLimit, ProjectionCache::kKeyBytes, saveEventProjection()};
}

gpu::Status Plan::allocateEventPages(int slots) {
  const EventPageLayout pages(shape_, pageRequirements(), slots);
  char* storage = nullptr;
  gpu::Status status = gpu::allocate(&storage, pages.bytes);
  if (status != gpu::success) return status;
  ProjectionCache cache{};
  cache.rows = pages.projectionRows;
  if (cache.rows) {
    cache.tags = reinterpret_cast<unsigned long long*>(storage + pages.projections);
    cache.keys = reinterpret_cast<uint8_t*>(cache.tags + cache.rows);
    cache.residual = {reinterpret_cast<uint16_t*>(cache.keys + size_t(cache.rows) * cache.kKeyBytes),
                      size_t(cache.rows) * shape_.hidden()};
    cache.qkv = {cache.residual.data + cache.residual.limit, size_t(cache.rows) * 3 * shape_.attention()};
    status = gpu::clear(cache.tags, 0, size_t(cache.rows) * sizeof(*cache.tags), stream_);
    if (status != gpu::success) {
      (void)gpu::release(storage);
      return status;
    }
  }
  pageStorage_ = storage;
  pageSlots_ = slots;
  pageMap_ = reinterpret_cast<int*>(storage + pages.map);
  pageFeatures_ = reinterpret_cast<uint8_t*>(storage + pages.features);
  pageInput_ = saveEventProjection() ? storage + pages.embedding : nullptr;
  pageFixedKeys_ = storage + pages.fixedKeys;
  pageExtraKeys_ = storage + pages.extraKeys;
  compactAttention_ = reinterpret_cast<uint16_t*>(storage + pages.scratch);
  projectionCache_ = cache;
  return gpu::success;
}

LaneCapacity Plan::configureEventPages(int remaining, int events, double reserveFraction) {
  const auto& a = shape_.architecture;
  require(remaining > 0 && events >= -1 && reserveFraction >= 0 && reserveFraction <= 1 && !pageStorage_,
          "invalid event page allocation request");
  gpu::SelectedDevice scope(device_);
  size_t before, after, total;
  PRISM_GPU(gpu::memoryInfo(&before, &total));
  LaneCapacity capacity{};
  capacity.freeBefore = capacity.freeAfter = int64_t(before);
  capacity.total = int64_t(total);
  capacity.dictionary = candidateOffsets_ && candidateOffsets_->available();
  capacity.splitCandidate = candidateAttention_ != nullptr;
  if (events == -1) return capacity;
  require(shape_.options.has(PlanOptions::kCausal) && a.allTokenLayers() == 1 && !a.reread(),
          "event cache needs a prefix-independent event trunk");
  require(packed_.version >= 0, "event pages require published weights");
  if (events > 0) {
    const size_t needed = EventPageLayout(shape_, pageRequirements(), events).bytes;
    requireAllocated(needed > before ? gpu::outOfMemory : allocateEventPages(events), needed, "the event pages");
  } else {
    // 1 バッチのイベントが収まらないページ数は確保しない（収まらない提出は全てのイベントを計算し直す）。
    const size_t headroom = static_cast<size_t>(total * reserveFraction);
    const size_t budget = before > headroom ? (before - headroom) / remaining : 0;
    const size_t batch = static_cast<size_t>(shape_.events);
    size_t slots = EventPageLayout::fittingSlots(shape_, pageRequirements(), budget);
    while (slots >= batch && slots > 0) {
      const gpu::Status status = allocateEventPages(static_cast<int>(slots));
      if (status == gpu::success) break;
      (void)gpu::lastError();
      if (status != gpu::outOfMemory) PRISM_GPU(status);
      if (slots == batch) break;
      slots = std::max(batch, slots * 3 / 4);
    }
  }
  PRISM_GPU(gpu::memoryInfo(&after, &total));
  const EventPageLayout layout(shape_, pageRequirements(), pageSlots_);
  capacity.freeAfter = int64_t(after);
  capacity.pageSlots = pageSlots_;
  capacity.projectionRows = layout.projectionRows;
  capacity.stagingBytes = int64_t(layout.features + layout.projections - layout.scratch);
  capacity.pageBytes = int64_t(layout.scratch - layout.features);
  capacity.projectionBytes = int64_t(layout.bytes - layout.projections);
  return capacity;
}

void Plan::uploadPageMap(const BatchShape& batch, const int* map, int misses, int slotCount, size_t transferBytes) {
  const auto& a = shape_.architecture;
  require(shape_.options.has(PlanOptions::kCausal) && a.allTokenLayers() == 1 && !a.reread(),
          "event cache needs a prefix-independent event trunk");
  require(misses >= 0 && misses <= batch.events, "invalid event delta");
  require(slotCount > 0 && slotCount >= batch.events && (!pageStorage_ || slotCount == pageSlots_),
          "invalid event cache capacity");
  const size_t mapBytes = (size_t(batch.rows) + batch.events + misses) * sizeof(int);
  const int* slots = map + batch.rows;
  for (int e = 0; e < batch.events; ++e)
    if (static_cast<unsigned>(slots[e]) >= static_cast<unsigned>(slotCount))
      throw std::invalid_argument("event slot outside cache");
  for (int e = 0; e < misses; ++e)
    if (static_cast<unsigned>(slots[batch.events + e]) >= static_cast<unsigned>(batch.events))
      throw std::invalid_argument("event delta index outside batch");
  if (!pageStorage_) {
    requireAllocated(allocateEventPages(slotCount), EventPageLayout(shape_, pageRequirements(), slotCount).bytes,
                     "the event pages");
  }
  require(!transferBytes ||
              transferBytes == mapBytes + size_t(misses) * eventInputWidth() + basicCodeBytes(batch.transitions),
          "event transfer size mismatch");
  PRISM_GPU(gpu::copy(pageMap_, map, transferBytes ? transferBytes : mapBytes, gpu::hostToDevice, stream_));
  auto* device = reinterpret_cast<uint8_t*>(workspace_ + layout_.offset[ordinal(Region::kInput)]);
  EventInputColumns columns{};
  columns.categoryWidth = packed_.header.width(Section::kEventCategory);
  columns.scalarBytes = sizeof(uint16_t) * packed_.header.width(Section::kEventScalar);
  columns.relationWidth = packed_.header.width(Section::kRelation);
  columns.categories = device + batch.section[ordinal(Section::kEventCategory)];
  columns.scalars = device + batch.section[ordinal(Section::kEventScalar)];
  columns.relations = device + batch.section[ordinal(Section::kRelation)] +
                      size_t(batch.rows) * shape_.architecture.fixed() * columns.relationWidth;
  const int* deviceSlots = pageMap_ + batch.rows;
  const int* missing = deviceSlots + batch.events;
  const auto* raw = transferBytes ? reinterpret_cast<const uint8_t*>(missing + misses) : nullptr;
  restoreEventInputs(columns, pageFeatures_, deviceSlots, missing, batch.events, misses, raw, stream_);
  if (transferBytes && restoreCandidates_) {
    auto at = [&](Section section) { return device + batch.section[ordinal(section)]; };
    CandidateInput input{reinterpret_cast<const int*>(at(Section::kRowColumn)),
                         columns.categories,
                         at(Section::kTileCategory),
                         at(Section::kSeatCategory),
                         at(Section::kRelation) + size_t(batch.rows) * (a.fixed() - a.tiles) * columns.relationWidth,
                         raw + size_t(misses) * eventInputWidth(),
                         at(Section::kTransitionCode),
                         packed_.header.width(Section::kRowColumn),
                         packed_.header.value(Schema::kRowFirstEvent),
                         packed_.header.value(Schema::kRowEvents),
                         packed_.header.value(Schema::kRowFirstTransition),
                         packed_.header.value(Schema::kRowTransitions),
                         columns.categoryWidth,
                         packed_.header.width(Section::kTileCategory),
                         packed_.header.width(Section::kSeatCategory),
                         columns.relationWidth,
                         packed_.header.value(Schema::kRelationTile),
                         a.tiles,
                         a.seats,
                         a.fixed()};
    restoreCandidateCodes(input, candidateInput_, shape_.options.segments, batch.rows, stream_);
  }
  denseEvents_ = batch.events;
}

void Plan::uploadInputs(const void* input, const BatchShape& batch, bool sparse) {
  char* device = workspace_ + layout_.offset[ordinal(Region::kInput)];
  const auto* host = static_cast<const char*>(input);
  auto copy = [&](size_t first, size_t end) {
    if (end > first) PRISM_GPU(gpu::copy(device + first, host + first, end - first, gpu::hostToDevice, stream_));
  };
  if (!sparse) {
    copy(0, batch.bytes);
    return;
  }
  copy(0, batch.section[ordinal(Section::kEventCategory)]);
  size_t relation = batch.section[ordinal(Section::kRelation)];
  copy(relation,
       relation + size_t(batch.rows) * shape_.architecture.fixed() * packed_.header.width(Section::kRelation));
  size_t transitions = batch.section[ordinal(Section::kTransitionCategory)];
  if (!restoreCandidates_) {
    copy(transitions, batch.bytes);
    return;
  }
  size_t codes = batch.section[ordinal(Section::kTransitionCode)];
  // 候補の基本コードは配置表の後ろに連続している。遷移ごとに短い H2D を発行しない。
  copy(transitions, codes);
  copy(batch.section[ordinal(Section::kDiscardExact)], batch.bytes);
}

void Plan::gatherEventFeatures(const BatchShape& batch, EmbedArgs& args) {
  if (!batch.events) return;
  auto* input = reinterpret_cast<uint8_t*>(workspace_ + layout_.offset[ordinal(Region::kInput)]);
  auto* categories = input + batch.section[ordinal(Section::kEventCategory)];
  auto* scalars = reinterpret_cast<uint16_t*>(input + batch.section[ordinal(Section::kEventScalar)]);
  int categoryWidth = packed_.header.width(Section::kEventCategory),
      scalarWidth = packed_.header.width(Section::kEventScalar);
  size_t catBytes = size_t(batch.events) * categoryWidth, scalarBytes = size_t(batch.events) * scalarWidth * 2;
  // 埋め込みの間はまだ読まない Attention の出力領域に詰め、埋め込みに渡す入力のうちイベントの 2 列だけをそこに向ける。
  size_t scalarOffset = (catBytes + 1) & ~size_t(1);
  require(scalarOffset + scalarBytes <= regionBytes(Region::kAttention), "event features exceed scratch");
  size_t offset = layout_.offset[ordinal(Region::kAttention)] - layout_.offset[ordinal(Region::kInput)];
  require(offset + scalarOffset + scalarBytes <= std::numeric_limits<uint32_t>::max(),
          "embedding offset exceeds 32 bits");
  auto* scratch = input + offset;
  auto* packedScalars = reinterpret_cast<uint16_t*>(scratch + scalarOffset);
  gatherPageFeatures(categories, scalars, categoryWidth, scalarWidth, scratch, packedScalars,
                     pageMap_ + batch.rows + denseEvents_, batch.events, stream_);
  args.input.limit = offset + scalarOffset + scalarBytes;
  args.section[ordinal(Section::kEventCategory)] = offset;
  args.section[ordinal(Section::kEventScalar)] = offset + scalarOffset;
}

void Plan::exchangePages(const BatchShape& batch, bool embedding) {
  const size_t fixed = static_cast<size_t>(batch.rows) * shape_.architecture.fixed();
  PageColumns columns{};
  const int* slots = pageMap_ + batch.rows;
  if (embedding) {
    int width = 3 * shape_.attention();
    auto* input = regionSpan<uint16_t>(Region::kQkv0, fixed * width).data;
    columns.column[columns.count++] = {input, input, pageInput_, width * int(sizeof(uint16_t))};
  } else {
    const int width = 2 * shape_.attention(), bytes = width * int(sizeof(uint16_t));
    auto* first = regionSpan<uint16_t>(Region::kQkv0, 0).data;
    columns.column[columns.count++] = {first, first, pageFixedKeys_, bytes};
    if (kernels_.pageAttention) {
      // 新しい K/V だけを書き足す。最初の固定クエリの層は、このページをそのまま読む。
      exchangeEventPages(slots, slots + denseEvents_, 0, batch.events, columns, stream_);
      columns.count = 0;
    }
    for (int layer = 0; layer < shape_.architecture.copiedEventLayers(); ++layer) {
      const size_t offset = size_t(layer) * width;
      auto* compact = regionSpan<uint16_t>(Region::kEventKeyValue, offset * batch.events).data;
      auto* dense = regionSpan<uint16_t>(Region::kEventKeyValue, offset * denseEvents_).data;
      columns.column[columns.count++] = {compact, dense, pageExtraKeys_ + size_t(layer) * pageSlots_ * bytes, bytes};
    }
  }
  if (columns.count) exchangeEventPages(slots, slots + denseEvents_, denseEvents_, batch.events, columns, stream_);
}

void Plan::compactEventInputs(const BatchShape& batch) {
  const int fixed = batch.rows * shape_.architecture.fixed();
  const int* misses = pageMap_ + batch.rows + denseEvents_;
  if (!batch.events) return;
  auto compact = [&](Region region, int width) {
    auto* dense = regionSpan<uint16_t>(region, static_cast<size_t>(fixed) * width).data;
    compactEventRows(dense, compactAttention_, misses, batch.events, width, stream_);
    PRISM_GPU(gpu::copy(dense, compactAttention_, size_t(batch.events) * width * sizeof(uint16_t), gpu::deviceToDevice,
                        stream_));
  };
  // 前段 FFNに渡す残差と Attention の出力を、同じ未計算イベントの順に詰める。
  if (!saveEventProjection()) compact(Region::kResidual, shape_.hidden());
  compact(Region::kAttention, shape_.attention());
}

void Plan::forward(const void* input, const BatchShape& batch, void* scores, int64_t version, const int* eventMap,
                   int misses, int slots, size_t transferBytes) {
  validate(batch);
  if (version != packed_.version) {
    throw std::invalid_argument("weight pack version " + std::to_string(packed_.version) +
                                " is stale; the network is at " + std::to_string(version));
  }
  gpu::SelectedDevice scope(device_);
  countGroupRows(static_cast<const char*>(input) + batch.section[ordinal(Section::kRowColumn)], batch);
  require(eventMap || (misses == 0 && slots == 0 && transferBytes == 0), "event submission has no page map");
  require(!eventMap || pagesUsable_, "event pages were invalidated by a failed submission");
  chooseHotOffsets(input, batch);
  BatchShape compact = batch;
  paged_ = eventMap != nullptr;
  try {
    uploadInputs(input, batch, transferBytes != 0);
    if (eventMap) {
      uploadPageMap(batch, eventMap, misses, slots, transferBytes);
      compact.events = misses;
    }
    for (size_t i = 0; i < execution_.steps.size(); ++i) {
      const auto& step = execution_.steps[i];
      (this->*submits_[i])(step, step.compactEvents ? compact : batch);
    }
    PRISM_GPU(gpu::copy(scores, workspace_ + layout_.offset[ordinal(Region::kScores)],
                        static_cast<size_t>(batch.actions) * sizeof(float), gpu::deviceToHost, stream_));
  } catch (...) {
    if (paged_) pagesUsable_ = false;
    paged_ = false;
    throw;
  }
  paged_ = false;
}

}  // namespace prism
