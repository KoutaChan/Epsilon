// V4-only launch arguments. The released kernel arguments and PackHeader are fixed.
#pragma once
#include "prism_hip_candidate_layout.h"
#include "prism_hip_event_pages.h"
#include "prism_hip_kernels.h"
#include "prism_hip_training.h"

namespace prism {

void launchV4WideCausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                                 hipStream_t stream);
void launchV4WideAttention(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                           const int (&groupRows)[kAttentionGroups], hipStream_t stream);
void launchV4WideQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, hipStream_t stream);
void launchV4WideMlp(int hidden, const MlpArgs& args, hipStream_t stream);
void launchV4WideMlpKv(int hidden, const MlpKvArgs& args, hipStream_t stream);
void launchV4WideValueGraph(int hidden, const ValueGraphArgs& args, hipStream_t stream);

void launchV4CausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                             hipStream_t stream);

void causalAttentionForward(const AttentionShape& shape, int fixed, const AttentionForward& arrays, hipStream_t stream);
void causalAttentionBackward(const AttentionShape& shape, int fixed, const AttentionBackward& arrays,
                             hipStream_t stream);

}  // namespace prism

namespace prism::v4 {

// K1 が作り、K2 と K3 も同じバッチで読む。表の後ろの 1 要素は代表の件数。
struct ProjectionIndex {
  int* owners{};
  int tokens{};
  ProjectionCache cache{};
  __host__ __device__ int* sources() const { return owners + tokens; }
  __host__ __device__ int* table() const { return sources() + tokens; }
  __host__ __device__ int* count() const { return table() + tokens; }
  static size_t bytes(size_t tokens) { return (3 * tokens + 1) * sizeof(int); }
};

// 最初の固定注意が、観測席で共有するイベントの K/V をページ番号で直接読む。
void launchPageAttention(const AttentionArgs& args, Span<const int> slots, int headPairs,
                         const int (&groupRows)[kAttentionGroups], hipStream_t stream);
// 固定行の残差は最後の層まで空いている STATE、イベントの残差は密な RESIDUAL にある。
struct MlpSources {
  Span<const uint16_t> fixed;
  Span<const uint16_t> cached;
  Span<const int> owners;
  Span<const int> misses;
};
void launchPageMlp(int hidden, const MlpArgs& args, MlpSources sources, hipStream_t stream);
void indexLayerInputs(const EmbedArgs& args, const EmbeddingSpec (&specs)[kEmbeddingSlots], ProjectionIndex index,
                      hipStream_t stream);
void launchSharedProjection(AttentionArgs args, ProjectionIndex index, Span<uint16_t> projections, int headPairs,
                            const int (&groupRows)[kAttentionGroups], hipStream_t stream);

void launchWideEmbed(const EmbedArgs& args, Span<const float> gain, const Span<uint16_t> (&normalized)[kEmbeddingSlots],
                     ProjectionIndex index, hipStream_t stream);

struct CountWeights {
  Span<const float> gate;
  Span<const uint16_t> tile;
  Span<const uint16_t> output;
  int firstCode;
};

struct SegmentWeights {
  Span<const float> offsets;  // [segment codes, 2 * attention], key then value
  SegmentLayout layout;
};

inline constexpr int kHotCodeTag = 1 << 15;

struct HotCandidateOffsets {
  Span<const float> keys;
  Span<const float> values;
  Span<const uint16_t> codes;
};

void launchCandidateCore(int hidden, const CandidateArgs& base, const CountWeights& count,
                         const SegmentWeights& segments, uint16_t* workspace, hipStream_t stream,
                         HotCandidateOffsets hot = {}, Span<uint16_t> attentionState = {});
void launchCandidateWithOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                SegmentWeights segments, hipStream_t stream);
void launchCandidateWithHotOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                   SegmentWeights segments, HotCandidateOffsets hot, hipStream_t stream,
                                   Span<uint16_t> attentionState);

struct CountTraining {
  int rows, transitions, tiles, codes, width;
};

void launchCountTraining(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                         const int32_t* codes, uint16_t* output, hipStream_t stream);
void launchCountTrainingBackward(CountTraining shape, const uint16_t* projected, const float* gates,
                                 const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                 uint16_t* gradProjected, float* gradGate, hipStream_t stream);

struct SegmentTraining {
  int transitions, keys, tiles, seats, baseCodes, width;
  SegmentLayout layout;
};

struct SegmentAttention {
  SegmentTraining input;
  int rows, heads;
  float scale;
};

struct SegmentAttentionArrays {
  const float *query, *memory, *base, *segments;
  const uint8_t* codes;
  const int32_t* groups;
};

struct SegmentAttentionBackward {
  const float *probs, *upstream;
  float *query, *memory, *base, *segments, *score, *partial;
  int32_t *cursor, *offsets, *occurrences;
};

void launchSegmentAttention(SegmentAttention shape, SegmentAttentionArrays arrays, float* output, float* probs,
                            hipStream_t stream);
void launchSegmentAttentionBackward(SegmentAttention shape, SegmentAttentionArrays arrays,
                                    SegmentAttentionBackward gradients, hipStream_t stream);

}  // namespace prism::v4
