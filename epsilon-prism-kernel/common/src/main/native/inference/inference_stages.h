#pragma once

#include "event_submission.h"

namespace prism {
void launchProjectedCausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                                    gpu::Stream stream);
void launchProjectedAttention(int hidden, bool folded, const AttentionArgs& args, int headPairs,
                              const int (&groupRows)[kAttentionGroups], gpu::Stream stream);
void launchProjectedQkvFragments(int hidden, const QkvFragmentArgs& args, int headPairs, gpu::Stream stream);

void launchFoldedCausalAttention(const AttentionArgs& args, int headPairs, const int (&groupRows)[kAttentionGroups],
                                 gpu::Stream stream);

// 最初の固定クエリの層が、観測席で共有するイベントの K/V をページ番号で直接読む。
void launchPageAttention(const AttentionArgs& args, Span<const int> slots, int headPairs,
                         const int (&groupRows)[kAttentionGroups], bool splitShortHeads, gpu::Stream stream);
// 固定行の残差は最後の層まで空いている STATE、イベントの残差は密な RESIDUAL にある。
struct MlpSources {
  Span<const uint16_t> fixed;
  Span<const uint16_t> cached;
  Span<const int> owners;
  Span<const int> misses;
};
void launchPageMlp(int hidden, const MlpArgs& args, MlpSources sources, gpu::Stream stream);
void indexLayerInputs(const EmbedArgs& args, const EmbeddingSpec (&specs)[kEmbeddingSlots], ProjectionIndex index,
                      gpu::Stream stream);
void launchSharedProjection(AttentionArgs args, ProjectionIndex index, Span<uint16_t> projections, int headPairs,
                            const int (&groupRows)[kAttentionGroups], gpu::Stream stream);

void launchWideEmbed(const EmbedArgs& args, Span<const float> gain, const Span<uint16_t> (&normalized)[kEmbeddingSlots],
                     ProjectionIndex index, gpu::Stream stream);

struct CountWeights {
  Span<const float> gate;
  Span<const uint16_t> tile;
  Span<const uint16_t> output;
  int firstCode;
};

struct SegmentWeights {
  Span<const float> offsets;  // [区分のコード, 2 × Attention の幅]（キーの補正、次にバリューの補正）
  SegmentLayout layout;
};

inline constexpr int kHotCodeTag = 1 << 15;

struct HotCandidateOffsets {
  Span<const float> keys;
  Span<const float> values;
  Span<const uint16_t> codes;
};

void launchCandidateCore(int hidden, const CandidateArgs& base, const CountWeights& count,
                         const SegmentWeights& segments, uint16_t* workspace, gpu::Stream stream,
                         HotCandidateOffsets hot = {}, Span<uint16_t> attentionState = {});
void launchCandidateWithOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                SegmentWeights segments, gpu::Stream stream);
void launchCandidateWithHotOffsets(int hidden, const CandidateArgs& args, Span<const uint16_t> count,
                                   SegmentWeights segments, HotCandidateOffsets hot, gpu::Stream stream,
                                   Span<uint16_t> attentionState);

}  // namespace prism
