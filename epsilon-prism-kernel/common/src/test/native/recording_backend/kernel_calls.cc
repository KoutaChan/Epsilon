#include "candidate_classification.h"
#include "inference_stages.h"

namespace prism {
namespace {
void launch(gpu::Stream stream) { test::append(test::Command::kLaunch, stream); }
}

size_t embedSharedBytes(int, const EmbeddingSpec (&)[kEmbeddingSlots]) { return 0; }
void launchEmbed(int, const EmbedArgs&, int, size_t, gpu::Stream stream) { launch(stream); }
void launchRelation(const RelationArgs&, gpu::Stream stream) { launch(stream); }
void launchRelationBias(const RelationBiasArgs&, int, gpu::Stream stream) { launch(stream); }
void launchQkvFragments(int, const QkvFragmentArgs&, int, gpu::Stream stream) { launch(stream); }
void launchRowOrder(const RowOrderArgs&, gpu::Stream stream) { launch(stream); }
void launchAttention(int, bool, const AttentionArgs&, int, const int (&)[kAttentionGroups], gpu::Stream stream) {
  launch(stream);
}
void launchMlpFragments(const MlpFragmentArgs&, gpu::Stream stream) { launch(stream); }
void launchMlp(int, const MlpArgs&, gpu::Stream stream) { launch(stream); }
void launchMlpKv(int, const MlpKvArgs&, gpu::Stream stream) { launch(stream); }
void launchKeyValue(int, const KeyValueArgs&, gpu::Stream stream) { launch(stream); }
void launchCandidate(int, const CandidateArgs&, gpu::Stream stream) { launch(stream); }
void launchReread(int, const RereadArgs&, gpu::Stream stream) { launch(stream); }
void launchValueGraph(int, const ValueGraphArgs&, gpu::Stream stream) { launch(stream); }
void restoreCandidateCodes(CandidateInput, CandidateInputLayout, SegmentLayout, int, gpu::Stream stream) {
  launch(stream);
}
void restoreEventInputs(EventInputColumns, uint8_t*, const int*, const int*, int, int, const uint8_t*,
                        gpu::Stream stream) {
  launch(stream);
}
void launchPageRelation(const RelationArgs&, const int*, gpu::Stream stream) { launch(stream); }
void exchangeEventPages(const int*, const int*, int, int, PageColumns, gpu::Stream stream) { launch(stream); }
void gatherPageFeatures(const uint8_t*, const uint16_t*, int, int, uint8_t*, uint16_t*, const int*, int,
                        gpu::Stream stream) {
  launch(stream);
}
void compactEventRows(const uint16_t*, uint16_t*, const int*, int, int, gpu::Stream stream) { launch(stream); }
void launchProjectedCausalAttention(const AttentionArgs&, int, const int (&)[kAttentionGroups], gpu::Stream stream) {
  launch(stream);
}
void launchProjectedAttention(int, bool, const AttentionArgs&, int, const int (&)[kAttentionGroups],
                              gpu::Stream stream) {
  launch(stream);
}
void launchProjectedQkvFragments(int, const QkvFragmentArgs&, int, gpu::Stream stream) { launch(stream); }
void launchFoldedCausalAttention(const AttentionArgs&, int, const int (&)[kAttentionGroups], gpu::Stream stream) {
  launch(stream);
}

void launchPageAttention(const AttentionArgs&, Span<const int>, int, const int (&)[kAttentionGroups], bool,
                         gpu::Stream stream) {
  launch(stream);
}
void launchPageMlp(int, const MlpArgs&, MlpSources, gpu::Stream stream) { launch(stream); }
void indexLayerInputs(const EmbedArgs&, const EmbeddingSpec (&)[kEmbeddingSlots], ProjectionIndex, gpu::Stream stream) {
  launch(stream);
}
void launchSharedProjection(AttentionArgs, ProjectionIndex, Span<uint16_t>, int, const int (&)[kAttentionGroups],
                            gpu::Stream stream) {
  launch(stream);
}
void launchWideEmbed(const EmbedArgs&, Span<const float>, const Span<uint16_t> (&)[kEmbeddingSlots], ProjectionIndex,
                     gpu::Stream stream) {
  launch(stream);
}
void launchCandidateCore(int, const CandidateArgs&, const CountWeights&, const SegmentWeights&, uint16_t*,
                         gpu::Stream stream, HotCandidateOffsets, Span<uint16_t>) {
  launch(stream);
}
void buildHotOffsets(const uint64_t*, int, int, Span<const float>, Span<const float>, SegmentWeights, float*, float*,
                     gpu::Stream stream) {
  launch(stream);
}
HotCodeClassifier hotCodeClassifier(int) {
  return [](Span<const uint8_t>, int, int, SegmentLayout, const HotEntry*, uint16_t*, gpu::Stream stream) {
    launch(stream);
  };
}
}  // namespace prism
