#pragma once
#include "inference_kernels.h"
#include "inference/candidate_segments.h"
#include "training/backbone_tensors.h"
#include "training/candidate_tensors.h"

namespace prism {

// 形を確かめる。範囲外なら std::invalid_argument を投げる。
void checkRelationCodeShape(const RelationCodeShape& shape);

// 関係のコードを作る。出力先の先頭は 4 バイト境界にそろえる。
void relationCodes(const RelationCodeShape& shape, const RelationCodeArrays& arrays, gpu::Stream stream);

// 形を確かめる。範囲外なら std::invalid_argument を投げる。
void checkAttentionShape(const AttentionShape& shape);

void attentionForward(const AttentionShape& shape, const AttentionForward& arrays, gpu::Stream stream);

void attentionBackward(const AttentionShape& shape, const AttentionBackward& arrays, gpu::Stream stream);

// 形を確かめる。範囲外なら std::invalid_argument を投げる。
void checkNormShape(const NormShape& shape);

void normForward(const NormShape& shape, const NormForward& arrays, gpu::Stream stream);

void normBackward(const NormShape& shape, const NormBackward& arrays, gpu::Stream stream);

void causalAttentionForward(const AttentionShape& shape, int fixed, const AttentionForward& arrays, gpu::Stream stream);
void causalAttentionBackward(const AttentionShape& shape, int fixed, const AttentionBackward& arrays,
                             gpu::Stream stream);

}  // namespace prism

namespace prism {

void launchCountTraining(CountTraining shape, const uint16_t* projected, const float* gates, const int32_t* rows,
                         const int32_t* codes, uint16_t* output, gpu::Stream stream);
void launchCountTrainingBackward(CountTraining shape, const uint16_t* projected, const float* gates,
                                 const int32_t* rows, const int32_t* codes, const uint16_t* gradOutput,
                                 uint16_t* gradProjected, float* gradGate, gpu::Stream stream);

void launchSegmentAttention(SegmentAttention shape, SegmentAttentionArrays arrays, float* output, float* probs,
                            gpu::Stream stream);
void launchSegmentAttentionBackward(SegmentAttention shape, SegmentAttentionArrays arrays,
                                    SegmentAttentionBackward gradients, gpu::Stream stream);

}  // namespace prism
