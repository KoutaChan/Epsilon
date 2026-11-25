#pragma once
#include "inference_kernels.h"
#include "candidate_segments.h"

#include "cache/page_columns.h"

namespace prism {

void restoreCandidateCodes(CandidateInput input, CandidateInputLayout layout, SegmentLayout segments, int rows,
                           gpu::Stream stream);

void restoreEventInputs(EventInputColumns columns, uint8_t* pages, const int* slots, const int* missing, int events,
                        int misses, const uint8_t* payload, gpu::Stream stream);

void launchPageRelation(const RelationArgs& args, const int* queryMask, gpu::Stream stream);

void exchangeEventPages(const int* slots, const int* missIndex, int events, int misses, PageColumns columns,
                        gpu::Stream stream);

void gatherPageFeatures(const uint8_t* categories, const uint16_t* scalars, int categoryWidth, int scalarWidth,
                        uint8_t* packedCategories, uint16_t* packedScalars, const int* missIndex, int misses,
                        gpu::Stream stream);

void compactEventRows(const uint16_t* dense, uint16_t* compact, const int* missIndex, int misses, int width,
                      gpu::Stream stream);
}
