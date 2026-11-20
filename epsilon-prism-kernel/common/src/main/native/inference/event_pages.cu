#include "event_submission.h"

#include "cache/event_pages.cuh"

namespace prism {

void restoreCandidateCodes(CandidateInput input, CandidateInputLayout layout, SegmentLayout segments, int rows,
                           gpu::Stream stream) {
  if (!rows) return;
  candidateInputKernel<<<dim3(rows), dim3(input.tilesPerRow), 0, stream>>>(input, layout, segments);
  PRISM_GPU(gpu::lastError());
}

void restoreEventInputs(EventInputColumns columns, uint8_t* pages, const int* slots, const int* missing, int events,
                        int misses, const uint8_t* payload, gpu::Stream stream) {
  constexpr int threads = 128;
  if (misses) {
    rawEventKernel<true><<<dim3((misses + threads - 1) / threads), dim3(threads), 0, stream>>>(
        columns, pages, slots, missing, misses, payload);
    PRISM_GPU(gpu::lastError());
  }
  if (payload && events) {
    rawEventKernel<false><<<dim3((events + threads - 1) / threads), dim3(threads), 0, stream>>>(
        columns, pages, slots, missing, events, nullptr);
    PRISM_GPU(gpu::lastError());
  }
}

void exchangeEventPages(const int* slots, const int* missIndex, int events, int misses, PageColumns columns,
                        gpu::Stream stream) {
  if (misses) {
    pageKernel<true><<<dim3(misses), dim3(64), 0, stream>>>(slots, missIndex, columns);
    PRISM_GPU(gpu::lastError());
  }
  if (events) {
    pageKernel<false><<<dim3(events), dim3(64), 0, stream>>>(slots, missIndex, columns);
    PRISM_GPU(gpu::lastError());
  }
}

void gatherPageFeatures(const uint8_t* categories, const uint16_t* scalars, int categoryWidth, int scalarWidth,
                        uint8_t* packedCategories, uint16_t* packedScalars, const int* missIndex, int misses,
                        gpu::Stream stream) {
  if (!misses) return;
  featureKernel<<<dim3((misses + 127) / 128), dim3(128), 0, stream>>>(
      categories, scalars, categoryWidth, scalarWidth, packedCategories, packedScalars, missIndex, misses);
  PRISM_GPU(gpu::lastError());
}

void compactEventRows(const uint16_t* dense, uint16_t* compact, const int* missIndex, int misses, int width,
                      gpu::Stream stream) {
  if (!misses) return;
  compactKernel<<<dim3(misses), dim3(32), 0, stream>>>(reinterpret_cast<const u32x4*>(dense),
                                                       reinterpret_cast<u32x4*>(compact), missIndex, width / 8);
  PRISM_GPU(gpu::lastError());
}
}
