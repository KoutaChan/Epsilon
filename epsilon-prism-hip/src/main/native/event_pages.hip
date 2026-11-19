#include "prism_hip_event_pages.h"

namespace prism {
namespace {
template <bool Store>
__global__ void pageKernel(const int* slots, const int* missIndex, PageColumns columns) {
  const int row = blockIdx.x;
  const int slot = slots[Store ? missIndex[row] : row];
  for (int c = 0; c < columns.count; ++c) {
    const PageColumn& col = columns.column[c];
    auto* page = static_cast<uint8_t*>(col.pages) + static_cast<size_t>(slot) * col.bytes;
    auto* batch = static_cast<uint8_t*>(col.dense) + static_cast<size_t>(row) * col.bytes;
    auto* from = Store ? batch : page;
    auto* to = Store ? page : batch;
    for (int i = threadIdx.x; i < col.bytes / 16; i += blockDim.x)
      reinterpret_cast<u32x4*>(to)[i] = reinterpret_cast<const u32x4*>(from)[i];
  }
}

__global__ void featureKernel(const uint8_t* categories, const uint16_t* scalars, int categoryWidth, int scalarWidth,
                              uint8_t* packedCategories, uint16_t* packedScalars, const int* missIndex, int misses) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= misses) return;
  int from = missIndex[row];
  for (int c = 0; c < categoryWidth; ++c)
    packedCategories[row * categoryWidth + c] = categories[from * categoryWidth + c];
  for (int c = 0; c < scalarWidth; ++c) packedScalars[row * scalarWidth + c] = scalars[from * scalarWidth + c];
}

template <bool Store>
__global__ void rawEventKernel(EventInputColumns columns, uint8_t* pages, const int* slots, const int* missing,
                               int count, const uint8_t* payload) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int event = Store ? missing[i] : i;
  size_t width = columns.categoryWidth + columns.scalarBytes + columns.relationWidth;
  uint8_t* page = pages + size_t(slots[event]) * width;
  uint8_t* dense[3] = {columns.categories, columns.scalars, columns.relations};
  int widths[3] = {columns.categoryWidth, columns.scalarBytes, columns.relationWidth};
  for (int column = 0; column < 3; column++) {
    int bytes = widths[column];
    uint8_t* row = dense[column] + size_t(event) * bytes;
    if constexpr (Store) {
      const uint8_t* source = payload ? payload + size_t(i) * bytes : row;
      for (int b = 0; b < bytes; b++) page[b] = source[b];
      if (payload) payload += size_t(count) * bytes;
    } else {
      for (int b = 0; b < bytes; b++) row[b] = page[b];
    }
    page += bytes;
  }
}

// 1 ブロックが判断を 1 つ持つ。自席への安全度は判断ごと・牌ごとに求め、全ての遷移に共用する。
__global__ void candidateInputKernel(CandidateInput in, CandidateInputLayout layout, v4::SegmentLayout segments) {
  int row = blockIdx.x;
  int tile = threadIdx.x;
  if (tile >= in.tilesPerRow) return;
  int tileIndex = row * in.tilesPerRow + tile;
  const auto* tileRow = in.tiles + tileIndex * in.tileWidth;
  int physical = in.tileRelations[tileIndex * in.relationWidth + in.relationTile];
  const int* index = in.rowColumns + row * in.rowWidth;
  bool declared = false, after = false, cycle = false;
  for (int i = 0; i < index[in.eventCount]; i++) {
    auto* event = in.events + (index[in.firstEvent] + i) * in.eventWidth;
    if (event[layout.kind] != layout.discardKind) continue;
    if (event[layout.author] == layout.selfSeat) {
      cycle = false;
      declared |= event[layout.declaration] != 0;
    } else if (event[layout.eventTile] == physical) {
      cycle = true;
      after |= declared;
    }
  }
  int self = tileRow[layout.river] ? layout.safeRiver : after ? layout.safeAfter : cycle ? layout.safeCycle : 0;
  for (int i = 0; i < index[in.transitionCount]; i++) {
    int transition = index[in.firstTransition] + i;
    auto* dest = in.codes + transition * segments.codeWidth;
    const auto* basic = in.basicCodes + transition * in.fixed;
    for (int key = tile; key < in.fixed; key += in.tilesPerRow) dest[key] = basic[key];
  }
  for (int s = 0; s < segments.count; ++s) {
    auto slot = layout.slots[s];
    int digit = slot.seat == 0 ? self : tileRow[slot.tileColumn];
    int lower = slot.firstLower * (digit - (tile == slot.bonusTile));
    if (slot.seat >= 0) {
      int riichi = in.seats[(row * in.seatsPerRow + slot.seat) * in.seatWidth + layout.riichi];
      lower += slot.secondLower * (riichi != layout.noRiichi);
    }
    for (int i = 0; i < index[in.transitionCount]; i++) {
      int transition = index[in.firstTransition] + i;
      auto* code = in.codes + transition * segments.codeWidth;
      int base = in.basicCodes[transition * in.fixed + in.fixed - in.tilesPerRow + tile] - layout.firstCode;
      int upper = slot.seat < 0 ? slot.firstUpper * (base / layout.ukeire.weight % layout.ukeire.radix)
                                : slot.firstUpper * (base / layout.wait.weight % layout.wait.radix != 0) +
                                      slot.secondUpper * (base / layout.discarded.weight % layout.discarded.radix);
      code[segments.spans[s].column + tile] = lower + upper;
    }
  }
}

__global__ void compactKernel(const u32x4* dense, u32x4* compact, const int* missIndex, int vectors) {
  const size_t source = static_cast<size_t>(missIndex[blockIdx.x]) * vectors;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x)
    compact[static_cast<size_t>(blockIdx.x) * vectors + i] = dense[source + i];
}
}

void restoreCandidateCodes(CandidateInput input, CandidateInputLayout layout, v4::SegmentLayout segments, int rows,
                           hipStream_t stream) {
  if (!rows) return;
  hipLaunchKernelGGL(candidateInputKernel, dim3(rows), dim3(input.tilesPerRow), 0, stream, input, layout, segments);
  PRISM_HIP(hipGetLastError());
}

void restoreEventInputs(EventInputColumns columns, uint8_t* pages, const int* slots, const int* missing, int events,
                        int misses, const uint8_t* payload, hipStream_t stream) {
  constexpr int threads = 128;
  if (misses) {
    hipLaunchKernelGGL(rawEventKernel<true>, dim3((misses + threads - 1) / threads), dim3(threads), 0, stream, columns,
                       pages, slots, missing, misses, payload);
    PRISM_HIP(hipGetLastError());
  }
  if (payload && events) {
    hipLaunchKernelGGL(rawEventKernel<false>, dim3((events + threads - 1) / threads), dim3(threads), 0, stream, columns,
                       pages, slots, missing, events, nullptr);
    PRISM_HIP(hipGetLastError());
  }
}

void exchangeEventPages(const int* slots, const int* missIndex, int events, int misses, PageColumns columns,
                        hipStream_t stream) {
  if (misses) {
    hipLaunchKernelGGL(pageKernel<true>, dim3(misses), dim3(64), 0, stream, slots, missIndex, columns);
    PRISM_HIP(hipGetLastError());
  }
  if (events) {
    hipLaunchKernelGGL(pageKernel<false>, dim3(events), dim3(64), 0, stream, slots, missIndex, columns);
    PRISM_HIP(hipGetLastError());
  }
}

void gatherPageFeatures(const uint8_t* categories, const uint16_t* scalars, int categoryWidth, int scalarWidth,
                        uint8_t* packedCategories, uint16_t* packedScalars, const int* missIndex, int misses,
                        hipStream_t stream) {
  hipLaunchKernelGGL(featureKernel, dim3((misses + 127) / 128), dim3(128), 0, stream, categories, scalars,
                     categoryWidth, scalarWidth, packedCategories, packedScalars, missIndex, misses);
  PRISM_HIP(hipGetLastError());
}

void compactEventRows(const uint16_t* dense, uint16_t* compact, const int* missIndex, int misses, int width,
                      hipStream_t stream) {
  if (!misses) return;
  hipLaunchKernelGGL(compactKernel, dim3(misses), dim3(32), 0, stream, reinterpret_cast<const u32x4*>(dense),
                     reinterpret_cast<u32x4*>(compact), missIndex, width / 8);
  PRISM_HIP(hipGetLastError());
}
}

// 公開済みの通常版と別の翻訳単位で、必要なクエリだけに関係を書く。
#define PRISM_PAGED_RELATION
#include "k0_relation.hip"
