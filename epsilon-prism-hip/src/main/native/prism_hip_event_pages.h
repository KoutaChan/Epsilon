#pragma once
#include "prism_hip_kernels.h"
#include "prism_hip_common.h"
#include "prism_hip_candidate_layout.h"

namespace prism {

// GPU ページの列。未計算の行を先に保存し、全行をバッチの順に読み戻す。二つの領域は同じでもよい。
struct PageColumn {
  void* dense;
  void* pages;
  int bytes;
};

struct PageColumns {
  PageColumn column[2];
  int count;
};

// イベントの各領域の先頭。スカラーも BF16 のビット列のまま写す。
struct EventInputColumns {
  uint8_t* categories;
  uint8_t* scalars;
  uint8_t* relations;
  int categoryWidth, scalarBytes, relationWidth;
};

// 入力形式が決めた列と混合基数。重みの pack や既存の候補カーネルの引数には含めない。
struct CandidateInputLayout {
  struct Digit {
    int weight, radix;
  };
  struct Slot {
    int seat, bonusTile, tileColumn, firstUpper, secondUpper, firstLower, secondLower;
  };
  int kind, author, eventTile, declaration, discardKind, selfSeat, river, riichi, noRiichi;
  int safeRiver, safeAfter, safeCycle, firstCode;
  Digit ukeire, wait, discarded;
  Slot slots[kMaxSegmentSpans];
};

struct CandidateInput {
  const int* rowColumns;
  const uint8_t *events, *tiles, *seats, *tileRelations, *basicCodes;
  uint8_t* codes;
  int rowWidth, firstEvent, eventCount, firstTransition, transitionCount;
  int eventWidth, tileWidth, seatWidth, relationWidth, relationTile;
  int tilesPerRow, seatsPerRow, fixed;
};

void restoreCandidateCodes(CandidateInput input, CandidateInputLayout layout, v4::SegmentLayout segments, int rows,
                           hipStream_t stream);

void restoreEventInputs(EventInputColumns columns, uint8_t* pages, const int* slots, const int* missing, int events,
                        int misses, const uint8_t* payload, hipStream_t stream);

void launchPageRelation(const RelationArgs& args, const int* queryMask, hipStream_t stream);

void exchangeEventPages(const int* slots, const int* missIndex, int events, int misses, PageColumns columns,
                        hipStream_t stream);

void gatherPageFeatures(const uint8_t* categories, const uint16_t* scalars, int categoryWidth, int scalarWidth,
                        uint8_t* packedCategories, uint16_t* packedScalars, const int* missIndex, int misses,
                        hipStream_t stream);

void compactEventRows(const uint16_t* dense, uint16_t* compact, const int* missIndex, int misses, int width,
                      hipStream_t stream);
}
