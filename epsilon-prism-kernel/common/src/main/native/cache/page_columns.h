#pragma once

#include <cstdint>

namespace prism {
// 重みの公開ごとに空にする入力の表。行には全入力のバイトと、その埋め込みと射影を置く。
struct ProjectionCache {
  // この幅を超える入力と衝突した入力は、そのバッチの代表だけで計算する。
  static constexpr int kKeyBytes = 32;
  unsigned long long* tags{};
  uint8_t* keys{};
  Span<uint16_t> residual{}, qkv{};
  unsigned generation{};
  int rows{};
};

// バッチ内の代表、計算するトークン、その検索表、代表の件数を続けて置く。
struct ProjectionIndex {
  int* owners{};
  int tokens{};
  ProjectionCache cache{};
  __host__ __device__ int* sources() const { return owners + tokens; }
  __host__ __device__ int* table() const { return sources() + tokens; }
  __host__ __device__ int* count() const { return table() + tokens; }
  static size_t bytes(size_t tokens) { return (3 * tokens + 1) * sizeof(int); }
};

// GPU ページの列。未計算の行を先に保存し、全行をバッチの順に読み戻す。二つの領域は同じでもよい。
struct PageColumn {
  void* compact;
  void* dense;
  void* pages;
  int bytes;
};

struct PageColumns {
  PageColumn column[kMaxDepth - 1];
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

}  // namespace prism
