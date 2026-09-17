package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.DoraState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.core.Tile;

/** 判断中に固定される牌種別の未確認枚数・表ドラ・赤牌公開情報を一度だけ索引化する。 */
final class DecisionTileIndex {

  private final byte[] visibleCopies = new byte[Tile.NUM_TILE_TYPES];
  private final int[] unseenCopies = new int[Tile.NUM_TILE_TYPES];
  private DoraState doraState;
  private int totalUnseenCopies;
  private int visibleAkaTileTypeMask;
  private int currentHandDoraCount;
  private long unseenFirstCopyTileTypeMask;
  private long unseenSecondCopyTileTypeMask;
  private long unseenThirdCopyTileTypeMask;
  private long unseenFourthCopyTileTypeMask;

  void load(HandView hand, RoundPublicStateIndex publicState, DoraState doraState) {
    this.doraState = doraState;
    totalUnseenCopies = 0;
    currentHandDoraCount = 0;
    unseenFirstCopyTileTypeMask = 0L;
    unseenSecondCopyTileTypeMask = 0L;
    unseenThirdCopyTileTypeMask = 0L;
    unseenFourthCopyTileTypeMask = 0L;
    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      int handCount = hand.count(tileType);
      int visible = publicState.visibleTileCount(tileType);
      visibleCopies[tileType] = (byte) visible;
      int unseen = Tile.TILES_PER_TYPE - visible - handCount;
      unseenCopies[tileType] = unseen;
      totalUnseenCopies += unseen;
      long tileTypeBit = 1L << tileType;
      unseenFirstCopyTileTypeMask |= unseen > 0 ? tileTypeBit : 0L;
      unseenSecondCopyTileTypeMask |= unseen > 1 ? tileTypeBit : 0L;
      unseenThirdCopyTileTypeMask |= unseen > 2 ? tileTypeBit : 0L;
      unseenFourthCopyTileTypeMask |= unseen > 3 ? tileTypeBit : 0L;

      int multiplicity = doraState.doraMultiplicity(tileType);
      currentHandDoraCount += handCount * multiplicity;
    }
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
        currentHandDoraCount += doraMultiplicity(meld.tileAt(tileIndex));
      }
    }

    visibleAkaTileTypeMask =
        publicState.visibleDiscardOrMeldAkaMask() | doraState.visibleIndicatorAkaMask();
  }

  int unseenCopies(int tileType) {
    return unseenCopies[tileType];
  }

  int visibleCopies(int tileType) {
    return visibleCopies[tileType];
  }

  int[] unseenCopies() {
    return unseenCopies;
  }

  int totalUnseenCopies() {
    return totalUnseenCopies;
  }

  int countUnseenCopies(long tileTypeMask) {
    return Long.bitCount(tileTypeMask & unseenFirstCopyTileTypeMask)
        + Long.bitCount(tileTypeMask & unseenSecondCopyTileTypeMask)
        + Long.bitCount(tileTypeMask & unseenThirdCopyTileTypeMask)
        + Long.bitCount(tileTypeMask & unseenFourthCopyTileTypeMask);
  }

  int countLiveTileTypes(long tileTypeMask) {
    return Long.bitCount(tileTypeMask & unseenFirstCopyTileTypeMask);
  }

  int visibleAkaTileTypeMask() {
    return visibleAkaTileTypeMask;
  }

  int currentHandDoraCount() {
    return currentHandDoraCount;
  }

  int doraCountAfter(Action action) {
    return currentHandDoraCount
        + switch (action.type()) {
          case DAHAI, RIICHI_DAHAI -> -doraMultiplicity(action.tileType());
          case CHI, PON, DAIMINKAN -> doraMultiplicity(action.tileType());
          default -> 0;
        };
  }

  int doraCountAfterDiscard(int count, Action discard) {
    return count - doraMultiplicity(discard.tileType());
  }

  int doraMultiplicity(int tileType) {
    return doraState.doraMultiplicity(tileType);
  }

  int doraIndicatorMultiplicity(int tileType) {
    return doraState.indicatorMultiplicity(tileType);
  }

  DoraState doraState() {
    return doraState;
  }
}
