package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/** 包が成立する和了と責任払いの席を判定する。 */
final class PaoRules {

  private PaoRules() {}

  static boolean applies(HandView winnerHand, HandScoreBuffer score) {
    return liableSeatOffset(winnerHand, score) >= 0;
  }

  static int liableSeat(int winnerSeat, HandView winnerHand, HandScoreBuffer score) {
    int offset = liableSeatOffset(winnerHand, score);
    return offset < 0 ? -1 : (winnerSeat + offset) % GameState.NUM_PLAYERS;
  }

  private static int liableSeatOffset(HandView winnerHand, HandScoreBuffer score) {
    if (score.hasYakuman(ScoringYaku.DAISANGEN)) {
      int offset = liableSeatOffset(winnerHand, ScoringYaku.DAISANGEN);
      if (offset >= 0) {
        return offset;
      }
    }
    if (score.hasYakuman(ScoringYaku.DAISUUSHII)) {
      return liableSeatOffset(winnerHand, ScoringYaku.DAISUUSHII);
    }
    return -1;
  }

  private static int liableSeatOffset(HandView winnerHand, ScoringYaku yakuman) {
    int requiredMelds = yakuman == ScoringYaku.DAISANGEN ? 3 : 4;
    int found = 0;
    for (int meldIndex = 0; meldIndex < winnerHand.meldCount(); meldIndex++) {
      Meld meld = winnerHand.meld(meldIndex);
      int tile = meld.baseTileType();
      boolean relevant = yakuman == ScoringYaku.DAISANGEN ? Tile.isDragon(tile) : Tile.isWind(tile);
      if (!relevant || ++found < requiredMelds) {
        continue;
      }
      return meld.preservesMenzen() ? -1 : meld.relativeSource().playerOffset();
    }
    return -1;
  }
}
