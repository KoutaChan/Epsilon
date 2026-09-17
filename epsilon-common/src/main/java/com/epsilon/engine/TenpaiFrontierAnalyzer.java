package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeCursor;
import com.epsilon.core.HandView;

/** 一向聴の受け入れ牌だけを探索し、役あり聴牌への到達を集計する。 */
final class TenpaiFrontierAnalyzer {
  private final HandShapeCursor cursor = new HandShapeCursor();
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();

  void analyze(
      HandView hand,
      long ownRiver,
      EngineDecisionBuffer context,
      int shanten,
      long improvingTiles,
      TenpaiFrontierBuffer out,
      HandScoreEvaluator scores) {
    out.clear();
    if (hand.concealedTileCount() + 3 * hand.meldCount() != 13 || shanten != 1) return;
    cursor.load(hand);
    WinConditions conditions = context.winContext(RiichiState.NONE);
    for (long draws = improvingTiles & PossibleDrawsLookup.possibleDraws(hand, hand);
        draws != 0L;
        draws &= draws - 1) {
      int draw = Long.numberOfTrailingZeros(draws);
      if (context.unseenCopies(draw) == 0) continue;
      boolean freeRon = false, furitenRon = false, tsumo = false;
      cursor.add(draw);
      try {
        for (long discards = hand.concealedTileTypeMask() & ~(1L << draw);
            discards != 0L && !(freeRon && tsumo);
            discards &= discards - 1) {
          int discard = Long.numberOfTrailingZeros(discards);
          cursor.remove(discard);
          try {
            if (shapes.calculateMinimum(cursor) != 0) continue;
            long waits =
                shapes.agariTileTypeMask(cursor) & PossibleDrawsLookup.possibleDraws(cursor, hand);
            boolean furiten = (waits & (ownRiver | (1L << discard))) != 0L;
            int found = 0;
            for (long remaining = waits;
                remaining != 0L && found != 3;
                remaining &= remaining - 1) {
              int tile = Long.numberOfTrailingZeros(remaining);
              found |= scores.probeAfterAdding(cursor, hand, tile, 3 & ~found, conditions);
            }
            if ((found & 1) != 0) {
              if (furiten) furitenRon = true;
              else freeRon = true;
            }
            tsumo |= (found & 2) != 0;
          } finally {
            cursor.add(discard);
          }
        }
      } finally {
        cursor.remove(draw);
      }
      out.add(draw, freeRon, furitenRon, tsumo);
    }
  }
}
