package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.prism.input.FrontierField.Reach;

/**
 * 1 向聴の後の手の前線（{@link Reach} ごとの牌のマスク）。エンジンの {@code TenpaiFrontierAnalyzer} と同じ決まりを、エンジンの公開の口
 * （{@link HandShapeAnalyzer}・{@link HandScoreEvaluator#probeAfterAdding}）で作る。割り当てをしない。
 *
 * <p>13 枚相当で最小の向聴が 1 の手だけを探す（向聴は受け入れを集めない {@link HandShapeAnalyzer#calculateMinimum} で先に見る。値は
 * {@link HandShapeAnalyzer#analyzeInto} の最小の向聴と同じ）。受け入れ（自分の副露と手で 5 枚目になる牌を除き、見えない枚数が 0 の牌も除く）を
 * 1 枚ずつ引き、手の牌（引いた牌の種類を除く）を 1 枚ずつ捨てて、聴牌になれば待ち（5 枚目を除く）のどれかで役があるかを調べる。振聴でないロンと
 * ツモが両方見つかったら、その牌の残りの打牌は調べない。
 */
final class TenpaiFrontier {

  private static final int THIRTEEN = 13;
  private static final int FRONTIER_SHANTEN = 1;
  private static final Reach[] REACHES = Reach.values();

  private final HandShapeState cursor = new HandShapeState();
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandShapeAnalysisBuffer shape = new HandShapeAnalysisBuffer();
  private final HandScoreEvaluator scores = new HandScoreEvaluator();
  private final long[] masks = new long[REACHES.length];

  /**
   * 前線を探す。13 枚相当で 1 向聴でなければ、全てのマスクを空にする。
   *
   * @param hand 後の手
   * @param river 自分の河の牌と、この遷移で捨てる牌のマスク
   * @param tiles 牌ごとの枚数（見えない枚数を読む）
   * @param conditions 立直なしの公開の和了の条件
   */
  void analyze(AfterHand hand, long river, TileCounts tiles, WinConditions conditions) {
    long freeRon = 0L;
    long furitenRon = 0L;
    long tsumo = 0L;
    if (hand.effectiveTileCount() == THIRTEEN
        && shapes.calculateMinimum(hand.shape()) == FRONTIER_SHANTEN) {
      shapes.analyzeInto(hand.shape(), shape);
      cursor.load(hand.shape());
      long concealed = hand.concealedTileTypeMask();
      for (long draws = shape.ukeireTileTypeMask() & ShapeAnalysis.possibleDraws(hand.shape(), hand);
          draws != 0L;
          draws &= draws - 1) {
        int draw = Long.numberOfTrailingZeros(draws);
        if (tiles.unseen(draw) == 0) {
          continue;
        }
        boolean free = false;
        boolean furiten = false;
        boolean self = false;
        cursor.add(draw);
        for (long discards = concealed & ~(1L << draw);
            discards != 0L && !(free && self);
            discards &= discards - 1) {
          int discard = Long.numberOfTrailingZeros(discards);
          cursor.remove(discard);
          if (shapes.calculateMinimum(cursor) == 0) {
            long waits = shapes.agariTileTypeMask(cursor) & ShapeAnalysis.possibleDraws(cursor, hand);
            int found = yaku(hand, waits, conditions);
            if ((found & HandScoreEvaluator.RON) != 0) {
              if ((waits & (river | 1L << discard)) != 0L) {
                furiten = true;
              } else {
                free = true;
              }
            }
            self |= (found & HandScoreEvaluator.TSUMO) != 0;
          }
          cursor.add(discard);
        }
        cursor.remove(draw);
        long bit = 1L << draw;
        if (free) {
          freeRon |= bit;
        } else if (furiten) {
          furitenRon |= bit;
        }
        if (self) {
          tsumo |= bit;
        }
      }
    }
    masks[Reach.FREE_RON.ordinal()] = freeRon;
    masks[Reach.FURITEN_RON.ordinal()] = furitenRon;
    masks[Reach.TSUMO.ordinal()] = tsumo;
  }

  /**
   * 探した前線の牌のマスクを返す。
   *
   * @param reach 分け方
   * @return 牌のマスク（物理の番号）
   */
  long mask(Reach reach) {
    return masks[reach.ordinal()];
  }

  /** 聴牌の手（cursor）の待ちのどれかで役がある和了の方法（{@link HandScoreEvaluator#RON}・{@link HandScoreEvaluator#TSUMO} の和）。 */
  private int yaku(AfterHand hand, long waits, WinConditions conditions) {
    int found = 0;
    for (long rest = waits; rest != 0L && found != HandScoreEvaluator.BOTH; rest &= rest - 1) {
      found |=
          scores.probeAfterAdding(
              cursor,
              hand,
              Long.numberOfTrailingZeros(rest),
              HandScoreEvaluator.BOTH & ~found,
              conditions);
    }
    return found;
  }
}
