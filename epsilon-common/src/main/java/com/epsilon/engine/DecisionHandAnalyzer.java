package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBatchBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.HandView;
import com.epsilon.core.Tile;

/** 判断対象の手牌形と現在の待ちを解析し、呼び出し側のバッファへ結果を書き込む。 */
final class DecisionHandAnalyzer {
  private final ShapeAnalysisCache shapes = new ShapeAnalysisCache();
  private final HandScoreEvaluator scores = new HandScoreEvaluator();
  private final HandScoreBatchBuffer currentWaitScores = new HandScoreBatchBuffer(2).size(2);
  private final HandScoreBatchBuffer intrinsicWaitScores = new HandScoreBatchBuffer(4).size(4);
  private final TenpaiFrontierAnalyzer frontier = new TenpaiFrontierAnalyzer();

  HandScoreEvaluator scoreEvaluator() {
    return scores;
  }

  void analyzeShape(HandView hand, EngineDecisionBuffer context, DecisionHandAnalysisBuffer out) {
    out.clearTransitionValues();
    shapes.analyzeInto(hand, out);
    // 手牌（副露・暗槓を除く）の5枚目は形状解析で除外済み。確定面子がある場合だけ合計所有枚数を調べる。
    if (hand.meldCount() != 0) {
      long possibleDraws = PossibleDrawsLookup.possibleDraws(hand, hand);
      out.improvingTileTypeMask &= possibleDraws;
      out.shapeWaitTileTypeMask &= possibleDraws;
    }
    out.liveImprovingCopies = context.countUnseenCopies(out.improvingTileTypeMask);
    out.liveImprovingTileTypes = context.countLiveTileTypes(out.improvingTileTypeMask);
  }

  void analyzeWaits(
      HandView hand,
      int doraCount,
      RiichiState riichi,
      EngineDecisionBuffer context,
      DecisionHandAnalysisBuffer out) {
    analyzeShape(hand, context, out);
    out.menzen = hand.isMenzen();
    out.meldCount = hand.meldCount();
    out.concealedTileCount = hand.concealedTileCount();
    out.ownedAkaMask = hand.ownedAkaMask();
    out.doraCount = doraCount;
    WinConditions current = context.winContext(riichi);
    boolean compareIntrinsic = riichi != RiichiState.NONE;
    HandScoreBatchBuffer waitScores = compareIntrinsic ? intrinsicWaitScores : currentWaitScores;
    waitScores.configure(0, WinMethod.RON, current, out.ownedAkaMask);
    waitScores.configure(1, WinMethod.TSUMO, current, out.ownedAkaMask);
    if (compareIntrinsic) {
      WinConditions intrinsic = context.winContext(RiichiState.NONE);
      waitScores.configure(2, WinMethod.RON, intrinsic, out.ownedAkaMask);
      waitScores.configure(3, WinMethod.TSUMO, intrinsic, out.ownedAkaMask);
    }
    var ron = waitScores.result(0);
    var tsumo = waitScores.result(1);
    var intrinsicRonScore = compareIntrinsic ? waitScores.result(2) : null;
    var intrinsicTsumoScore = compareIntrinsic ? waitScores.result(3) : null;
    DoraState dora = context.doraState();
    int knownAka =
        context.visibleAkaTileTypeMask()
            | context.state().hand(context.playerIndex()).ownedAkaMask()
            | out.ownedAkaMask;
    long intrinsicRon = 0L;
    long intrinsicTsumo = 0L;
    for (long waits = out.shapeWaitTileTypeMask; waits != 0L; waits &= waits - 1) {
      int tile = Long.numberOfTrailingZeros(waits);
      scores.scoreAfterAddingBatch(hand, hand, tile, dora, false, waitScores);
      out.addWait(
          tile,
          ron.available() ? ron : null,
          tsumo.available() ? tsumo : null,
          Tile.canBeAka(tile) && !AkaTileMask.containsTile(knownAka, tile));
      if (compareIntrinsic) {
        if (intrinsicRonScore.available()) intrinsicRon |= 1L << tile;
        if (intrinsicTsumoScore.available()) intrinsicTsumo |= 1L << tile;
      }
    }
    if (!compareIntrinsic) {
      intrinsicRon = out.ronWaitTileTypeMask;
      intrinsicTsumo = out.tsumoWaitTileTypeMask;
    }
    out.improvingHitWithinOneDraw =
        hitProbability(out.liveImprovingCopies, context.totalUnseenCopies(), 1);
    out.improvingHitWithinTwoDraws =
        hitProbability(out.liveImprovingCopies, context.totalUnseenCopies(), 2);
    out.improvingHitWithinThreeDraws =
        hitProbability(out.liveImprovingCopies, context.totalUnseenCopies(), 3);
    out.liveShapeWaitCopies = context.countUnseenCopies(out.shapeWaitTileTypeMask);
    out.intrinsicRonWaitTileTypes = Long.bitCount(intrinsicRon);
    out.intrinsicRonWaitCopies = context.countUnseenCopies(intrinsicRon);
    out.intrinsicTsumoWaitTileTypes = Long.bitCount(intrinsicTsumo);
    out.intrinsicTsumoWaitCopies = context.countUnseenCopies(intrinsicTsumo);
  }

  void analyzeFrontier(
      HandView hand, long ownRiver, EngineDecisionBuffer context, DecisionHandAnalysisBuffer out) {
    frontier.analyze(
        hand,
        ownRiver,
        context,
        out.minimumShanten,
        out.improvingTileTypeMask,
        out.frontier,
        scores);
    out.furitenFreeRonFrontierCopies = context.countUnseenCopies(out.frontier.furitenFreeRon());
    out.furitenOnlyRonFrontierCopies = context.countUnseenCopies(out.frontier.furitenOnlyRon());
    out.tsumoFrontierCopies = context.countUnseenCopies(out.frontier.tsumo());
  }

  private static float hitProbability(int targets, int total, int draws) {
    if (targets == 0 || total == 0) return 0.0f;
    double miss = 1.0;
    for (int draw = 0, count = Math.min(draws, total); draw < count; draw++) {
      int remaining = total - targets - draw;
      if (remaining <= 0) return 1.0f;
      miss *= remaining / (double) (total - draw);
    }
    return (float) (1.0 - miss);
  }
}
