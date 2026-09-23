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

/** 判断対象の手牌形と待ちを解析し、再利用バッファへ結果を書き込む。 */
final class HandAnalyzer {

  private static final int RON_SLOT = 0;
  private static final int TSUMO_SLOT = 1;
  private static final int RON_WITHOUT_RIICHI_SLOT = 2;
  private static final int TSUMO_WITHOUT_RIICHI_SLOT = 3;

  private final ShapeAnalysisCache shapeCache = new ShapeAnalysisCache();
  private final HandScoreEvaluator scoreEvaluator = new HandScoreEvaluator();
  private final HandScoreBatchBuffer waitScoreBatch = new HandScoreBatchBuffer(4);
  private final TenpaiFrontierAnalyzer frontierAnalyzer = new TenpaiFrontierAnalyzer();

  HandScoreEvaluator scoreEvaluator() {
    return scoreEvaluator;
  }

  void analyzeShape(HandView hand, EngineDecisionBuffer context, HandAnalysisBuffer output) {
    output.clear();
    shapeCache.analyzeInto(hand, output);
    // 手牌（副露・暗槓を除く）の5枚目は形状解析で除外済み。確定面子がある場合だけ合計所有枚数を調べる。
    if (hand.meldCount() != 0) {
      long possibleDraws = PossibleDrawsLookup.possibleDraws(hand, hand);
      output.improvingTileTypeMask &= possibleDraws;
      output.shapeWaitTileTypeMask &= possibleDraws;
    }
    output.liveImprovingCopies = context.countUnseenCopies(output.improvingTileTypeMask);
    output.liveImprovingTileTypes = context.countLiveTileTypes(output.improvingTileTypeMask);
  }

  void analyzeWaits(
      HandView hand,
      int doraCount,
      RiichiState riichi,
      EngineDecisionBuffer context,
      HandAnalysisBuffer output) {
    analyzeShape(hand, context, output);
    output.menzen = hand.isMenzen();
    output.meldCount = hand.meldCount();
    output.concealedTileCount = hand.concealedTileCount();
    output.ownedAkaMask = hand.ownedAkaMask();
    output.doraCount = doraCount;

    boolean compareWithoutRiichi = riichi != RiichiState.NONE;
    configureWaitScoreBatch(context, riichi, output.ownedAkaMask, compareWithoutRiichi);
    var ronScore = waitScoreBatch.result(RON_SLOT);
    var tsumoScore = waitScoreBatch.result(TSUMO_SLOT);
    var ronScoreWithoutRiichi =
        compareWithoutRiichi ? waitScoreBatch.result(RON_WITHOUT_RIICHI_SLOT) : ronScore;
    var tsumoScoreWithoutRiichi =
        compareWithoutRiichi ? waitScoreBatch.result(TSUMO_WITHOUT_RIICHI_SLOT) : tsumoScore;

    DoraState doraState = context.doraState();
    int knownAkaTileTypeMask =
        context.visibleAkaTileTypeMask()
            | context.state().hand(context.playerIndex()).ownedAkaMask()
            | output.ownedAkaMask;
    long ronWaitTileTypeMaskWithoutRiichi = 0L;
    long tsumoWaitTileTypeMaskWithoutRiichi = 0L;
    for (long waits = output.shapeWaitTileTypeMask; waits != 0L; waits &= waits - 1) {
      int tileType = Long.numberOfTrailingZeros(waits);
      scoreEvaluator.scoreAfterAddingBatch(hand, hand, tileType, doraState, false, waitScoreBatch);
      boolean ronAvailable = ronScore.available();
      boolean tsumoAvailable = tsumoScore.available();
      output.appendWait(
          tileType,
          ronAvailable ? ronScore : null,
          tsumoAvailable ? tsumoScore : null,
          Tile.canBeAka(tileType) && !AkaTileMask.containsTile(knownAkaTileTypeMask, tileType),
          ronAvailable && PaoRules.applies(hand, ronScore),
          tsumoAvailable && PaoRules.applies(hand, tsumoScore));
      if (ronScoreWithoutRiichi.available()) {
        ronWaitTileTypeMaskWithoutRiichi |= 1L << tileType;
      }
      if (tsumoScoreWithoutRiichi.available()) {
        tsumoWaitTileTypeMaskWithoutRiichi |= 1L << tileType;
      }
    }
    finishWaitStatistics(
        output, context, ronWaitTileTypeMaskWithoutRiichi, tsumoWaitTileTypeMaskWithoutRiichi);
  }

  private void configureWaitScoreBatch(
      EngineDecisionBuffer context,
      RiichiState riichi,
      int ownedAkaMask,
      boolean compareWithoutRiichi) {
    WinConditions conditions = context.winContext(riichi);
    waitScoreBatch
        .size(compareWithoutRiichi ? 4 : 2)
        .configure(RON_SLOT, WinMethod.RON, conditions, ownedAkaMask)
        .configure(TSUMO_SLOT, WinMethod.TSUMO, conditions, ownedAkaMask);
    if (compareWithoutRiichi) {
      WinConditions conditionsWithoutRiichi = context.winContext(RiichiState.NONE);
      waitScoreBatch
          .configure(RON_WITHOUT_RIICHI_SLOT, WinMethod.RON, conditionsWithoutRiichi, ownedAkaMask)
          .configure(
              TSUMO_WITHOUT_RIICHI_SLOT, WinMethod.TSUMO, conditionsWithoutRiichi, ownedAkaMask);
    }
  }

  private static void finishWaitStatistics(
      HandAnalysisBuffer output,
      EngineDecisionBuffer context,
      long ronWaitTileTypeMaskWithoutRiichi,
      long tsumoWaitTileTypeMaskWithoutRiichi) {
    output.improvingHitWithinOneDraw =
        drawHitProbability(output.liveImprovingCopies, context.totalUnseenCopies(), 1);
    output.improvingHitWithinTwoDraws =
        drawHitProbability(output.liveImprovingCopies, context.totalUnseenCopies(), 2);
    output.improvingHitWithinThreeDraws =
        drawHitProbability(output.liveImprovingCopies, context.totalUnseenCopies(), 3);
    output.liveShapeWaitCopies = context.countUnseenCopies(output.shapeWaitTileTypeMask);
    output.ronWaitTileTypesWithoutRiichi = Long.bitCount(ronWaitTileTypeMaskWithoutRiichi);
    output.ronWaitCopiesWithoutRiichi = context.countUnseenCopies(ronWaitTileTypeMaskWithoutRiichi);
    output.tsumoWaitTileTypesWithoutRiichi = Long.bitCount(tsumoWaitTileTypeMaskWithoutRiichi);
    output.tsumoWaitCopiesWithoutRiichi =
        context.countUnseenCopies(tsumoWaitTileTypeMaskWithoutRiichi);
  }

  void analyzeFrontier(
      HandView hand, long ownRiver, EngineDecisionBuffer context, HandAnalysisBuffer output) {
    frontierAnalyzer.analyze(
        hand,
        ownRiver,
        context,
        output.minimumShanten,
        output.improvingTileTypeMask,
        output.tenpaiFrontier,
        scoreEvaluator);
    output.furitenFreeRonFrontierCopies =
        context.countUnseenCopies(output.tenpaiFrontier.furitenFreeRon());
    output.furitenOnlyRonFrontierCopies =
        context.countUnseenCopies(output.tenpaiFrontier.furitenOnlyRon());
    output.tsumoFrontierCopies = context.countUnseenCopies(output.tenpaiFrontier.tsumo());
  }

  private static float drawHitProbability(int targets, int total, int draws) {
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
