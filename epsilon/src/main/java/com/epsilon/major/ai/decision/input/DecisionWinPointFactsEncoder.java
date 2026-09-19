package com.epsilon.major.ai.decision.input;

import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.calculate.scoring.YakuBits;
import com.epsilon.engine.VisibleHandScoreBuffer;

/** 公開情報だけの和了採点結果を、MajorのPoint Factsへ変換する。 */
final class DecisionWinPointFactsEncoder {

  private DecisionWinPointFactsEncoder() {}

  static void encodeAction(
      VisibleHandScoreBuffer score,
      boolean requiresPaoCorrection,
      DecisionInputWriter writer,
      int actionSlot) {
    writer.actionWinFacts(
        actionSlot,
        score.available() ? 1 : 0,
        score.visibleHan(),
        ScoreMath.encodeFuCode(score.fu()),
        score.yakumanMultiplier(),
        requiresPaoCorrection ? 1 : 0);
  }

  static void encodeWait(
      long yakuBits,
      int hanWithoutUra,
      int fu,
      boolean requiresPaoCorrection,
      DecisionInputSchema.WaitWinType winType,
      DecisionInputWriter writer,
      int actionSlot,
      int transitionSlot,
      int waitSlot) {
    writer.waitWinFacts(
        actionSlot,
        transitionSlot,
        waitSlot,
        winType,
        yakuBits != 0L ? 1 : 0,
        hanWithoutUra,
        ScoreMath.encodeFuCode(fu),
        YakuBits.yakumanCount(yakuBits),
        requiresPaoCorrection ? 1 : 0);
  }
}
