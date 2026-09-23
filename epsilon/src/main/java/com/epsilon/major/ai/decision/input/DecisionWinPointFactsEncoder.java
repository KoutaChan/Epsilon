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
        encodeFuCode(score.visibleHan(), score.fu()),
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
        encodeFuCode(hanWithoutUra, fu),
        YakuBits.yakumanCount(yakuBits),
        requiresPaoCorrection ? 1 : 0);
  }

  private static int encodeFuCode(int han, int fu) {
    // 3翻110符ですでに満貫。高符の手は既存コードへまとめても、追加ドラを含め基本点は変わらない。
    // 採点結果の符は保持し、Point Factsだけを正規化して保存済みモデルの入力形式を維持する。
    if (han >= 3 && fu > 110 && fu % 10 == 0) return ScoreMath.encodeFuCode(110);
    return ScoreMath.encodeFuCode(fu);
  }
}
