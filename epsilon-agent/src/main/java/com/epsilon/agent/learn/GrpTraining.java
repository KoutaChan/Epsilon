package com.epsilon.agent.learn;

import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/**
 * GRP の学習の設定。牌譜から GRP を学習するときと、PRISM で GRP を学習し直すときに読む。学習率とエポック数の上限は、ゼロから
 * 学習するときと前のモデルから続けて学習するときで別に持ち、ほかの値は共通。重み減衰は掛けない。キーを変えてよいかどうかは、
 * この設定を持つセクションの決まりに従う。
 */
public record GrpTraining(
    @Key(doc = "1 ステップで使う行の数", value = "4096") @Range(min = 1) int batchRows,
    @Key(doc = "勾配全体のノルムの上限（超えたときだけ縮める）", value = "10.0")
        @Range(min = 0, minInclusive = false)
        double clipNorm,
    @Key(
            doc = "早期終了用の検証の行で、損失がこの回数続けて良くならなければ、学習率を半分にする",
            value = "2")
        @Range(min = 1)
        int halvingPatience,
    @Key(doc = "早期終了用の検証の行で、損失がこの回数続けて良くならなければ、学習を止める（学習率を半分にする回数以上）", value = "4")
        @Range(min = 1)
        int stopPatience,
    @Key(doc = "ゼロから学習するときの学習率", value = "1e-3") @Range(min = 0, minInclusive = false)
        double scratchRate,
    @Key(doc = "ゼロから学習するときのエポック数の上限", value = "16") @Range(min = 1) int scratchEpochs,
    @Key(doc = "前のモデルの重みから続けて学習するときの学習率", value = "3e-4")
        @Range(min = 0, minInclusive = false)
        double warmRate,
    @Key(doc = "前のモデルの重みから続けて学習するときのエポック数の上限", value = "1") @Range(min = 1)
        int warmEpochs) {

  public GrpTraining {
    if (stopPatience < halvingPatience) {
      throw new InvalidSetting(
          "stopPatience", "stopPatience " + stopPatience + " is below halvingPatience " + halvingPatience);
    }
  }
}
