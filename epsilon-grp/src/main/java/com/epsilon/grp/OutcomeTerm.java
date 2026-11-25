package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * q(o|s) の対数確率の項のまとまり。定数ごとに、ルールに応じた幅を持つ。位置は HandOutcomes が宣言の順に 1 回だけ決める。
 * 1 つの結果の log q は、次の項を足したもの: 終わり方の 1 項、和了者ごとに打点の 1 項、立直の有無を確率に入れる席ごとに
 * 「立直した」か「立直していない」の 1 項。打点と立直の項は終わり方ごとに持つので、終わり方を与えた上での確率になる。
 */
public enum OutcomeTerm {
  /** 終わり方（HandOutcomes の番号）の log softmax。 */
  MAIN(OutcomeSection::total),
  /**
   * 打点の log softmax。和了者が 1 人の終わり方（ツモとロン。HandOutcomes の番号の順）ごとに、打点を BaseClass の宣言の順に
   * 並べる。並びは [終わり方 × 打点の種類の数 ＋ 打点]。ダブロンの和了者の打点は、その和了者と放銃者のロンの項で表す。
   */
  GRADE(
      variant ->
          (OutcomeSection.TSUMO.count(variant) + OutcomeSection.RON.count(variant))
              * BaseClass.values().length),
  /** 「立直した」の項 log σ(z)。並びは [終わり方 × S ＋ 相対席]。立直の有無を確率に入れない席の位置は使わない。 */
  RIICHI(variant -> OutcomeSection.total(variant) * variant.seats()),
  /** 「立直していない」の項 log σ(−z)。並びは「立直した」と同じ。 */
  NO_RIICHI(variant -> OutcomeSection.total(variant) * variant.seats());

  private final ToIntFunction<Variant> width;

  OutcomeTerm(ToIntFunction<Variant> width) {
    this.width = width;
  }

  public int width(Variant variant) {
    return width.applyAsInt(variant);
  }
}
