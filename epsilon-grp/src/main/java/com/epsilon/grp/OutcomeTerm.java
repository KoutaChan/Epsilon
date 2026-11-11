package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * q(o|s) の対数確率の項のまとまり。定数ごとに、ルールに応じた幅を持つ。位置は HandOutcomes が宣言の順に 1 回だけ決める。
 * 1 つの結果の log q は、次の項を足したもの: 終わり方の 1 項、和了者ごとに打点の 1 項、立直の有無を確率に入れる席ごとに
 * 「立直した」か「立直していない」の 1 項。ネットワークのロジットは、先頭の 3 つのまとまりと同じ位置に並ぶ。「立直していない」の
 * 項は、「立直した」のロジット z から log σ(−z) で求める。
 */
public enum OutcomeTerm {
  /** 終わり方（HandOutcomes の番号）の log softmax。 */
  MAIN(OutcomeSection::total),
  /**
   * 打点の log softmax。和了者の相対席と和了の形の組（[相対席 × 2 ＋ 和了の形]）ごとに、打点を BaseClass の宣言の順に並べる。
   * 並びは [組 × 打点の種類の数 ＋ 打点]。
   */
  GRADE(variant -> variant.seats() * WinMethod.values().length * BaseClass.values().length),
  /** 「立直した」の項 log σ(z)。並びは [相対席 × 立場の数 ＋ 立場]。 */
  RIICHI(variant -> variant.seats() * RiichiRole.values().length),
  /** 「立直していない」の項 log σ(−z)。並びは「立直した」と同じ。 */
  NO_RIICHI(variant -> variant.seats() * RiichiRole.values().length);

  private final ToIntFunction<Variant> width;

  OutcomeTerm(ToIntFunction<Variant> width) {
    this.width = width;
  }

  public int width(Variant variant) {
    return width.applyAsInt(variant);
  }
}
