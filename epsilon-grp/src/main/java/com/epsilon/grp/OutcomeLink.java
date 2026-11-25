package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * OutcomeNetwork が、終わり方を与えた上での打点と立直の確率に足すロジットのまとまり。形（OutcomeShape）が持つものだけを、
 * 終わり方・打点・「立直した」のロジットの後ろに宣言の順で並べる。席は次局の親を 0 とする相対席。
 */
public enum OutcomeLink {
  /**
   * 和了者が精算の後に就く順位ごとのロジット。並びは [(和了者の相対席 × 2 ＋ 和了の形) × S ＋ 順位]。打点のロジットに、その打点で
   * 和了したときに就く順位のロジットを足してから softmax を取る。順位は WinRanks が決める。
   */
  RANK(variant -> variant.seats() * WinMethod.values().length * variant.seats()),
  /**
   * 和了者ごとのロジット。並びは [(相対席 × 立場の数 ＋ 立場) × S ＋ 和了者の相対席]。和了者でない席の「立直した」のロジットに、
   * その席の立場と和了者（ダブロンなら 2 人とも）のロジットを足す。
   */
  WINNER(variant -> variant.seats() * RiichiRole.values().length * variant.seats());

  private final ToIntFunction<Variant> width;

  OutcomeLink(ToIntFunction<Variant> width) {
    this.width = width;
  }

  /** そのルールでのロジットの数を返す。 */
  public int width(Variant variant) {
    return width.applyAsInt(variant);
  }
}
