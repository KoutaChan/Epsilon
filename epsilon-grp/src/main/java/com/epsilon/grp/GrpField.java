package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * GRP の入力のフィールド。定数ごとに幅を持ち、位置は GrpLayout が宣言の順に 1 回だけ決める。席は次局の親を 0 とする相対席。
 * 点のフィールドの単位は 10,000 点。順位は点の高い順で、同点なら起家に近い席が上（Placement）。
 */
public enum GrpField {
  /** 次局の one-hot。東 1 局を 0 として、西場の最後の局まで並ぶ（四人打ち 12、三人打ち 9）。 */
  ROUND(variant -> variant.finalRound() + 1),
  /** 本場の数に、ロンでの 1 本場あたりの点を掛けた値。 */
  HONBA(variant -> 1),
  /** 供託の点。 */
  KYOTAKU(variant -> 1),
  /** 席ごとの持ち点から配給点を引いた値。 */
  SCORE(Variant::seats),
  /** 席ごとの今の順位の one-hot。並びは [席 × S ＋ 順位]。 */
  RANK(variant -> variant.seats() * variant.seats()),
  /** 2 つの順位 a < b の組ごと（辞書順）に、順位 a の席の点から順位 b の席の点を引いた値。 */
  GAP(variant -> Seats.pairs(variant.seats()));

  private final ToIntFunction<Variant> width;

  GrpField(ToIntFunction<Variant> width) {
    this.width = width;
  }

  /** そのルールでの幅を返す。 */
  public int width(Variant variant) {
    return width.applyAsInt(variant);
  }
}
