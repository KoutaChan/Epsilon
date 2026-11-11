package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * 次局の終わり方の区分。OutcomeNetwork の q(o|s) のうち、終わり方の出力に当たる。定数ごとに、ルールに応じた終わり方の数を持つ。
 * 終わり方の番号は、HandOutcomes が区分を宣言の順に並べて 1 回だけ振る。席は次局の親を 0 とする相対席。包と流し満貫はめったに
 * 起きないので入れない（記録の中で起きた回数は RuleCheck が数える）。
 */
public enum OutcomeSection {
  /** ツモ。和了者ごとに分け、相対席の小さい順に並べる。 */
  TSUMO(Variant::seats),
  /** ロン。(和了者, 放銃者) ごとに分ける。和了者の順に並べ、同じ和了者の中では和了者以外の放銃者を相対席の順に並べる。 */
  RON(variant -> variant.seats() * (variant.seats() - 1)),
  /**
   * ダブロン。(放銃者, 2 人の和了者) ごとに分ける。放銃者の順に並べ、同じ放銃者の中では放銃者以外の 2 席の組を辞書順に
   * 並べる。2 人の打点は独立に決まる。
   */
  DOUBLE_RON(variant -> variant.seats() * Seats.pairs(variant.seats() - 1)),
  /** 荒牌。聴牌した席のビット（相対席）ごとに分ける。 */
  DRAW(variant -> 1 << variant.seats()),
  /** 途中流局。どの種類でも点の動きと局の進み方は同じ（四人打ちの三家和もここに入る）。 */
  ABORTIVE(variant -> 1);

  private final ToIntFunction<Variant> count;

  OutcomeSection(ToIntFunction<Variant> count) {
    this.count = count;
  }

  /** この区分の終わり方の数を返す。 */
  public int count(Variant variant) {
    return count.applyAsInt(variant);
  }

  /** 終わり方の数（全ての区分の和）を返す。 */
  public static int total(Variant variant) {
    int total = 0;
    for (OutcomeSection section : values()) {
      total += section.count(variant);
    }
    return total;
  }
}
