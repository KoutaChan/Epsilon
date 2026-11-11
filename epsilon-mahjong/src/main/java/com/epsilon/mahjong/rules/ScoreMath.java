package com.epsilon.mahjong.rules;

/**
 * 点数の計算式（天鳳）。帳簿と精算はどちらもこの式だけを使い、計算は int だけで行う。b は基本点（BaseClass の値）。払いは
 * 100 点単位に切り上げる。切り上げ満貫は無い。
 */
public final class ScoreMath {

  /** 満貫以上の点数の区分。翻の多い順に並び、それぞれその区分になる最小の翻を持つ。13 翻以上は数え役満。 */
  private enum Limit {
    KAZOE(13, BaseClass.YAKUMAN),
    SANBAIMAN(11, BaseClass.SANBAIMAN),
    BAIMAN(8, BaseClass.BAIMAN),
    HANEMAN(6, BaseClass.HANEMAN),
    MANGAN(5, BaseClass.MANGAN);

    private final int han;
    private final int base;

    Limit(int han, BaseClass base) {
      this.han = han;
      this.base = base.base();
    }
  }

  private static final Limit[] LIMITS = Limit.values();

  /** 払いを切り上げる単位。 */
  private static final int UNIT = 100;

  /** 翻と符で点を数える和了の、符の最小値（平和のツモ）。 */
  private static final int MIN_FU = 20;

  /** ロンで払う額の倍率（子の和了は 4b、親の和了は 6b）。 */
  private static final int CHILD_RON = 4;

  private static final int DEALER_RON = 6;

  /** ツモで 1 人が払う額の倍率。親が払う分と、親の和了で各席が払う分は 2b。子の和了で子が払う分は b。 */
  private static final int DEALER_SHARE = 2;

  private static final int CHILD_SHARE = 1;

  private static final int MANGAN = BaseClass.MANGAN.base();

  private static final int YAKUMAN = BaseClass.YAKUMAN.base();

  private ScoreMath() {}

  /**
   * 基本点を求める。役満の倍率が 1 以上なら 8000 × 倍率で、翻と符は見ない。倍率が 0 なら 符 × 2^(2 ＋ 翻) を満貫で頭打ちにし、
   * 5 翻以上はその区分の値にする。倍率が負のとき、または倍率 0 で翻が 1 未満か符が 20 未満のときは IllegalArgumentException。
   */
  public static int base(int han, int fu, int yakumanMultiplier) {
    if (yakumanMultiplier > 0) {
      return YAKUMAN * yakumanMultiplier;
    }
    if (yakumanMultiplier < 0 || han < 1 || fu < MIN_FU) {
      throw new IllegalArgumentException(
          "No base points for han " + han + ", fu " + fu + " and yakuman multiplier " + yakumanMultiplier);
    }
    for (Limit limit : LIMITS) {
      if (han >= limit.han) {
        return limit.base;
      }
    }
    return Math.min(MANGAN, fu << (2 + han));
  }

  /** ロンで放銃者が払う額（本場を除く）。和了者が親なら 6b、子なら 4b。 */
  public static int ron(int base, boolean winnerIsDealer) {
    requireBase(base);
    return roundUp(base * (winnerIsDealer ? DEALER_RON : CHILD_RON));
  }

  /**
   * ツモで 1 人が払う額（本場を除く）。親の和了なら 2b、子の和了なら親が 2b・子が b。和了者と払う人がどちらも親なら
   * IllegalArgumentException。
   */
  public static int tsumo(int base, boolean winnerIsDealer, boolean payerIsDealer) {
    requireBase(base);
    if (winnerIsDealer && payerIsDealer) {
      throw new IllegalArgumentException("The dealer cannot pay for its own tsumo");
    }
    return roundUp(base * (winnerIsDealer || payerIsDealer ? DEALER_SHARE : CHILD_SHARE));
  }

  private static int roundUp(int points) {
    return (points + UNIT - 1) / UNIT * UNIT;
  }

  private static void requireBase(int base) {
    if (base <= 0) {
      throw new IllegalArgumentException("Base points must be positive: " + base);
    }
  }
}
