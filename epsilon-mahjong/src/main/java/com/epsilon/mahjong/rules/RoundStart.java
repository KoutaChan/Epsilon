package com.epsilon.mahjong.rules;

import java.util.Arrays;
import java.util.Objects;

/**
 * 局の始まりの卓。点は絶対席の順に並べ、渡された配列のコピーを持つ。
 *
 * <p>不変条件: 局の番号は 0 から西場の最後の局まで。本場と供託は 0 以上。点の数は席の数と同じ。Σ点 ＋ 1000 × 供託 = 席の数 × 配給点。
 * 破ると IllegalArgumentException。
 */
public record RoundStart(Variant variant, int round, int honba, int kyotaku, int[] scores) {

  public RoundStart {
    Objects.requireNonNull(variant, "variant");
    check(variant, round, honba, kyotaku, scores);
    scores = scores.clone();
  }

  /** 上の不変条件を確かめる。{@link TableState#set} からも呼ぶ。破っていれば IllegalArgumentException。 */
  static void check(Variant variant, int round, int honba, int kyotaku, int[] scores) {
    checkRanges(variant, round, honba, kyotaku, scores);
    long total = (long) Variant.DEPOSIT * kyotaku;
    for (int score : scores) {
      total += score;
    }
    if (total != (long) variant.seats() * variant.startPoints()) {
      throw new IllegalArgumentException(
          "Scores " + Arrays.toString(scores) + " and " + kyotaku + " deposits do not add up to the starting total");
    }
  }

  /**
   * 点と供託の和を除いて、不変条件を確かめる（局の番号の範囲、本場と供託が 0 以上、点の数が席の数と同じ）。もしもの卓を作る
   * {@link TableState#setScenario} からも呼ぶ。破っていれば IllegalArgumentException。
   */
  static void checkRanges(Variant variant, int round, int honba, int kyotaku, int[] scores) {
    if (round < 0 || round > variant.finalRound()) {
      throw new IllegalArgumentException("Round " + round + " is outside " + variant);
    }
    if (honba < 0 || kyotaku < 0) {
      throw new IllegalArgumentException("Honba and kyotaku must not be negative: " + honba + ", " + kyotaku);
    }
    if (scores.length != variant.seats()) {
      throw new IllegalArgumentException(variant + " needs " + variant.seats() + " scores, got " + scores.length);
    }
  }

  /** 半荘の始め（東 1 局 0 本場、供託なし、全員が配給点）。 */
  public static RoundStart first(Variant variant) {
    int[] scores = new int[variant.seats()];
    Arrays.fill(scores, variant.startPoints());
    return new RoundStart(variant, 0, 0, 0, scores);
  }

  public int score(int seat) {
    return scores[seat];
  }

  /** 点の配列のコピーを返す。 */
  @Override
  public int[] scores() {
    return scores.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof RoundStart start
        && variant == start.variant
        && round == start.round
        && honba == start.honba
        && kyotaku == start.kyotaku
        && Arrays.equals(scores, start.scores);
  }

  @Override
  public int hashCode() {
    return Objects.hash(variant, round, honba, kyotaku, Arrays.hashCode(scores));
  }

  @Override
  public String toString() {
    return "RoundStart[" + variant + " round=" + round + " honba=" + honba + " kyotaku=" + kyotaku
        + " scores=" + Arrays.toString(scores) + "]";
  }
}
