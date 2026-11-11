package com.epsilon.grp;

import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;

/**
 * GRP に渡す卓。局末の精算が済み、次局の配牌はまだの、半荘が続く卓を表す。キーや記録の値をそのまま持てるように、点の和は
 * ここでは確かめない。ネットワークに渡すときに、TableState の決まり（Σ点 ＋ 1000 × 供託 = 席の数 × 配給点）で確かめる。
 *
 * <p>局は 0 から西場の最後の局まで、本場と供託は 0 以上で、点は席の数だけあること。満たさなければ IllegalArgumentException を
 * 投げる。
 *
 * @param round 次局の番号（東 1 局が 0）
 * @param sticks 供託の本数
 * @param scores 起家を 0 とする絶対席の順の点。コピーを持つ
 */
public record GrpState(Variant variant, int round, int honba, int sticks, int[] scores) {

  public GrpState {
    Objects.requireNonNull(variant, "variant");
    if (round < 0 || round > variant.finalRound()) {
      throw new IllegalArgumentException("Round " + round + " is outside " + variant);
    }
    if (honba < 0 || sticks < 0) {
      throw new IllegalArgumentException("Honba and sticks must not be negative: " + honba + ", " + sticks);
    }
    if (scores.length != variant.seats()) {
      throw new IllegalArgumentException(variant + " needs " + variant.seats() + " scores, got " + scores.length);
    }
    scores = scores.clone();
  }

  /** 半荘が続く卓から作る。終局した卓では IllegalArgumentException を投げる。 */
  public static GrpState of(TableState table) {
    if (table.gameOver()) {
      throw new IllegalArgumentException("A finished game has no next hand: " + table);
    }
    int[] scores = new int[table.variant().seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = table.score(seat);
    }
    return new GrpState(table.variant(), table.round(), table.honba(), table.kyotaku(), scores);
  }

  /** キーから作る。そのルールのキーでなければ（{@code GrpKey.valid}）IllegalArgumentException を投げる。 */
  public static GrpState ofKey(Variant variant, long key) {
    if (!GrpKey.valid(variant, key)) {
      throw new IllegalArgumentException("Not a " + variant + " key: " + Long.toHexString(key));
    }
    int[] scores = new int[variant.seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = GrpKey.score(key, seat);
    }
    return new GrpState(variant, GrpKey.round(key), GrpKey.honba(key), GrpKey.sticks(key), scores);
  }

  /** 次局の親の絶対席を返す。 */
  public int dealer() {
    return variant.dealer(round);
  }

  public int score(int seat) {
    return scores[seat];
  }

  /** 点の高い順の順位（0 始まり）を返す。同点なら起家に近い席が上（Placement）。 */
  public int rank(int seat) {
    return Placement.rank(scores, seat);
  }

  /** 学習の行とメモに使うキーを返す。収まらなければ GrpKey.NONE を返す。 */
  public long key() {
    return GrpKey.of(this);
  }

  /**
   * この卓を、渡した TableState に写す。メモリは確保しない。ルールが違うときと、TableState の決まり（RoundStart と同じ。点の和も
   * 含む）を満たさないときは IllegalArgumentException を投げる。
   */
  public void copyInto(TableState table) {
    if (table.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: table " + table.variant() + ", state " + variant);
    }
    table.set(round, honba, sticks, scores);
  }

  /** 点のコピーを返す。 */
  @Override
  public int[] scores() {
    return scores.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof GrpState state
        && variant == state.variant
        && round == state.round
        && honba == state.honba
        && sticks == state.sticks
        && Arrays.equals(scores, state.scores);
  }

  @Override
  public int hashCode() {
    return Objects.hash(variant, round, honba, sticks, Arrays.hashCode(scores));
  }

  @Override
  public String toString() {
    return "GrpState[" + variant + " round=" + round + " honba=" + honba + " sticks=" + sticks
        + " scores=" + Arrays.toString(scores) + "]";
  }
}
