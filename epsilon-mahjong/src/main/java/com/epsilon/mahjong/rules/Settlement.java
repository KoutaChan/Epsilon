package com.epsilon.mahjong.rules;

import java.util.Arrays;
import java.util.Objects;

/**
 * 局の終わりに点の動きを精算する（天鳳）。精算はここだけで行い、点数は {@link ScoreMath} の式で求める。どのメソッドもメモリを確保しない。
 * 点の動きは絶対席の順の配列で、受け取りが正、払いが負。ツモ・ロン・荒牌の払いは配列に足し込むので、ダブロンや複数の流し満貫は
 * 続けて呼んで作る。包の席は放銃者に数えない。
 */
public final class Settlement {

  /** 流し満貫の払いは満貫のツモと同じ。 */
  private static final int NAGASHI_BASE = BaseClass.MANGAN.base();

  /** 包が付くのは大三元・大四喜を含む役満の和了だけなので、包のある和了の基本点はこの倍数になる。 */
  private static final int YAKUMAN = BaseClass.YAKUMAN.base();

  private Settlement() {}

  /**
   * 局の始めからの点の動きを書く。前の中身は捨てる。含めるのは、立直が成立した席の供託、和了・不聴罰符・流し満貫の払いと本場、
   * 和了のときに最初の和了者が受け取る場の供託（局の始めの供託 ＋ この局の立直）。点の動きの和は
   * 1000 × (局の始めの供託 − 次局へ残る供託)。
   *
   * @param variant ルール。卓と結果のルールと同じであること
   * @param start 局の始まりの卓
   * @param outcome 局の結果
   * @param delta 点の動きを書く先
   * @throws IllegalArgumentException ルールが合わないとき、または終局の卓を渡したとき
   */
  public static void apply(Variant variant, TableState start, RoundOutcome outcome, int[] delta) {
    if (start.variant() != variant || outcome.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: " + variant + ", table " + start.variant() + ", outcome " + outcome.variant());
    }
    if (start.gameOver()) {
      throw new IllegalArgumentException("No round starts from a finished game: " + start);
    }
    requireDelta(variant, delta);
    Arrays.fill(delta, 0);
    int riichiMask = outcome.riichiMask();
    for (int seat = 0; seat < delta.length; seat++) {
      if ((riichiMask & (1 << seat)) != 0) {
        delta[seat] -= Variant.DEPOSIT;
      }
    }
    int dealer = start.dealer();
    switch (outcome.kind()) {
      case TSUMO -> tsumo(variant, outcome.base(0), outcome.winner(0), dealer, start.honba(), outcome.pao(0), delta);
      case RON -> {
        for (int i = 0; i < outcome.winnerCount(); i++) {
          int honba = i == 0 ? start.honba() : 0;
          ron(variant, outcome.base(i), outcome.winner(i), outcome.loser(), dealer, honba, outcome.pao(i), delta);
        }
      }
      case EXHAUSTIVE -> exhaustive(variant, dealer, outcome.tenpaiMask(), outcome.nagashiMask(), delta);
      case ABORTIVE -> {
        // 立直の供託のほかは点が動かない。
      }
    }
    if (outcome.kind().win()) {
      delta[outcome.winner(0)] += Variant.DEPOSIT * deposits(start, outcome);
    }
  }

  /**
   * ツモ和了の払いを足し込む。和了者以外の各席が {@link ScoreMath#tsumo} の額と 100 × 本場を払い、和了者がその全てを受け取る。
   * 払うのは卓にいる席だけなので、三人打ちではいない席の分を誰も払わない（ツモ損）。包の席があれば、各席の額と本場の合計を包の席が
   * 1 人で払う（三人打ちではツモ損のままで、和了者以外の 2 席分の合計）。
   *
   * @param variant ルール
   * @param base 基本点
   * @param winner 和了者
   * @param dealer 親の席
   * @param honba 本場
   * @param pao 包の席。無ければ {@link RoundOutcome#NO_SEAT}
   * @param delta 点の動きを足し込む先
   */
  public static void tsumo(Variant variant, int base, int winner, int dealer, int honba, int pao, int[] delta) {
    int seats = variant.seats();
    Objects.checkIndex(winner, seats);
    Objects.checkIndex(dealer, seats);
    requirePao(variant, base, winner, pao);
    requireHonba(honba);
    requireDelta(variant, delta);
    boolean winnerIsDealer = winner == dealer;
    int bonus = Variant.HONBA * honba;
    int gain = 0;
    for (int seat = 0; seat < seats; seat++) {
      if (seat != winner) {
        int paid = ScoreMath.tsumo(base, winnerIsDealer, seat == dealer) + bonus;
        delta[pao == RoundOutcome.NO_SEAT ? seat : pao] -= paid;
        gain += paid;
      }
    }
    delta[winner] += gain;
  }

  /**
   * ロンの払いを足し込む。放銃者が {@link ScoreMath#ron} の額と、ロンの本場 1 本の額（{@link Variant#ronHonba}）× 本場を払う。
   * ダブロンの 2 人目以降は本場 0 で呼ぶ。包の席があれば、額を放銃者と包の席で折半し、本場は包の席が払う。包は役満だけなので、
   * 折半は割り切れる。
   *
   * @param variant ルール
   * @param base 基本点
   * @param winner 和了者
   * @param loser 放銃者
   * @param dealer 親の席
   * @param honba 本場
   * @param pao 包の席。無ければ {@link RoundOutcome#NO_SEAT}
   * @param delta 点の動きを足し込む先
   * @throws IllegalArgumentException 包の席が放銃者のとき（包の席が放銃したロンは、包の無いロンとして書く）
   */
  public static void ron(
      Variant variant, int base, int winner, int loser, int dealer, int honba, int pao, int[] delta) {
    int seats = variant.seats();
    Objects.checkIndex(winner, seats);
    Objects.checkIndex(loser, seats);
    Objects.checkIndex(dealer, seats);
    if (winner == loser) {
      throw new IllegalArgumentException("Seat " + winner + " cannot ron its own discard");
    }
    requirePao(variant, base, winner, pao);
    requireLoserNotLiable(loser, pao);
    requireHonba(honba);
    requireDelta(variant, delta);
    int points = ScoreMath.ron(base, winner == dealer);
    int bonus = variant.ronHonba() * honba;
    delta[winner] += points + bonus;
    if (pao == RoundOutcome.NO_SEAT) {
      delta[loser] -= points + bonus;
    } else {
      int half = points / 2;
      delta[loser] -= half;
      delta[pao] -= points - half + bonus;
    }
  }

  /**
   * 荒牌の流局の払いを足し込む。本場は付かない。流し満貫の席があれば、その席ごとに満貫のツモ（ツモ損を含む）と同じ額を受け取り、
   * 不聴罰符は払わない。無ければ、不聴罰符の総額を聴牌の席が等分して受け取り、不聴の席が等分して払う（全員が聴牌か全員が不聴なら
   * 点は動かない）。
   *
   * @param variant ルール
   * @param dealer 親の席
   * @param tenpaiMask 聴牌した席のマスク
   * @param nagashiMask 流し満貫になった席のマスク
   * @param delta 点の動きを足し込む先
   */
  public static void exhaustive(Variant variant, int dealer, int tenpaiMask, int nagashiMask, int[] delta) {
    int seats = variant.seats();
    Objects.checkIndex(dealer, seats);
    if (((tenpaiMask | nagashiMask) & ~variant.seatMask()) != 0) {
      throw new IllegalArgumentException("Masks " + Integer.toBinaryString(tenpaiMask) + " and "
          + Integer.toBinaryString(nagashiMask) + " have seats outside " + variant);
    }
    requireDelta(variant, delta);
    if (nagashiMask != 0) {
      for (int seat = 0; seat < seats; seat++) {
        if ((nagashiMask & (1 << seat)) != 0) {
          tsumo(variant, NAGASHI_BASE, seat, dealer, 0, RoundOutcome.NO_SEAT, delta);
        }
      }
      return;
    }
    int tenpai = Integer.bitCount(tenpaiMask);
    if (tenpai == 0 || tenpai == seats) {
      return;
    }
    int received = variant.notenTotal() / tenpai;
    int paid = variant.notenTotal() / (seats - tenpai);
    for (int seat = 0; seat < seats; seat++) {
      delta[seat] += (tenpaiMask & (1 << seat)) != 0 ? received : -paid;
    }
  }

  /** 局の終わりに場にある供託の本数（局の始めの供託 ＋ この局で成立した立直）。和了なら最初の和了者が受け取り、流局なら次局へ残る。 */
  static int deposits(TableState start, RoundOutcome outcome) {
    return start.kyotaku() + Integer.bitCount(outcome.riichiMask());
  }

  /**
   * 包の席を確かめる。包の席は無し（{@link RoundOutcome#NO_SEAT}）か、和了者以外の席。包のある和了は役満に限る。破っていれば
   * IllegalArgumentException。
   */
  static void requirePao(Variant variant, int base, int winner, int pao) {
    if (pao == RoundOutcome.NO_SEAT) {
      return;
    }
    Objects.checkIndex(pao, variant.seats());
    if (pao == winner) {
      throw new IllegalArgumentException("Seat " + winner + " cannot be liable for its own win");
    }
    if (base % YAKUMAN != 0) {
      throw new IllegalArgumentException("Pao needs a yakuman, but the base is " + base);
    }
  }

  /** ロンで、包の席が放銃者でないことを確かめる。包の席が放銃したロンは、包の無いロンとして書く。 */
  static void requireLoserNotLiable(int loser, int pao) {
    if (pao != RoundOutcome.NO_SEAT && pao == loser) {
      throw new IllegalArgumentException("Seat " + loser + " deals in, so it cannot also be liable for the win");
    }
  }

  static void requireDelta(Variant variant, int[] delta) {
    if (delta.length != variant.seats()) {
      throw new IllegalArgumentException(variant + " needs " + variant.seats() + " deltas, got " + delta.length);
    }
  }

  private static void requireHonba(int honba) {
    if (honba < 0) {
      throw new IllegalArgumentException("Honba must not be negative: " + honba);
    }
  }
}
